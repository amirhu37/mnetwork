use crate::random_array;
use ndarray::{ArrayBase, ArrayD, Dim, IxDyn, IxDynImpl, OwnedRepr};
use numpy::{IntoPyArray, PyArrayDyn};
use pyo3::{prelude::*, types::{IntoPyDict, PyDict, PyList}};

pub type Ndarray = ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>>;
pub type NDArray2 = ArrayBase<OwnedRepr<Vec<Vec<f64>>>, Dim<[usize; 2]>>;
pub type Object = Py<PyArrayDyn<f64>>;

#[pyclass(
    module = "nn",
    name = "Linear",
    unsendable,
    subclass,
    sequence,
    dict,
    get_all,
    set_all
)]
// #[pyo3( text_signature = "(in_features : u16 , out_features : u16,  is_bias: bool = True)")]
pub struct Linear {
    pub weights: PyObject,
    pub bias: PyObject,
    pub is_bias: bool,
    shape: (u16, u16),
}

#[pymethods]
impl Linear {
    #[new]
    #[pyo3(signature = (in_features , out_features, is_bias = true))]
    pub fn __new__<'py>(
        in_features: u16,
        out_features: u16,
        is_bias: Option<bool>,
    ) -> PyResult<Self> {
        let is_bias = match is_bias {
            Some(is_bias) => is_bias,
            None => false,
        };
        Python::with_gil(|py| {
            let random_weight: Ndarray = random_array(in_features.into(), out_features.into());
            let random_bias: Ndarray = if is_bias {
                random_array(1, out_features.into())
            } else {
                ArrayD::zeros(IxDyn(&[1, out_features.into()]))
            };
            Ok(Self {
                weights: random_weight.into_pyarray(py).to_owned().into(),
                bias: random_bias.into_pyarray(py).to_owned().into(),
                is_bias,
                shape: (in_features, out_features),
            })
        })
    }

    #[pyo3(text_signature = "($cls )")]
    pub fn parameters<'py>(&mut self) -> PyObject {
        Python::with_gil(|py| {
            let parameter = PyDict::new(py);
            parameter
                .set_item("weights", self.weights.as_ref(py))
                .unwrap();
            parameter.set_item("bias", self.bias.as_ref(py)).unwrap();
            parameter.to_object(py)
        })
    }

    fn __call__(&self, py: Python<'_>, value: &PyAny) -> PyResult<PyObject> {
        // Convert value to PyArrayDyn
        let value: &PyArrayDyn<f64> = value.extract()?;

        // Get NumPy module
        let np = PyModule::import(py, "numpy")?;
        // Transpose value
        // let transpose = np.getattr("transpose")?.call1((self.weights.clone(),))?;
        // Perform dot product: self.weights \dot value
        let weight = self.weights.as_ref(py);
        let bias = self.bias.as_ref(py);

        // println!("{}", shape.call1(value)? );
        let dot_result = np
            .getattr("dot")
            .expect("Invalid import dot")
            .call1((value, weight))
            .expect("dot prod error");

        // Add bias: (self.weights \dot value) + self.bias
        let result = np
            .getattr("add")
            .expect("error import add")
            .call1((dot_result, bias))
            .expect("summation error");

        Ok(result.to_object(py))
    }

    fn __str__(&self) -> String {
        let bias_shape = if !self.is_bias { 0 } else { self.shape.1 };
        format!(
            "Linear(in = {},out = {}, params={}) ",
            self.shape.0,
            self.shape.1,
            self.shape.0 * self.shape.1 + bias_shape
        )
    }

    fn __repr__(slf: &Bound<Self>) -> PyResult<String> {
        // This is the equivalent of `self.__class__.__name__` in Python.
        let class_name = slf.get_type().qualname()?;
        Ok(format!("{}(weights : {} , bias : {})", class_name, slf.borrow().weights, slf.borrow().bias))
    }

    #[getter]
    fn __doc__(&self) -> String {
        format!(
            "
        linear Layer. linear Layer. linear Layer. 
        "
        )
    }
    fn __iter__(slf : &Bound<Self>)-> PyObject{
        let class_name = slf.get_type().qualname().unwrap();
        Python::with_gil(|py| {
            // let list = PyList::new_bound(py, slf.borrow().weights.clone().to_object(py));
            let locals = [ ("weighs", slf.borrow().weights.clone()), 
            ("self", class_name.to_object(py)),
            ("bias", slf.borrow().bias.clone()) ].into_py_dict_bound(py);
            let result = py.eval_bound("list(self.__dict__.values())", None, Some(&locals)).unwrap().unbind();
            let py_obj: PyObject = result.downcast_bound(py).unwrap().clone().unbind();
            py_obj
        })
    }
}
