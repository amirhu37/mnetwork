
use numpy::{IntoPyArray, PyArrayDyn};
use pyo3::{prelude::*, types::PyDict};
use crate::random_array;
use ndarray::{ArrayBase, ArrayD, Dim, IxDyn, IxDynImpl, OwnedRepr};

pub type Ndarray = ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>>;
pub type NDArray2 = ArrayBase<OwnedRepr<Vec<Vec<f64>>>, Dim<[usize; 2]>>;
pub type Object = Py<PyArrayDyn<f64>>;




#[derive(Debug)]
#[pyclass(name = "Linear", unsendable, subclass, sequence, dict, get_all, set_all)]
#[pyo3(text_signature = "(in_features : u16 , out_features : u16,  is_bias: bool = True)")]
pub struct Linear {
    pub weights: PyObject,
    pub bias: PyObject,
    pub is_bias: bool,
    shape: (u16, u16),
}

#[pymethods]
impl Linear {
    #[new]
    #[pyo3(signature = (in_features , out_features, is_bias))]
    pub fn new<'py>(in_features: u16, out_features: u16, is_bias: bool) -> PyResult<Self> {
        Python::with_gil(|py| {
            let random_weight: Ndarray = random_array(in_features.into(), out_features.into());
            let random_bias: Ndarray = if is_bias {
                random_array(out_features.into(), 1)
            } else {
                ArrayD::zeros(IxDyn(&[out_features.into(), 1]))
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
            parameter.set_item("weights", self.weights.as_ref(py)).unwrap();
            parameter.set_item("bias", self.bias.as_ref(py)).unwrap();
            parameter.to_object(py)
        })
    }

    fn __str__(&self) -> String {
        format!(
            "Linear(in = {},out = {}, params={}) ",
            self.shape.0,
            self.shape.1,
            self.shape.0 * self.shape.1 + self.shape.1
        )
    }

    fn __repr__(&self) -> String {
        self.__str__()}

    fn __dict__<'py>(&self)-> PyResult<PyObject> {
        let __dict__= Python::with_gil(|py| {
            let parameter = PyDict::new(py);
            parameter.set_item("weights", self.weights.as_ref(py)).unwrap();
            parameter.set_item("bias", self.bias.as_ref(py)).unwrap();
            parameter.to_object(py)});
        Ok(__dict__)}

    #[getter]
    fn __doc__(&self) -> String{
        format!("
        linear Layer. linear Layer. linear Layer. 
        ")
    }
}