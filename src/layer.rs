use ndarray::{ArrayBase, Dim, IxDynImpl, OwnedRepr};
use numpy::PyArrayDyn;
use pyo3::{
    prelude::*,
    types::{IntoPyDict, PyDict, PyNone, PyTuple},
};

use crate::add_class;

pub type Ndarray = ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>>;
pub type NDArray2 = ArrayBase<OwnedRepr<Vec<Vec<f64>>>, Dim<[usize; 2]>>;
pub type Object = Py<PyArrayDyn<f64>>;

#[pyclass(
    module = "nn",
    name = "Layer",
    unsendable,
    subclass,
    sequence,
    dict,
    // get_all,
    // set_all
)]
pub struct Layers;
//  {
//     pub args:   Py<PyTuple>,
//     pub kwargs: Py<PyDict>  ,
// }

#[pymethods]
impl Layers {
    #[new]
    #[pyo3(signature = (*args, **kwargs)
        ,text_signature = "(*args = None, **kwargs = None )"
)]
    pub fn __new__(py: Python, args: &Bound<'_, PyAny>, kwargs: Option<&Bound<'_, PyAny>>) -> Self {
        Layers
    }

    pub fn __call__(&self, _py: Python, value: &Bound<PyAny>) -> PyResult<PyObject> {
        // Convert value to PyArrayDyn
        let value: &PyArrayDyn<f64> = value.extract()?;
        Ok(value.into())
    }

    fn __str__(&self) -> String {
        "Layers instance".to_string()
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("Layers()"))
    }

    #[getter]
    fn __doc__(&self) -> String {
        "
        linear Layer. linear Layer. linear Layer.
        "
        .to_string()
    }
}


