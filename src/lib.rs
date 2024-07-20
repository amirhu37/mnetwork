pub mod functions;
pub mod layer;
pub mod nn;

use std::ops::Deref;

use layer::Linear;
use nn::Mlp;

const MODULE: &str = "nnet";

use ndarray::{ArrayBase, ArrayD, Dim, IxDyn, IxDynImpl, OwnedRepr};
use pyo3::{
    pymodule,
    types::{IntoPyDict, PyAnyMethods, PyDict, PyModule, PyString},
    IntoPy, Py, PyObject, PyResult, Python, ToPyObject,
};
use rand::Rng;

pub type Darrayf64 = ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>>;

fn random_array(n: usize, m: usize) -> Darrayf64 {
    let mut rng = rand::thread_rng();
    let mut array: ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>> = ArrayD::zeros(IxDyn(&[n, m]));
    for i in 0..n {
        for j in 0..m {
            array[[i, j]] = rng.gen::<f64>();
        }
    }
    array
}
use pyo3::Bound;

pub fn _py_run(value: &PyObject, command: &str) -> PyResult<Py<PyDict>> {
    Python::with_gil(|py| {
        let locals = [("value", value)].into_py_dict_bound(py);
        let result = py.eval_bound(command, None, Some(&locals))?.unbind();
        let py_dict = result.downcast_bound(py).unwrap().clone().unbind();
        Ok(py_dict.into())
    })
}

#[pymodule]
#[pyo3(name = "nnet")]
pub fn nnet_module(_py: Python, m: &PyModule) -> PyResult<()> {
    add_class!(m, Linear, Mlp);

    // Optionally, set the __name__ attribute if needed
    let class = m.getattr("Mlp")?;
    class.setattr("__name__", PyString::new(_py, "MyClass"))?;

    Ok(())
}

#[macro_export]
macro_rules! add_class {
    ($module : ident , $($class : ty), +) => {
        $(
            $module.add_class::<$class>()?;
        )+

    };
}
