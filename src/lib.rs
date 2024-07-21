/// interduce file and modules
pub mod functions;
pub mod layer;
pub mod linear;
pub mod loss;
pub mod neuaral;
pub mod optimizers;
pub mod tools;

/// import files and modules
use functions::*;
use layer::Layers;
use linear::Linear;
use neuaral::Neuaral;

use ndarray::{ArrayBase, ArrayD, Dim, IxDyn, IxDynImpl, OwnedRepr};
#[allow(unused_imports)]
use pyo3::Bound;
use pyo3::*;
use pyo3::{
    pymodule,
    types::{IntoPyDict, PyDict, PyModule},
    Py, PyObject, PyResult, Python,
};
use rand::Rng;

pub type Darrayf64 = ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>>;

#[allow(non_camel_case_types)]
pub type np_ndarray = Py<numpy::PyArray<f64, ndarray::Dim<ndarray::IxDynImpl>>>;

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

pub fn _py_run(value: &PyObject, command: &str) -> PyResult<Py<PyDict>> {
    Python::with_gil(|py| {
        let locals = [("value", value)].into_py_dict_bound(py);
        let result = py.eval_bound(command, None, Some(&locals))?.unbind();
        let py_dict = result.downcast_bound(py).unwrap().clone().unbind();
        Ok(py_dict.into())
    })
}

#[macro_export]
macro_rules! add_class {
    ($module : ident , $($class : ty), +) => {
        $(
            $module.add_class::<$class>()?;
        )+

    };
}
macro_rules! add_function {
    ($module : ident , $($function : ident), +) => {
        $(
           $module.add_wrapped(wrap_pyfunction!($function))?;
        )+
    };
}
#[pymodule]
#[pyo3(name = "nnet")]
pub fn nnet(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    add_class!(m, Linear, Neuaral, Layers, ActiovationFunction);
    // add functions
    add_function!(m, sigmoid);
    // add_function!(m, tanh);
    // add_function!(m, relu);
    // add_function!(m, softmax);
    // add_function!(m, cross_entropy);
    // add_function!(m, mse);
    Ok(())
}
