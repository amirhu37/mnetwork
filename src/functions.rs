#[derive(Debug)]
pub enum ActiovationFunction {
    Sigmoid {
        function: fn(f64),
        derivation: fn(f64),
    },
    ReLU {
        function: fn(f64),
        derivation: fn(f64),
    },
    Tanh {
        function: fn(f64),
        derivation: fn(f64),
    },
    Softmax {
        function: fn(f64),
        derivation: fn(f64),
    },
}

use numpy::{IxDyn, PyArrayDyn};
use pyo3::{ffi::PyObject, pyclass, pymethods, types::PyModule, PyAny, Python};
use ActiovationFunction::*;

use ndarray::{ArrayBase, ArrayD, Dim, IxDynImpl, OwnedRepr};

use crate::darrayf64;
// #[pyclass]
struct Sigmoid{
    function: fn(f64),
    derivation: fn(f64),
}

// #[pymethods]
impl Sigmoid {
    // #[new]
    // #[staticmethod]
    fn __new__() -> Self {
        let f = |x : f64| { 1.0 / (1.0 + (-x).exp())};
        let df = |x: f64| { f(x) * (1.0 - f(x))};
todo!()
}

}

// static CODE = r"#

// #";

fn sigmoid(x :PyAny){
    Python::with_gil(|py| {
        let value: &PyArrayDyn<f64> = x.extract().unwrap();

        const CODE: &str = r"#from numpy import shape
        function = lambda x : shape(x)";
        let module = PyModule::from_code(py, CODE, "", "").unwrap();
        let fun = module.getattr("function").unwrap();
        let args = ("hello",);
        let result = fun.call1(args).unwrap();
        // let s = (1,2);
        // let shape = result.extract::<&(u16, u16)>().unwrap();
        // assert_eq!(result.extract::<&str>().unwrap(), "called with args");
        // let np = PyModule::import(py, "numpy").unwrap();
        // let shape = np.getattr("shape").unwrap().  call1::<u16>(value).unwrap();
        let mut array: darrayf64 = ArrayD::zeros(IxDyn(&[2, 3]));
    
    })

}


