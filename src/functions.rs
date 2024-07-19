
use numpy::{IntoPyArray, IxDyn, PyArrayDyn};
use pyo3::{ffi::PyObject, pyclass, pymethods, types::PyModule, Py, PyAny, PyResult, Python};
// use ActiovationFunction::*;

use ndarray::{ArrayBase, ArrayD, ArrayView1, ArrayViewD, Dim, IxDynImpl, OwnedRepr};

use crate::darrayf64;
// #[pyclass]
trait ActiovationFunction  {
    fn fx(x : fn(f64))  -> f64;
    fn dfx(y : fn(f64)) -> f64 ;
}

struct Sigmoid;
impl ActiovationFunction for Sigmoid {
    fn fx(x : fn(f64))  -> f64 {
        todo!()}
        fn dfx(y : fn(f64)) -> f64 {todo!()}    
}

fn apply_func(py: Python, input: &PyAny, func : impl Fn(f64) -> f64) -> PyResult<Py<PyArrayDyn<f64>>> {
    // Convert the input to a NumPy array
    let input_array: &PyArrayDyn<f64> = input.extract()?;

    // Convert the NumPy array to an ndarray ArrayD
    let input_array = input_array.readonly();
    let input_array = input_array.as_array();

    // Apply the function to each element of the array
    let result: ArrayD<f64> = input_array.mapv(func);

    // Convert the result back to a NumPy array and return
    Ok(result.into_pyarray(py).to_owned())
}
fn funcs(){
let f_sigmoid = |x : f64| { 1.0 / (1.0 + (-x).exp())};
// derivative of sigmoid
let df_dsigmoid = |y : f64| { y * (1.0 - y )};

// sfotmax and its derivative
// let f_softmax = |x : f64| { x.exp() / (x.exp().sum() )};
let df_dsoftmax = |y : f64| { y * (1.0 -y) };}


