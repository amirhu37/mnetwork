use numpy::{IntoPyArray, PyArrayDyn};
use pyo3::{Py, PyAny, PyResult, Python};
// use ActiovationFunction::*;

use ndarray::ArrayD;

#[allow(dead_code)]
trait ActiovationFunction {
    fn dfx(y: fn(f64)) -> f64;
    fn fx(x: fn(f64)) -> f64;
}

struct Sigmoid;
impl ActiovationFunction for Sigmoid {
    #[allow(unused_variables)]

    fn fx(x: fn(f64)) -> f64 {
        todo!()
    }
    #[allow(unused_variables)]

    fn dfx(y: fn(f64)) -> f64 {
        todo!()
    }
}

#[allow(dead_code)]
fn apply_func(
    py: Python,
    input: &PyAny,
    func: impl Fn(f64) -> f64,
) -> PyResult<Py<PyArrayDyn<f64>>> {
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
#[allow(dead_code)]
fn funcs() {
    #[allow(unused_variables)]
    let f_sigmoid = |x: f64| 1.0 / (1.0 + (-x).exp());
    // derivative of sigmoid
    #[allow(unused_variables)]
    let df_dsigmoid = |y: f64| y * (1.0 - y);

    // sfotmax and its derivative
    // let f_softmax = |x : f64| { x.exp() / (x.exp().sum() )};
    #[allow(unused_variables)]
    let df_dsoftmax = |y: f64| y * (1.0 - y);
}
