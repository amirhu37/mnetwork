#[allow(unused_variables)]
use numpy::{IntoPyArray, PyArrayDyn};
use pyo3::{Py, PyAny, PyResult, Python};
// use ActiovationFunction::*;
use ndarray::ArrayD;
use pyo3::prelude::*;

use crate::NpNdarray;

#[allow(dead_code)]
// trait ActiovationFunction {
//     fn dfx(y: fn(f64)) -> f64;
//     fn fx(x: fn(f64)) -> f64;
// }

// #[derive(FromPyObject)]
#[pyclass]
pub struct ActiovationFunction {
    fx: fn(Py<PyAny>) -> Py<PyAny>,
    df: fn(Py<PyAny>) -> Py<PyAny>,
}

#[pyfunction]
pub fn sigmoid(x: Bound<PyAny>) -> NpNdarray {
    let func = |value: f64| 1.0 / (1.0 + (-value).exp());
    let y: NpNdarray = apply_func(&x, func).unwrap();
    y
}
#[pyfunction]
pub fn tanh(x: Bound<PyAny>) -> NpNdarray {
    let func = |value: f64| (2.0 / (1.0 + (-value).exp())).tanh();
    let y: NpNdarray = apply_func(&x, func).unwrap();
    y
}
#[pyfunction]
pub fn relu(x: Bound<PyAny>) -> NpNdarray {
    let func = |value: f64| if value > 0.0 { value } else { 0.0 };
    let y: NpNdarray = apply_func(&x, func).unwrap();
    y
}
#[pyfunction]
pub fn softmax(x: Bound<PyAny>) -> NpNdarray {
    let func = |value: f64| 1.0 / (1.0 + (-value).exp());
    let y: NpNdarray = apply_func(&x, func).unwrap();
    y
}

#[allow(dead_code)]
pub fn apply_func(
    input: &Bound<PyAny>,
    func: impl Fn(f64) -> f64,
) -> PyResult<Py<PyArrayDyn<f64>>> {
    // Convert the input to a NumPy array
    let input_array: &PyArrayDyn<f64> = input.extract()?;
    let res: NpNdarray = Python::with_gil(|py| {
        // Convert the NumPy array to an ndarray ArrayD
        let input_array: numpy::PyReadonlyArray<f64, ndarray::Dim<ndarray::IxDynImpl>> =
            input_array.readonly();
        let input_array: ndarray::ArrayBase<
            ndarray::ViewRepr<&f64>,
            ndarray::Dim<ndarray::IxDynImpl>,
        > = input_array.as_array();

        // Apply the function to each element of the array
        let result: ArrayD<f64> = input_array.mapv(func);
        result.into_pyarray_bound(py).to_owned().into()
    });
    // Convert the result back to a NumPy array and return
    Ok(res)
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
