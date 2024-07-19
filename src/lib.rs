pub mod functions;
pub mod layer;
pub mod nn;

use nn::Mlp;
use layer::Linear;


use ndarray::{ArrayBase, ArrayD, Dim, IxDyn, IxDynImpl, OwnedRepr};
use pyo3::{ffi::PyObject, pymodule, types::PyModule, PyResult, Python};
use rand::Rng;

pub type darrayf64 = ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>>;

fn random_array(n: usize, m: usize) -> darrayf64 {
    let mut rng = rand::thread_rng();
    let mut array: ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>> = ArrayD::zeros(IxDyn(&[n, m]));
    for i in 0..n {
        for j in 0..m {
            array[[i, j]] = rng.gen::<f64>();
        }
    }
    array
}



#[pymodule]
#[pyo3(name = "nnet")]
pub fn nnet_module(_py: Python, m: &PyModule) -> PyResult<()> {
    add_class!(m, Linear, Mlp);
    // m.add_class::<Linear>()?;
    // m.add_class::<Mlp>()?;

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
