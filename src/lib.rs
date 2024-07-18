pub mod functions;
pub mod layer;
pub mod nn;

use layer::Linear;
use ndarray::{ArrayBase, ArrayD, Dim, IxDyn, IxDynImpl, OwnedRepr};
use nn::mlp;
use pyo3::{pymodule, types::PyModule, PyResult, Python};
use rand::Rng;

pub type Ndarray = ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>>;

fn random_array(n: usize, m: usize) -> Ndarray {
    let mut rng = rand::thread_rng();
    let mut array = ArrayD::zeros(IxDyn(&[n, m]));
    for i in 0..n {
        for j in 0..m {
            array[[i, j]] = rng.gen::<f64>();
        }
    }
    array
}

#[pymodule]
#[pyo3(name = "nnet")]
pub fn nnet(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Linear>()?;
    m.add_class::<mlp>()?;

    Ok(())
}
