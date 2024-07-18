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
