//! A trait for modifying a field description in an FMM container.

use ndarray::{Array2, ArrayBase, ArrayView2, Data, Ix2};
use ndarray_linalg::Scalar;

/// This enum defines the Evaluation Mode.
pub enum EvalMode {
    /// Only evaluate Green's function values.
    Value,
    /// Evaluate values and derivatives.
    ValueGrad,
}

pub trait Field {
    type A: Scalar;

    // Evaluate a field.
    //
    // # Arguments
    // * `points`:  A (3, N) array of N points at which to evaluate the field.
    // * `eval_mode`: Evaluate either only field values or values and gradient.
    //
    // If `eval_mode` is `Value` then the function returns a (1, N) array of
    // field values at the given points. If `eval_mode` is `ValueGrad` then
    // the returned array has dimension (4, N), where the second to last value
    // in the jth column is the gradient of the field evaluated at the point with
    // index j.
    fn evaluate(&self, points: ArrayView2<Self::A>, eval_mode: EvalMode) -> Array2<Self::A>;

    // Return the coefficients that describe the field.
    fn coefficients(&self) -> Array2<Self::A>;


    // Get the index of the field.
    fn index(&self) -> usize;
}

pub trait ModifiableField: Field {

    // Update the field coefficients.
    fn update<S: Data<Elem = Self::A>>(&mut self, coefficients: ArrayBase<S, Ix2>);

}

