//! Traits for translation operators.

use ndarray::Array1;
use ndarray_linalg::Scalar;

pub trait M2L {
    type A: Scalar;

    // Implementation of an m2l operation.
    //
    // The method returns the coefficients of the field
    // translated to the box with index `target_index`.
    fn m2l(&self, target_index: usize) -> Array1<Self::A>;

}
