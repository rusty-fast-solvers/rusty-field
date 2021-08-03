//! Traits for translation operators.

use ndarray::ArrayView2;
use ndarray_linalg::Scalar;

pub trait Operator {
    type A: Scalar;

    fn apply(&self, vec: ArrayView2<Self::A>) -> ArrayView2<Self::A>;

}
