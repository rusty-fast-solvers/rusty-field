//! Implementation of FMM translation operators

pub mod field_traits;
pub mod laplace_translation_traits;
pub mod numerical_translation_operators;
pub mod sources;

pub mod laplace_field_from_sources;

pub mod c_api;

use ndarray::{Array2, ArrayView2};
use ndarray_linalg::Scalar;

pub trait Operator {
    type A: Scalar;

    fn apply(&self, vec: ArrayView2<Self::A>) -> Array2<Self::A>;
}
