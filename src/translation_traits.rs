//! Traits for translation operators.

use crate::field_traits::Field;
use ndarray::{Array2, ArrayView2};
use ndarray_linalg::Scalar;

pub trait Operator {
    type A: Scalar;

    fn apply(&self, vec: ArrayView2<Self::A>) -> ArrayView2<Self::A>;
}

// pub trait TranslationOperator {
//     type A: Scalar;
//     type In: Field<A = Self::A>;
//     type Out: Field<A = Self::A>;

//     fn translate(&self, source: &Self::In, target: &mut Self::Out);
// }

pub trait NumericalTranslationOperator {
    type A: Scalar;
    type Out: Field<A = Self::A>;

    fn approximate_in_target(&self, source: &dyn Field<A=Self::A>, target: &Self::Out) -> Array2<Self::A>;
}

pub trait FieldWithTranslationOperator: Field {
    fn translate_from(&mut self, other: &dyn Field<A=Self::A>);
}
