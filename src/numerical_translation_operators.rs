//! Definition of numerical translation operators.

use crate::Operator;
use ndarray::{Array2, ArrayView2};
use ndarray_linalg::Inverse;
use ndarray_linalg::Scalar;
use rusty_compression::prelude::*;
use rusty_compression::two_sided_interp_decomp::TwoSidedIDResult;

pub struct NumericalTranslationOperator<A: Scalar> {
    approximation_operator: Box<dyn Operator<A = A>>,
    evaluation_operator: Box<dyn Operator<A = A>>,
}

pub struct FieldApproximationOperator<A: Scalar> {
    pseudo_inverse: Array2<A>,
}

pub struct NumericalEvaluationOperator<A: Scalar> {
    interp_decomp: TwoSidedIDResult<A>,
}

impl<A: Scalar> Operator for FieldApproximationOperator<A> {
    type A = A;
    fn apply(&self, vec: ArrayView2<Self::A>) -> Array2<Self::A> {
        self.pseudo_inverse.dot(&vec)
    }
}

impl<A: Scalar> Operator for NumericalEvaluationOperator<A> {
    type A = A;
    fn apply(&self, vec: ArrayView2<Self::A>) -> Array2<Self::A> {
        self.interp_decomp.apply_matrix(&vec)
    }
}

impl<A: Scalar> FieldApproximationOperator<A> {
    pub fn new<F>(
        evaluator: &F,
        sources: ArrayView2<A::Real>,
        check_potentials: ArrayView2<A::Real>,
    ) -> FieldApproximationOperator<A>
    where
        F: Fn(ArrayView2<A::Real>, ArrayView2<A::Real>) -> Array2<A>,
    {
        let mat = evaluator(sources, check_potentials);
        let compressed = mat
            .compute_svd()
            .unwrap()
            .compress(CompressionType::ADAPTIVE(1E-14))
            .unwrap()
            .to_qr()
            .unwrap();
        FieldApproximationOperator::<A> {
            pseudo_inverse: compressed
                .r
                .inv()
                .unwrap()
                .dot(&compressed.q.t().mapv(|val| val.conj())),
        }
    }
}
