//! Definition of a Laplace field from sources.

use crate::field_traits::Field;
use ndarray::{Array2, Array3, ArrayView2, Axis};
use rusty_base::{EvalMode, RealType, ThreadingType};
use rusty_green_kernel;
use rusty_green_kernel::make_laplace_evaluator;
use rusty_green_kernel::RealDirectEvaluator;

pub struct LaplaceFieldFromSources<A: RealType> {
    sources: Array2<A>,
    coefficients: Array2<A>,
    index: usize,
}

impl<A: RealType> LaplaceFieldFromSources<A> {
    pub fn new(
        sources: ArrayView2<A>,
        coefficients: ArrayView2<A>,
        index: usize,
    ) -> LaplaceFieldFromSources<A> {
        LaplaceFieldFromSources {
            sources: sources.to_owned(),
            coefficients: coefficients.to_owned(),
            index,
        }
    }
}

macro_rules! laplace_field_from_source_impl {
    ($A:ty) => {
        impl Field for LaplaceFieldFromSources<$A> {
            type A = $A;

            /// Evaluate a field.
            ///
            /// # Arguments
            /// * `points`:  A (3, N) array of N points at which to evaluate the field.
            /// * `eval_mode`: Evaluate either only field values or values and gradient.
            ///
            /// If `eval_mode` is `Value` then the function returns a (1, N) array of
            /// field values at the given points. If `eval_mode` is `ValueGrad` then
            /// the returned array has dimension (4, N), where the second to last value
            /// in the jth column is the gradient of the field evaluated at the point with
            /// index j.
            fn evaluate(&self, points: ArrayView2<$A>, eval_mode: EvalMode) -> Array3<$A> {
                let evaluator = make_laplace_evaluator(self.sources.view(), points.view());
                evaluator.evaluate(
                    self.coefficients.t(),
                    &eval_mode,
                    ThreadingType::Serial,
                )
            }

            /// Return the coefficients that describe the field.
            fn coefficients(&self) -> Array2<Self::A> {
                self.coefficients.clone()
            }

            /// Get the index of the field.
            fn index(&self) -> usize {
                self.index
            }

            /// Return the number of coefficient vectors.
            fn number_of_coefficient_vectors(&self) -> usize {
                self.coefficients.len_of(Axis(1))
            }

            /// Return the required number of coefficients.
            fn coefficient_dimension(&self) -> usize {
                self.coefficients.len_of(Axis(0))
            }

            /// Update the field coefficients.
            fn update_coefficients(&mut self, coefficients: ArrayView2<$A>) {
                let tmp = self.coefficients.clone() + coefficients;
                self.coefficients.assign(&tmp)
            }

            /// Set coefficients to the given coefficient vectors.
            fn set_coefficients(&mut self, coefficients: ArrayView2<$A>) {
                self.coefficients = coefficients.to_owned()
            }
        }
    };
}

laplace_field_from_source_impl!(f32);
laplace_field_from_source_impl!(f64);
