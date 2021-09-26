//! Definition of a Laplace field from sources.

use crate::field_traits::Field;
use ndarray::{Array2, Array4, ArrayView2, Axis};
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
            /// For scalar fields the returned array is of shape `(ncharge_vecs, ntargets, 1, 1)`
            /// if only field values are
            /// requested or  of shape `(ncharge_vecs, ntargets, 1, 4)` if function values and gradients
            /// are requested. The value `result[i][j][0][0]` contains the potential sum evaluated at
            /// the jth target, using the ith charge vector. The values `result[i][j][0][k]` for k=1,..,3
            /// contain the corresponding gradient in the x, y, and z coordinate direction.
            ///
            /// For vectorial fields the returned array is of shape `(ncharge_vecs, ntargets, dim, 1)`
            /// if only field values are requested and of shape `(ncharge_vecs, ntargets, dim, 4)` if
            /// field values and gradients are requested. For example, if the vectorial dimension is 3, then
            /// `result[i][j][2][0]` contains value of the second component of the potential sum evaluated at the jth target
            /// point with the ith target vector. The value `result[i][j][2][3]` contains correspondingly the third component
            /// of the derivative vector with respect to the second component of the field. In other words, the 3x3 matrix
            /// `result[i][j][:][1:]` is the derivative matrix of the vectorial field with 3 components.
            fn evaluate(&self, points: ArrayView2<$A>, eval_mode: EvalMode) -> Array4<$A> {
                let evaluator = make_laplace_evaluator(self.sources.view(), points.view());
                // We insert an ampty Axis since rusty-green-kernel currently only does scalar fields and therefore
                // only returns an Array3 type and not an Array4 type.
                evaluator
                    .evaluate(self.coefficients.t(), &eval_mode, ThreadingType::Serial)
                    .insert_axis(Axis(2))
            }

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

            /// Return the space dimension of the field.
            fn space_dimension(&self) -> usize {
                1
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
