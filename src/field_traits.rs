//! A trait for representing fields.

use ndarray::{Array2, Array3, ArrayView2};
use ndarray_linalg::Scalar;
use rusty_base::EvalMode;

pub trait Field {
    type A: Scalar;

    /// Evaluate a field.
    ///
    /// # Arguments
    /// * `points`:  A (3, N) array of N points at which to evaluate the field.
    /// * `eval_mode`: Evaluate either only field values or values and gradient.
    ///
    /// The returned array is of shape `(ncharge_vecs, ntargets, 1)` if only Greens fct. values are
    /// requested or  of shape `(ncharge_vecs, ntargets, 4)` if function values and gradients
    /// are requested. The value `result[i][j][0]` contains the potential sum evaluated at
    /// the jth target, using the ith charge vector. The values `result[i][j][k]` for k=1,..,3
    /// contain the corresponding gradient in the x, y, and z coordinate direction.
    fn evaluate(
        &self,
        points: ArrayView2<<<Self as Field>::A as Scalar>::Real>,
        eval_mode: EvalMode,
    ) -> Array3<Self::A>;

    // Return the coefficients that describe the field.
    // A field may have multiple coefficent vectors. Each row
    // corresponds to one set of coefficients.
    fn coefficients(&self) -> Array2<Self::A>;

    // Get the index of the field.
    fn index(&self) -> usize;

    // Return the number of coefficient vectors.
    fn number_of_coefficient_vectors(&self) -> usize;

    // Return the required number of coefficients.
    fn coefficient_dimension(&self) -> usize;

    // Update the field coefficients by adding the new coefficient
    // vector to the stored one.
    fn update_coefficients(&mut self, coefficients: ArrayView2<Self::A>);

    // Set coefficients to the given coefficient vectors.
    fn set_coefficients(&mut self, coefficients: ArrayView2<Self::A>);
}
