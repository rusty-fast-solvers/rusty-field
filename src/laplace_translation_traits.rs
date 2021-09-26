//! Laplace translation operators

use ndarray::ArrayView2;
use ndarray_linalg::Scalar;
use rusty_compression::prelude::randomized_adaptive_qr;
use rusty_compression::two_sided_interp_decomp::TwoSidedIDResult;

pub trait LaplaceTranslationTraits {
    type A: Scalar;

    fn make_laplace_compressed_source_target_evaluator(
        sources: ArrayView2<Self::A>,
        targets: ArrayView2<Self::A>,
    ) -> TwoSidedIDResult<Self::A>;
}
