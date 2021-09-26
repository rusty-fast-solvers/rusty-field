//! This module defines C API function to access all assembly and evaluation routines.

use ndarray;

#[no_mangle]
pub extern "C" fn cube_sources_f64(
    p: usize,
    length: f64,
    origin_ptr: *const f64,
    sources_ptr: *mut f64,
) {
    use crate::sources::CubeSources;
    let origin: &[f64; 3] = unsafe { &*(origin_ptr as *const [f64; 3]) };
    let mut sources =
        unsafe { ndarray::ArrayViewMut2::from_shape_ptr((3, 6 * p * p), sources_ptr) };

    let tmp = f64::cube_sources(p, origin, length);
    sources.assign(&tmp);
}
