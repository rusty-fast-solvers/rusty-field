//! Generation of sources for different geometries.

use ndarray::{s, Array2, ArrayView, ArrayView2, Axis};
use num;
use rusty_base::RealType;

pub trait DiscretePotential {
    type A: RealType;

    fn potential_points<'a>() -> ArrayView2<'a, Self::A>;

}



pub trait CubeSources
where
    Self: RealType,
{
    fn unit_plane_sources(p: usize) -> Array2<Self>;
    fn cube_sources(p: usize, origin: &[Self; 3], length: Self) -> Array2<Self>;
}

macro_rules! impl_cube_sources {
    ($A:ty) => {
        impl CubeSources for $A {
            fn unit_plane_sources(p: usize) -> Array2<Self> {
                let mut points = Array2::<$A>::zeros((2, p * p));

                let mut count = 0;
                for i in 1..(p + 1) {
                    for j in 1..(p + 1) {
                        points[[0, count]] = num::cast::<usize, $A>(i).unwrap()
                            / num::cast::<usize, $A>(1 + p).unwrap();
                        points[[1, count]] = num::cast::<usize, $A>(j).unwrap()
                            / num::cast::<usize, $A>(1 + p).unwrap();
                        count += 1;
                    }
                }

                points
            }

            fn cube_sources(p: usize, origin: &[$A; 3], length: $A) -> Array2<Self> {
                let m = p * p;
                let mut points = Array2::<$A>::zeros((3, 6 * m));
                let plane_points = length * Self::unit_plane_sources(p);

                let start = 0;
                let end = m;

                let origin = ArrayView::from(origin).to_owned();

                // Bottom x-y points
                points.slice_mut(s![0..2, start..end]).assign(&plane_points);
                points.slice_mut(s![2, start..end]).fill(0.0);

                let start = start + m;
                let end = end + m;

                // Top x-y points
                points.slice_mut(s![0..2, start..end]).assign(&plane_points);
                points.slice_mut(s![2, start..end]).fill(length);

                let start = start + m;
                let end = end + m;

                // Front x-z points
                points
                    .slice_mut(s![0, start..end])
                    .assign(&plane_points.index_axis(Axis(0), 0));
                points
                    .slice_mut(s![2, start..end])
                    .assign(&plane_points.index_axis(Axis(0), 1));

                points.slice_mut(s![1, start..end]).fill(0.0);

                let start = start + m;
                let end = end + m;

                // Back x-z points
                points
                    .slice_mut(s![0, start..end])
                    .assign(&plane_points.index_axis(Axis(0), 0));
                points
                    .slice_mut(s![2, start..end])
                    .assign(&plane_points.index_axis(Axis(0), 1));

                points.slice_mut(s![1, start..end]).fill(length);

                let start = start + m;
                let end = end + m;

                // Left y-z points
                points
                    .slice_mut(s![1, start..end])
                    .assign(&plane_points.index_axis(Axis(0), 0));
                points
                    .slice_mut(s![2, start..end])
                    .assign(&plane_points.index_axis(Axis(0), 1));

                points.slice_mut(s![0, start..end]).fill(0.0);

                let start = start + m;
                let end = end + m;

                // Right y-z points
                points
                    .slice_mut(s![1, start..end])
                    .assign(&plane_points.index_axis(Axis(0), 0));
                points
                    .slice_mut(s![2, start..end])
                    .assign(&plane_points.index_axis(Axis(0), 1));

                points.slice_mut(s![0, start..end]).fill(length);

                points + origin.into_shape((3, 1)).unwrap().broadcast((3, 6 * m)).unwrap()
            }
        }
    };
}

impl_cube_sources!(f32);
impl_cube_sources!(f64);
