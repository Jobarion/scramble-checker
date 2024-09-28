use ndarray::Array;
use opencv::{core as core_cv, prelude::*};

pub use element_type::*;
mod element_type {
    use super::*;

    pub trait OpenCvElement {
        const DEPTH: i32;
    }

    impl OpenCvElement for u8 {
        const DEPTH: i32 = core_cv::CV_8U;
    }

    impl OpenCvElement for i8 {
        const DEPTH: i32 = core_cv::CV_8S;
    }

    impl OpenCvElement for u16 {
        const DEPTH: i32 = core_cv::CV_16U;
    }

    impl OpenCvElement for i16 {
        const DEPTH: i32 = core_cv::CV_16S;
    }

    impl OpenCvElement for i32 {
        const DEPTH: i32 = core_cv::CV_32S;
    }

    impl OpenCvElement for f32 {
        const DEPTH: i32 = core_cv::CV_32F;
    }

    impl OpenCvElement for f64 {
        const DEPTH: i32 = core_cv::CV_64F;
    }
}

pub fn try_mat_to_array<T: OpenCvElement + Clone, D: ndarray::Dimension>(mat: &Mat) -> anyhow::Result<Array<T, D>> {
    anyhow::ensure!(mat.depth() == T::DEPTH, "element type mismatch");
    anyhow::ensure!(mat.is_continuous(), "Mat data must be continuous");

    let size = mat.mat_size();
    let size = size.iter().map(|&dim| dim as usize);
    let channels = mat.channels() as usize;
    let size: Vec<usize> = size.chain([channels]).collect();

    let numel: usize = size.iter().product();
    let ptr = mat.ptr(0)? as *const T;

    let slice = unsafe { std::slice::from_raw_parts(ptr, numel) };

    let array = ndarray::ArrayViewD::from_shape(size, slice)?;

    let array = array.into_dimensionality()?;
    Ok(array.into_owned())
}
