use std::fmt::{Debug, Formatter};
use std::ops::{Add, AddAssign};
use opencv::core::{Point, Point2f, Point2i};
use opencv::prelude::Mat;

use crate::{Bbox, Point2};

pub trait PuzzleDetector {
    fn ingest_frame(&mut self, mat: &Mat, boxes: &Vec<Bbox>);
}

pub struct CubeNxNDetector<const N: usize> {
    cube: CubePredictionNxN<N>
}

impl <const N: usize> PuzzleDetector for CubeNxNDetector<N> {
    fn ingest_frame(&mut self, mat: &Mat, boxes: &Vec<Bbox>) {

    }
}

pub struct CubePredictionNxN<const N: usize> {
    faces: [FacePredictionNxN<N>; 6]
}

impl <const N: usize> Default for CubePredictionNxN<N> {
    fn default() -> Self {
        Self {
            faces: [FacePredictionNxN::<N>::default(); 6],
        }
    }
}

impl <const N: usize> CubePredictionNxN<N> {
    pub fn ingest_face_prediction(&mut self, prediction: &FacePredictionNxN<N>) {
        let (best_face_id, best_rotation) = self.find_best_fit(prediction);
        let mut best_face = self.faces.get_mut(best_face_id).unwrap();
        best_face += &best_rotation;
    }

    fn find_best_fit(&self, prediction: &FacePredictionNxN<N>) -> (usize, FacePredictionNxN<N>) {
        let mut rotations = vec![];
        let mut prediction = prediction.clone();
        rotations.push(prediction.clone());
        for _ in 0..3 {
            prediction.rotate90();
            rotations.push(prediction.clone());
        }
        let mut best_rot_idx = 0;
        let mut best_face_idx = 0;
        let mut min_err = f32::INFINITY;
        for ri in 0..rotations.len() {
            for fi in 0..self.faces.len() {
                let err = self.faces[fi].diff_squared(&rotations[ri]);
                if err < min_err {
                    min_err = err;
                    best_face_idx = fi;
                    best_rot_idx = ri;
                }
            }
        }
        return (best_face_idx, rotations[best_rot_idx])
    }

    fn get_best_facelets(boxes: Vec<Bbox>) {

    }

    // fn arrange_facelet_grid(center: Point2f, )
}

#[derive(Copy, Clone)]
pub struct FacePredictionNxN<const N: usize> {
    facelets: [[FaceletPrediction<6>; N]; N]
}

impl <const N: usize> Default for FacePredictionNxN<N> {
    fn default() -> Self {
        Self {
            facelets: [[FaceletPrediction::default(); N]; N]
        }
    }
}

impl <const N: usize> FacePredictionNxN<N> {
    pub fn rotate90(&mut self) {
        for n in 0..N/2 {
            self.facelets.swap(n, N - n - 1)
        }
        self.transpose();
    }

    fn transpose(&mut self) {
        for n in 0..N {
            for m in 0..n {
                let (x, y) = self.facelets.split_at_mut(n);
                std::mem::swap(&mut y[0][m], &mut x[m][n]);
            }
        }
    }

    fn diff_squared(&self, other: &Self) -> f32 {
        self.facelets.iter()
            .flat_map(|x|x.iter())
            .zip(other.facelets.iter().flat_map(|x|x.iter()))
            .map(|(x, y)|x.diff_squared(y))
            .sum()
    }
}

impl <const N: usize> Debug for FacePredictionNxN<N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for n in 0..N {
            for m in 0..N {
                write!(f, "{:?} ", self.facelets[n][m])?;
            }
            writeln!(f, "")?;
        }
        Ok(())
    }
}

impl <'a, 'b, const N: usize> AddAssign<&'b FacePredictionNxN<N>> for &'a mut FacePredictionNxN<N> {

    fn add_assign(&mut self, rhs: &'b FacePredictionNxN<N>) {
        for n in 0..N {
            let mut nslice = &mut self.facelets[n];
            for m in 0..N {
                let mut nm = &mut nslice[m];
                nm += &rhs.facelets[n][m];
            }
        }
    }
}

impl <'b, const N: usize> Add<&'b FacePredictionNxN<N>> for FacePredictionNxN<N> {
    type Output = FacePredictionNxN<N>;

    fn add(mut self, rhs: &'b FacePredictionNxN<N>) -> Self::Output {
        let mut s = &mut self;
        s += rhs;
        self
    }
}

#[derive(Copy, Clone)]
pub struct FaceletPrediction<const COLORS: usize> {
    colors: [f32; COLORS],
    count: usize,
}

impl <const COLORS: usize> Default for FaceletPrediction<COLORS> {
    fn default() -> Self {
        Self {
            colors: [1f32 / COLORS as f32; COLORS],
            count: 0,
        }
    }
}

impl <const COLORS: usize> FaceletPrediction<COLORS> {
    pub fn from_color(color: usize, conf: f32) -> FaceletPrediction<COLORS> {
        let rest_conf = (1f32 - conf) / (COLORS - 1) as f32;
        let mut colors = [rest_conf; COLORS];
        colors[color] = conf;
        FaceletPrediction {
            colors,
            count: 1
        }
    }

    pub fn add_color(&mut self, color: usize, conf: f32) {
        let mut s = self;
        s += &Self::from_color(color, conf)
    }

    pub fn get_best(&self) -> (usize, f32) {
        self.colors.iter().cloned()
            .enumerate()
            .max_by(|(_, x), (_, y)|x.total_cmp(y))
            .unwrap()
    }

    pub fn diff_squared(&self, other: &Self) -> f32 {
        self.colors.iter().zip(other.colors.iter())
            .map(|(x, y)|(x - y).powi(2))
            .sum()
    }
}

impl <const COLORS: usize> Debug for FaceletPrediction<COLORS> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let (color, confidence) = self.get_best();
        if (confidence - (1f32 / COLORS as f32)).abs() <= f32::EPSILON {
            write!(f, "(?, ????)")
        } else {
            write!(f, "({color}, {confidence:.2})")
        }
    }
}

impl <'a, 'b, const COLORS: usize> AddAssign<&'b FaceletPrediction<COLORS>> for &'a mut FaceletPrediction<COLORS> {

    fn add_assign(&mut self, rhs: &'b FaceletPrediction<COLORS>) {
        for x in 0..COLORS {
            let merged = self.colors[x] * self.count as f32 + rhs.colors[x] * rhs.count as f32;
            self.colors[x] = merged / (self.count + rhs.count) as f32;
        }
        self.count += rhs.count;
    }
}

impl <'b, const COLORS: usize> Add<&'b FaceletPrediction<COLORS>> for FaceletPrediction<COLORS> {
    type Output = FaceletPrediction<COLORS>;

    fn add(mut self, rhs: &'b FaceletPrediction<COLORS>) -> Self::Output {
        let mut s = &mut self;
        s += rhs;
        self
    }
}

mod test {
    use crate::detector::FacePredictionNxN;

    #[test]
    pub fn test_transpose() {
        let mut face = FacePredictionNxN::<3>::default();
        face.facelets[0][0].add_color(1, 0.8f32);
        face.facelets[0][1].add_color(1, 0.4f32);
        face.facelets[0][2].add_color(2, 0.8f32);
        face.facelets[1][0].add_color(2, 0.3f32);
        face.facelets[1][1].add_color(2, 0.5f32);
        face.facelets[1][2].add_color(2, 0.6f32);
        face.facelets[2][0].add_color(3, 0.8f32);
        face.facelets[2][1].add_color(3, 0.4f32);
        face.facelets[2][2].add_color(4, 0.8f32);
        println!("{face:?}");
        face.transpose();
        println!("{face:?}");
        face.rotate90();
        println!("{face:?}");
    }
}