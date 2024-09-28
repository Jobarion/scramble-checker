use std::fmt::{Debug, Formatter};
use std::ops::{Add, AddAssign, Sub};

use anyhow::Result;
use opencv::core::{MatTraitConst, Point, Point2f, Vector};
use opencv::imgproc::ContourApproximationModes::CHAIN_APPROX_SIMPLE;
use opencv::imgproc::RetrievalModes::RETR_TREE;
use opencv::prelude::Mat;

use crate::{Bbox, Point2, YOLOResult};

const MIN_FACE_AREA: f64 = 2000.0;
const MAX_CNT_HULL_RATIO: f64 = 1.07;
const FACE_POLY_APPROX_FACTOR: f64 = 0.02;

pub type PuzzleFace = Vector<Point>;

pub trait PuzzleDetector {
    fn find_face(&mut self, mat: &Mat, face: &YOLOResult) -> Result<Option<PuzzleFace>>;
    fn ingest_frame(&mut self, mat: &Mat, face: PuzzleFace, facelets: &YOLOResult) -> Result<()>;
}

pub struct CubePredictionNxN<const N: usize> {
    pub faces: [FacePredictionNxN<N>; 6]
}

impl <const N: usize> Default for CubePredictionNxN<N> {
    fn default() -> Self {
        Self {
            faces: [FacePredictionNxN::<N>::default(); 6],
        }
    }
}

impl <const N: usize> PuzzleDetector for CubePredictionNxN<N> {
    fn find_face(&mut self, mat: &Mat, face: &YOLOResult) -> Result<Option<PuzzleFace>> {
        get_face_poly(mat, face, 4)
    }

    fn ingest_frame(&mut self, img: &Mat, face: PuzzleFace, facelets: &YOLOResult) -> Result<()> {
        let facelets = facelets.bboxes.as_ref()
            .and_then(|boxes|get_n_facelets(&face, boxes.clone(), N * N, 0f32, 10f32).transpose())
            .transpose()?;
        if let Some(facelets) = facelets {
            let moments = opencv::imgproc::moments_def(&face)?;
            let face_center = Point2::new((moments.m10 / moments.m00) as f32, (moments.m01 / moments.m00) as f32);
            let grid = arrange_facelets_nxn_grid::<N>(&face_center, &facelets)?;
            let mut prediction = FacePredictionNxN::<N>::default();
            for x in 0..N {
                for y in 0..N {
                    let g = &grid[x][y];
                    prediction.facelets[x][y].add_color(g.id() - 2, g.confidence());
                }
            }
            self.ingest_face_prediction(&prediction);
        }
        Ok(())
    }
}

fn get_n_facelets(face: &PuzzleFace, mut facelets: Vec<Bbox>, n: usize, max_diff: f32, min_dist_ignore: f32) -> Result<Option<Vec<Bbox>>>{
    let mut deduplicated_facelets: Vec<Bbox> = vec![];
    let min_dist_ignore = min_dist_ignore.powi(2);
    facelets.sort_by(|x, y|y.confidence().total_cmp(&x.confidence()));
    for b in facelets {
        let center = b.cxcy();
        let is_in_poly = opencv::imgproc::point_polygon_test(face, Point2f::new(center.x(), center.y()), false)? > 0f64;
        if !is_in_poly {
            continue
        }
        let is_distinct = deduplicated_facelets.iter()
            .all(|d|center.dist_squared(&d.cxcy()) >= min_dist_ignore);
        if is_distinct {
            deduplicated_facelets.push(b)
        }
        if deduplicated_facelets.len() == n {
            return Ok(Some(deduplicated_facelets))
        }
    }
    Ok(None)
}

fn get_face_poly(img: &Mat, face: &YOLOResult, face_vertices: usize) -> Result<Option<Vector<Point>>> {
    if let Some(masks) = face.masks() {
        let mat = Mat::from_bytes::<u8>(masks[0].as_slice())?;
        let mat = mat.reshape(0, img.size()?.height)?;
        let mut contours: Vector<Mat> = Vector::new();
        opencv::imgproc::find_contours(&mat, &mut contours, RETR_TREE.into(), CHAIN_APPROX_SIMPLE.into(), Point::default())?;

        let mut hull = None;
        for contour in contours {
            let cnt_area = opencv::imgproc::contour_area(&contour, false)?;
            if cnt_area < MIN_FACE_AREA {
                continue
            } else {
                let mut hull_candidate: Vector<Point> = Vector::new();
                opencv::imgproc::convex_hull(&contour, &mut hull_candidate, false, true)?;
                let hull_area = opencv::imgproc::contour_area(&hull_candidate, false)?;
                if hull_area / cnt_area > MAX_CNT_HULL_RATIO {
                    continue
                }
                if hull.is_some() {
                    return Ok(None)
                } else {
                    hull = Some(hull_candidate)
                }
            }
        }

        if let Some(hull) = hull {
            let peri = opencv::imgproc::arc_length(&hull, true)?;
            let mut approx_hull: Vector<Point> = Vector::new();
            opencv::imgproc::approx_poly_dp(&hull, &mut approx_hull, FACE_POLY_APPROX_FACTOR * peri, true)?;
            if approx_hull.len() != face_vertices {
                return Ok(None)
            }
            return Ok(Some(approx_hull))
        }
        Ok(None)
    } else {
        Ok(None)
    }
}

impl <const N: usize> CubePredictionNxN<N> {
    pub fn get_n(&self) -> usize {
        N
    }

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
}

fn arrange_facelets_nxn_grid<const N: usize>(center: &Point2, facelets: &Vec<Bbox>) -> Result<[[Bbox; N]; N]> {
    let outer_corner = facelets.iter()
        .map(|x|x.cxcy())
        .max_by(|x, y|x.dist_squared(center).total_cmp(&y.dist_squared(center)))
        .unwrap();
    let mut grid = [[Bbox::default(); N]; N];

    _arrange_facelets_nxn_grid(center, &outer_corner, facelets.clone(), N, &mut grid)?;
    Ok(grid)
}

fn _arrange_facelets_nxn_grid<const N: usize>(center: &Point2, outer_corner: &Point2, mut facelets: Vec<Bbox>, n: usize, mut grid: &mut [[Bbox; N]; N]) -> Result<()> {
    if n == 1 {
        grid[N / 2][N / 2] = facelets[0].clone();
        return Ok(());
    }

    let mut hull = Vector::<Point>::new();
    let mut facelets_cv2 = Vector::<Point>::new();
    for facelet in facelets.iter() {
        let cxcy = facelet.cxcy();
        facelets_cv2.push(Point::new(cxcy.x() as i32, cxcy.y() as i32));
    }
    opencv::imgproc::convex_hull(&facelets_cv2, &mut hull, false, true)?;

    facelets.sort_by_cached_key(|x|{
        let center = x.cxcy();
        (opencv::imgproc::point_polygon_test(&hull, Point2f::new(center.x(), center.y()), true).unwrap() * 100f64) as i32
    });

    let ring_size = N * 4 - 4;
    let (mut outer_ring, rest) = facelets.split_at_mut(ring_size);
    outer_ring.sort_by_cached_key(|x|{
        let c = x.cxcy();
        ((c.y() - center.y()).atan2(c.x() - center.x()) * 100.0) as i32
    });

    let closest_to_corner_id = outer_ring.iter().enumerate()
        .min_by_key(|(_, x)|(x.cxcy().dist_squared(outer_corner) * 100.0) as i32)
        .unwrap().0;

    outer_ring.rotate_left(closest_to_corner_id);

    let n1 = n - 1;
    let offset = (N - n) / 2;
    for x in 0..n1 {
        grid[offset][x + offset] = outer_ring[x].clone();
        grid[x + offset][n1 + offset] = outer_ring[n1 * 1 + x].clone();
        grid[n1 + offset][n1 - x + offset] = outer_ring[n1 * 2 + x].clone();
        grid[n1 - x + offset][offset] = outer_ring[n1 * 3 + x].clone();
    }
    if n > 2 {
        _arrange_facelets_nxn_grid(center, outer_corner, rest.to_vec(), n - 2, grid)
    } else {
        Ok(())
    }
}

fn cv2_dist_squared(a: &Point, b: &Point) -> f32 {
    ((a.x - b.x).pow(2) + (a.y - b.y).pow(2)) as f32
}

#[derive(Copy, Clone)]
pub struct FacePredictionNxN<const N: usize> {
    pub facelets: [[FaceletPrediction<6>; N]; N]
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