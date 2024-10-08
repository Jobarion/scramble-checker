use std::fmt::{Debug, Formatter};
use std::ops::{Add, AddAssign};

use crate::puzzle::Cube;
use crate::{Bbox, Point2, YOLOResult, COLORS, Frame};
use anyhow::Result;
use itertools::Itertools;
use kmeans::{KMeans, KMeansConfig};
use log::debug;
use opencv::core::{MatTraitConst, Point, Point2f, Rect, Vector, CV_32F};
use opencv::imgproc::ContourApproximationModes::CHAIN_APPROX_SIMPLE;
use opencv::imgproc::RetrievalModes::RETR_TREE;
use opencv::imgproc::{FILLED, LINE_8};
use opencv::prelude::{Mat, MatTraitConstManual};
use ordered_float::{OrderedFloat, Pow};
use crate::util::rotate90;

const MIN_FACE_AREA: f64 = 2000.0;
const MAX_CNT_HULL_RATIO: f64 = 1.07;
const FACE_POLY_APPROX_FACTOR: f64 = 0.02;

pub type CubePrediction222 = CubePredictionNxN<2>;
pub type CubePrediction333 = CubePredictionNxN<3>;
pub type CubePrediction444 = CubePredictionNxN<4>;
pub type CubePrediction555 = CubePredictionNxN<5>;
pub type CubePrediction666 = CubePredictionNxN<6>;
pub type CubePrediction777 = CubePredictionNxN<7>;

pub type PuzzleFace = Vector<Point>;

type Color = usize;
type ColorPrediction = (Color, OrderedFloat<f32>);

pub trait PuzzleDetector {
    fn find_face(&mut self, frame: &mut Frame, face: &YOLOResult) -> Result<Option<PuzzleFace>>;
    fn ingest_frame(&mut self, frame: &mut Frame, face: PuzzleFace, facelets: &YOLOResult) -> Result<()>;
    fn draw_state(&self, img: &mut Mat) -> Result<()>;
    fn compare_to_puzzle(&self) -> (usize, usize, usize);
    fn reset(&mut self);
}

pub struct CubePredictionNxN<const N: usize> {
    pub faces: [FacePredictionNxN<N>; 6],
    pub cube: Cube<N>
}

impl <const N: usize> PuzzleDetector for CubePredictionNxN<N> {
    fn find_face(&mut self, frame: &mut Frame, face: &YOLOResult) -> Result<Option<PuzzleFace>> {
        get_face_poly(frame, face, 4)
    }

    fn ingest_frame(&mut self, frame: &mut Frame, face: PuzzleFace, facelets: &YOLOResult) -> Result<()> {
        let facelets = facelets.bboxes.as_ref()
            .and_then(|boxes|get_n_facelets(&face, boxes.clone(), N * N, 0f32, 10f32).transpose())
            .transpose()?;
        if let Some(facelets) = facelets {
            for i in 1..face.len() {
                opencv::imgproc::line(&mut frame.output, face.get(i - 1).unwrap(), face.get(i).unwrap(), (0, 0, 0).into(), 3, LINE_8, 0)?;
            }
            opencv::imgproc::line(&mut frame.output, face.get(face.len() - 1).unwrap(), face.get(0).unwrap(), (0, 0, 0).into(), 3, LINE_8, 0)?;

            let moments = opencv::imgproc::moments_def(&face)?;
            let face_center = Point2::new((moments.m10 / moments.m00) as f32, (moments.m01 / moments.m00) as f32);

            let grid = arrange_facelets_nxn_grid::<N>(&face_center, &facelets)?;
            for x in 0..N {
                for y in 0..N {
                    let g = &grid[x][y];
                    let c = g.cxcy();

                    let roi = Rect::new(g.xmin() as i32, g.ymin() as i32, (g.xmax() - g.xmin()) as i32, (g.ymax() - g.ymin()) as i32);
                    println!("Calc roi: {:?}", roi);
                    let size = roi.size();
                    let roi = frame.output.roi(roi)?;
                    let mut roif = Mat::default();

                    let roi = roi.convert_to_def(&mut roif, CV_32F);

                    println!("Extracted roi");
                    println!("{:?}", roif.size());

                    // let criteria =

                    // opencv::core::kmeans_def()

                    // let points: Vec<f32> = roi.clone_pointee().data_bytes()?.iter()
                    //     .map(|x|*x as f32)
                    //     .collect();
                    // println!("Points!");

                    // let size = points.len();
                    //
                    // let kmeans: KMeans<_, 8> = KMeans::new(points, size / 3, 3);
                    // let result = kmeans.kmeans_lloyd(2, 100, KMeans::init_kmeanplusplus, &KMeansConfig::default());
                    // let most_frequent = result.centroid_frequency.iter()
                    //     .enumerate()
                    //     .max_by_key(|(_, x)|x.clone())
                    //     .unwrap()
                    //     .0;
                    // let cr = result.centroids[most_frequent + 2] as i32;
                    // let cg = result.centroids[most_frequent + 1] as i32;
                    // let cb = result.centroids[most_frequent + 0] as i32;



                    opencv::imgproc::circle(&mut frame.output, Point::new(c.x() as i32, c.y() as i32), 9, (0, 0, 0).into(), FILLED, LINE_8, 0)?;
                    opencv::imgproc::circle(&mut frame.output, Point::new(c.x() as i32, c.y() as i32), 7, COLORS[g.id() - 2].into(), FILLED, LINE_8, 0)?;
                    // opencv::imgproc::circle(&mut frame.output, Point::new(c.x() as i32, c.y() as i32), 7, (cr, cg, cb).into(), FILLED, LINE_8, 0)?;
                }
            }
            self.ingest_face_prediction(&grid);
        }
        Ok(())
    }

    fn draw_state(&self, mut frame: &mut Mat) -> Result<()> {
        for i in 0..6 {
            let offset_x = 0;
            let offset_y = 0;
            let tile_size = 10;
            let face_spacing = 5;

            // debug!("{:?}", self.faces[i]);

            for x in 0..self.get_n() {
                for y in 0..self.get_n() {
                    let (cid, conf) = self.faces[i].facelets[y][x].get_best();
                    let color = if conf > OrderedFloat(0.7) {
                        COLORS[cid]
                    } else {
                        (200, 200, 200)
                    };
                    let x_start = offset_x + tile_size * x;
                    let y_start = offset_y + tile_size * (i * self.get_n() + y) + face_spacing * i;

                    let rect = Rect::new(x_start as i32, y_start as i32, tile_size as i32, tile_size as i32);
                    opencv::imgproc::rectangle(&mut frame, rect, color.into(), FILLED, LINE_8, 0)?;
                }
            }
        }
        Ok(())
    }

    fn compare_to_puzzle(&self) -> (usize, usize, usize) {
        let mut sum_right = 0;
        let mut sum_wrong = 0;
        for fidx in 0..6  {
            for x in 0..N {
                for y in 0..N {
                    let (cid, conf) = self.faces[fidx].facelets[x][y].get_best();
                    if conf <= OrderedFloat(0.7f32) {
                        continue
                    } else if cid == self.cube.faces[fidx].stickers[x][y] {
                        sum_right += 1;
                    } else if self.faces[fidx].facelets[x][y].count > 5 {
                        sum_wrong += 1;
                    }
                }
            }
        }
        (sum_right, sum_wrong, N * N * 6 - sum_right - sum_wrong)
    }

    fn reset(&mut self) {
        *self = Self::new(self.cube)
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

fn get_face_poly(frame: &mut Frame, face: &YOLOResult, face_vertices: usize) -> Result<Option<Vector<Point>>> {
    if let Some(masks) = face.masks() {
        let mat = Mat::from_bytes::<u8>(masks[0].as_slice())?;
        let mat = mat.reshape(0, frame.input.size()?.height)?;
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

    pub fn new(target: Cube<N>) -> Self {
        Self {
            faces: [FacePredictionNxN::<N>::default(); 6],
            cube: target,
        }
    }

    fn ingest_face_prediction(&mut self, grid: &[[Bbox; N]; N]) {
        let (best_face_id, best_rotation) = self.find_best_fit(grid);
        let best_face = self.faces.get_mut(best_face_id).unwrap();
        for x in 0..N {
            for y in 0..N {
                let mut fl = &mut best_face.facelets[x][y];
                fl += &best_rotation[x][y];
            }
        }
    }

    fn find_best_fit(&self, grid: &[[Bbox; N]; N]) -> (usize, [[ColorPrediction; N]; N]) {
        let mut predictions: [[ColorPrediction; N]; N] = [[(0, OrderedFloat(0.0f32)); N]; N];
        for x in 0..N {
            for y in 0..N {
                let g = grid[x][y];
                predictions[x][y] = (g.id() - 2, OrderedFloat(g.confidence()));
            }
        }
        self.find_best_fit_min_squared_error(&predictions)
    }

    fn find_best_fit_min_squared_error(&self, prediction: &[[ColorPrediction; N]; N]) -> (usize, [[ColorPrediction; N]; N]) {
        let mut rotations = vec![];
        let mut prediction = prediction.clone();
        rotations.push(prediction.clone());
        for _ in 0..3 {
            rotate90(&mut prediction);
            rotations.push(prediction.clone());
        }
        let ((idx, _), rotation) = self.cube.faces.iter()
            .enumerate()
            .cartesian_product(rotations.into_iter())
            .min_by_key(|((cidx, c_face), p_face)| {
                let sum: OrderedFloat<f32> = (0..N).cartesian_product(0..N)
                    .map(|(x, y)|{
                        let c_color = c_face.stickers[x][y];
                        let (p_color, p_conf) = p_face[x][y];
                        if c_color != p_color {
                            p_conf.pow(2)
                        } else {
                            OrderedFloat(0.0f32) // ???
                        }
                    })
                    .sum();
                // println!("{cidx}: {sum}");
                sum
            })
            .unwrap();
        (idx, rotation)
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

fn _arrange_facelets_nxn_grid<const N: usize>(center: &Point2, outer_corner: &Point2, mut facelets: Vec<Bbox>, n: usize, grid: &mut [[Bbox; N]; N]) -> Result<()> {
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

    let ring_size = n * 4 - 4;
    let (outer_ring, rest) = facelets.split_at_mut(ring_size);
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

#[derive(Copy, Clone)]
pub struct FaceletPrediction<const COLORS: usize> {
    probabilities: [OrderedFloat<f32>; COLORS],
    count: usize,
}

impl <const COLORS: usize> Default for FaceletPrediction<COLORS> {
    fn default() -> Self {
        Self {
            probabilities: [OrderedFloat(1f32) / COLORS as f32; COLORS],
            count: 0,
        }
    }
}

impl <const COLORS: usize> FaceletPrediction<COLORS> {
    pub fn add_color(&mut self, color: Color, conf: OrderedFloat<f32>) {
        let mut s = self;
        s += &(color, conf)
    }

    pub fn get_best(&self) -> ColorPrediction {
        self.probabilities.iter().cloned()
            .enumerate()
            .max_by_key(|(_, x)|x.clone())
            .unwrap()
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

impl <'a, 'b, const COLORS: usize> AddAssign<&'b ColorPrediction> for &'a mut FaceletPrediction<COLORS> {

    fn add_assign(&mut self, rhs: &'b ColorPrediction) {
        for x in 0..COLORS {
            let new_probability = if x == rhs.0 {
                self.probabilities[x] * self.count as f32 + rhs.1
            } else {
                self.probabilities[x] * self.count as f32
            };
            self.probabilities[x] = new_probability / (self.count + 1) as f32;
        }
        self.count += 1;
    }
}

impl <'b, const COLORS: usize> Add<&'b ColorPrediction> for FaceletPrediction<COLORS> {
    type Output = FaceletPrediction<COLORS>;

    fn add(mut self, rhs: &'b ColorPrediction) -> Self::Output {
        let mut s = &mut self;
        s += rhs;
        self
    }
}