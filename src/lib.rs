#![feature(get_many_mut)]
#![feature(let_chains)]
#![allow(clippy::type_complexity)]

use std::io::{Read, Write};
use std::time::{Duration, Instant};

pub mod model;
pub mod ort_backend;
pub mod yolo_result;
pub mod convert;
pub mod detector;
pub mod puzzle;
pub mod session;

pub use crate::model::YOLOv8;
pub use crate::ort_backend::{Batch, OrtBackend, OrtConfig, OrtEP, YOLOTask};
pub use crate::yolo_result::{Bbox, Embedding, Point2, YOLOResult};

// B G R
// Blue, Green, Orange, Red, White, Yellow
pub const COLORS: [(i32, i32, i32); 6] = [(255, 0, 0), (0, 255, 0), (0, 180, 255), (0, 0, 255), (255, 255, 255), (0, 255, 255)];
pub const COLORS_OFFSET: usize = 2;

pub fn non_max_suppression(
    xs: &mut Vec<(Bbox, Option<Vec<f32>>)>,
    iou_threshold: f32,
) {
    xs.sort_by(|b1, b2| b2.0.confidence().partial_cmp(&b1.0.confidence()).unwrap());

    let mut current_index = 0;
    for index in 0..xs.len() {
        let mut drop = false;
        for prev_index in 0..current_index {
            let iou = xs[prev_index].0.iou(&xs[index].0);
            if iou > iou_threshold {
                drop = true;
                break;
            }
        }
        if !drop {
            xs.swap(current_index, index);
            current_index += 1;
        }
    }
    xs.truncate(current_index);
}

pub fn rotate90<T, const N: usize>(slice: &mut[[T; N]; N]) {
    for n in 0..N/2 {
        slice.swap(n, N - n - 1)
    }
    transpose(slice);
}

pub fn transpose<T, const N: usize>(slice: &mut[[T; N]; N]) {
    for n in 0..N {
        for m in 0..n {
            let (x, y) = slice.split_at_mut(n);
            std::mem::swap(&mut y[0][m], &mut x[m][n]);
        }
    }
}

pub fn gen_time_string(delimiter: &str) -> String {
    let offset = chrono::FixedOffset::east_opt(8 * 60 * 60).unwrap(); // Beijing
    let t_now = chrono::Utc::now().with_timezone(&offset);
    let fmt = format!(
        "%Y{}%m{}%d{}%H{}%M{}%S{}%f",
        delimiter, delimiter, delimiter, delimiter, delimiter, delimiter
    );
    t_now.format(&fmt).to_string()
}

pub const SKELETON: [(usize, usize); 16] = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (11, 13),
    (12, 14),
    (13, 15),
    (14, 16),
];

pub fn check_font(font: &str) -> rusttype::Font<'static> {
    // check then load font

    // ultralytics font path
    let font_path_config = match dirs::config_dir() {
        Some(mut d) => {
            d.push("Ultralytics");
            d.push(font);
            d
        }
        None => panic!("Unsupported operating system. Now support Linux, MacOS, Windows."),
    };

    // current font path
    let font_path_current = std::path::PathBuf::from(font);

    // check font
    let font_path = if font_path_config.exists() {
        font_path_config
    } else if font_path_current.exists() {
        font_path_current
    } else {
        println!("Downloading font...");
        let source_url = "https://ultralytics.com/assets/Arial.ttf";
        let resp = ureq::get(source_url)
            .timeout(std::time::Duration::from_secs(500))
            .call()
            .unwrap_or_else(|err| panic!("> Failed to download font: {source_url}: {err:?}"));

        // read to buffer
        let mut buffer = vec![];
        let total_size = resp
            .header("Content-Length")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap();
        let _reader = resp
            .into_reader()
            .take(total_size)
            .read_to_end(&mut buffer)
            .unwrap();

        // save
        let _path = std::fs::File::create(font).unwrap();
        let mut writer = std::io::BufWriter::new(_path);
        writer.write_all(&buffer).unwrap();
        println!("Font saved at: {:?}", font_path_current.display());
        font_path_current
    };

    // load font
    let buffer = std::fs::read(font_path).unwrap();
    rusttype::Font::try_from_vec(buffer).unwrap()
}

pub struct Stopwatch(Instant);

impl Default for Stopwatch {
    fn default() -> Self {
        Self(Instant::now())
    }
}

impl Stopwatch {
    pub fn elapsed_and_reset(&mut self) -> Duration {
        let elapsed = self.0.elapsed();
        self.reset();
        elapsed
    }

    pub fn reset(&mut self) {
        self.0 = Instant::now()
    }
}