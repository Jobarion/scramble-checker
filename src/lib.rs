#![feature(get_many_mut)]
#![feature(let_chains)]
#![allow(clippy::type_complexity)]

use std::io::{Read, Write};
use std::str::FromStr;
use std::time::{Duration, Instant};
use clap::{Parser, ArgGroup};
use log::{info, LevelFilter};
use opencv::core::Mat;
use opencv::{highgui, videoio};
use opencv::imgproc::{FONT_HERSHEY_SIMPLEX, LINE_8};
use opencv::prelude::{MatTraitConst, VideoCaptureTrait, VideoCaptureTraitConst};
use opencv::videoio::{VideoCapture, VideoWriter};
use serde::{Deserialize, Serialize};
use simple_logger::SimpleLogger;
use crate::detector::{CubePredictionNxN, PuzzleDetector};

pub mod model;
pub mod ort_backend;
pub mod yolo_result;
pub mod convert;
pub mod detector;
pub mod puzzle;
pub mod session;
mod util;

pub use crate::model::YOLOv8;
pub use crate::ort_backend::{Batch, OrtBackend, OrtConfig, OrtEP, YOLOTask};
use crate::puzzle::{Cube, CubeAlgorithm};
use crate::session::{DetectionSession, Output};
pub use crate::yolo_result::{Bbox, Embedding, Point2, YOLOResult};

// B G R
// Blue, Green, Orange, Red, White, Yellow
pub const COLORS: [(i32, i32, i32); 6] = [(255, 0, 0), (0, 255, 0), (0, 180, 255), (0, 0, 255), (255, 255, 255), (0, 255, 255)];
pub const COLORS_OFFSET: usize = 2;

#[derive(Parser, Clone)]
#[command(author, version, about, long_about = None)]
#[clap(group(
    ArgGroup::new("input")
    .multiple(false)
    .args(&["file", "camera"])
))]
pub struct Args {
    #[arg(long, default_value = "333", value_parser = ["222", "333", "444", "555"])]
    pub puzzle: String,
    #[arg(long, required = true)]
    pub scramble: String,
    #[arg(long)]
    pub file: Option<String>,
    #[arg(long, default_value = "0")]
    pub camera: i32,
    #[arg(long, default_value = "verify", value_enum)]
    pub mode: Mode,
}

#[derive(Copy, Clone, PartialEq, Eq, clap::ValueEnum)]
#[clap(rename_all = "kebab_case")]
pub enum Mode {
    Detect,
    Verify,
    Test,
}

pub struct Frame {
    input: Mat,
    output: Mat,
}

impl Frame {
    pub fn new(input: Mat) -> Frame {
        Frame {
            output: input.clone(),
            input,
        }
    }
}

const MODEL_COLOR_SCHEME: [usize; 6] = [4, 5, 1, 0, 2, 3];

pub fn run_detection(face_model: &YOLOv8, facelet_model: &YOLOv8, args: &Args) -> anyhow::Result<Option<bool>> {
    let alg = CubeAlgorithm::from_str(args.scramble.as_str())?;

    let detector: Box<dyn PuzzleDetector> = match args.puzzle.as_str() {
        "222" => {
            let mut cube = Cube::<2>::new_with_scheme(MODEL_COLOR_SCHEME);
            cube.turn_alg(&alg);
            Box::new(CubePredictionNxN::<2>::new(cube))
        },
        "333" => {
            let mut cube = Cube::<3>::new_with_scheme(MODEL_COLOR_SCHEME);
            cube.turn_alg(&alg);
            Box::new(CubePredictionNxN::<3>::new(cube))
        },
        "444" => {
            let mut cube = Cube::<4>::new_with_scheme(MODEL_COLOR_SCHEME);
            cube.turn_alg(&alg);
            Box::new(CubePredictionNxN::<4>::new(cube))
        },
        "555" => {
            let mut cube = Cube::<5>::new_with_scheme(MODEL_COLOR_SCHEME);
            cube.turn_alg(&alg);
            Box::new(CubePredictionNxN::<5>::new(cube))
        },
        _ => unreachable!()
    };

    let mut video_in = args.file.as_ref().map(|file|videoio::VideoCapture::from_file_def(file.as_str()))
        .unwrap_or_else(||videoio::VideoCapture::new(args.camera, videoio::CAP_ANY))?;

    let opened = videoio::VideoCapture::is_opened(&video_in)?;

    if !opened {
        panic!("Unable to open default camera!");
    }

    let mut frame = Mat::default();
    video_in.read(&mut frame)?;
    face_model.run(&frame)?;
    facelet_model.run(&frame)?;

    let outputs = if args.mode == Mode::Test {
        vec![Output::Window("output".to_string())]
    } else {
        let fourcc_code = VideoWriter::fourcc('X', 'V', 'I', 'D')?;
        let file = VideoWriter::new("output.avi", fourcc_code, 20.0, frame.size()?, true)?;
        vec![Output::Window("output".to_string()), Output::File(file)]
    };

    let mut session = DetectionSession {
        video_input: video_in,
        face_model: &face_model,
        facelet_model: &facelet_model,
        detector,
        outputs,
    };

    let mut cooldown_until = Instant::now();
    let mut last_result = 0.5;
    loop {
        if args.mode == Mode::Verify {
            // highgui::wait_key(0)?;
            highgui::poll_key()?;
        } else {
            highgui::poll_key()?; // Needs to be called to refresh UI, we don't actually care about keys
        }
        if cooldown_until > Instant::now() {
            let frame = session.read_frame()?;
            if frame.is_none() {
                return Ok(None)
            }
            let mut frame = frame.unwrap();
            session.draw_frame(||{
                let (text, color) = if last_result > 0.5 {
                    ("Correct", (0, 255, 0))
                } else {
                    ("Incorrect", (0, 0, 255))
                };
                opencv::imgproc::put_text(&mut frame, text, opencv::core::Point::new(40, 40), FONT_HERSHEY_SIMPLEX, 1.0, color.into(), 3, LINE_8, false)?;
                Ok(frame)
            })?;
        } else if let (found, mut frame) = session.process_frame()? {
            if args.mode != Mode::Detect {
                session.detector.draw_state(&mut frame.output)?;
                session.draw_frame(||Ok(frame.output))?;
            } else {
                session.detector.draw_state(&mut frame.input)?;
                session.draw_frame(||Ok(frame.input))?;
            }
            if !found {
                continue
            }
            let conf = session.get_confidence();
            if conf > 0.9 {
                info!("Correct!");
                if args.mode == Mode::Test {
                    return Ok(Some(true))
                }
                session.reset();
                last_result = conf;
                cooldown_until = Instant::now() + Duration::from_secs(3);
            } else if conf < 0.1 {
                info!("Incorrect!");
                if args.mode == Mode::Test {
                    return Ok(Some(false))
                }
                session.reset();
                last_result = conf;
                cooldown_until = Instant::now() + Duration::from_secs(3);
            }
        }
    }
}

pub fn init_face_model() -> anyhow::Result<YOLOv8> {
    let model = YOLOv8::new(OrtEP::Cuda(0), "face.onnx".to_string(), YOLOTask::Segment, 0.2, 0.45)?;
    model.summary();
    Ok(model)
}

pub fn init_facelet_model() -> anyhow::Result<YOLOv8> {
     let model = YOLOv8::new(OrtEP::Cuda(0), "facelet.onnx".to_string(), YOLOTask::Detect, 0.2, 0.45)?;
     model.summary();
     Ok(model)
}
