use std::any::Any;
use std::cmp::max;
use std::error::Error;
use std::fmt::format;
use std::str::FromStr;
use std::thread::sleep;
use std::time::{Duration, Instant};
use clap::builder::Str;
use clap::Parser;
use imageproc::point::Point;
use log::{debug, info, LevelFilter, trace};
use opencv::{highgui, prelude::*, videoio};
use opencv::core::{Rect, Size};
use opencv::imgproc::{FILLED, FONT_HERSHEY_SCRIPT_SIMPLEX, FONT_HERSHEY_SIMPLEX, LINE_8};
use opencv::videoio::{VideoCapture, VideoWriter};
use simple_logger::SimpleLogger;
use scramble_checker::{OrtEP, YOLOTask, YOLOv8};
use scramble_checker::detector::{CubePredictionNxN, PuzzleDetector};
use scramble_checker::puzzle::{BACK, Cube, CubeAlgorithm, DOWN, FRONT, LEFT, RIGHT, UP};
use scramble_checker::session::{DetectionSession, Output};

#[derive(Parser, Clone)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    #[arg(long, default_value = "333", value_parser = ["222", "333", "444", "555"])]
    pub puzzle: String,
    #[arg(long, required = true)]
    pub scramble: String,
}

const MODEL_COLOR_SCHEME: [usize; 6] = [4, 5, 1, 0, 2, 3];

fn main() -> anyhow::Result<()> {
    SimpleLogger::new()
        .with_module_level("timings", LevelFilter::Debug)
        .with_level(LevelFilter::Debug)
        .init()?;

    let args = Args::parse();

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


    // let mut video_in = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    let mut video_in = videoio::VideoCapture::from_file_def("video.mp4")?;
    let opened = videoio::VideoCapture::is_opened(&video_in)?;

    if !opened {
        panic!("Unable to open default camera!");
    }

    let facelet_model = YOLOv8::new(OrtEP::Cuda(0), "facelet.onnx".to_string(), YOLOTask::Detect, 0.3, 0.45)?;
    facelet_model.summary();
    let face_model = YOLOv8::new(OrtEP::Cuda(0), "face.onnx".to_string(), YOLOTask::Segment, 0.3, 0.45)?;
    face_model.summary();
    let mut frame = Mat::default();
    video_in.read(&mut frame)?;
    face_model.run(&frame)?;
    facelet_model.run(&frame)?;


    let fourcc_code = VideoWriter::fourcc('X', 'V', 'I', 'D')?;
    let mut file = VideoWriter::new("output.avi", fourcc_code, 20.0, frame.size()?, true)?;

    let mut session = DetectionSession {
        video_input: video_in,
        face_model: &face_model,
        facelet_model: &facelet_model,
        detector: detector,
        outputs: vec![Output::Window("output".to_string()), Output::File(file)],
    };

    let mut cooldown_until = Instant::now();
    let mut last_result = 0.5;
    loop {
        // highgui::wait_key_def()?;
        highgui::poll_key()?; // Needs to be called to refresh UI, we don't care about keys
        if cooldown_until > Instant::now() {
            let mut frame = session.read_frame()?;
            session.draw_frame(||{
                let (text, color) = if last_result > 0.5 {
                    ("Correct", (0, 255, 0))
                } else {
                    ("Incorrect", (0, 0, 255))
                };
                opencv::imgproc::put_text(&mut frame, text, opencv::core::Point::new(40, 40), FONT_HERSHEY_SIMPLEX, 1.0, color.into(), 3, LINE_8, false)?;
                Ok(frame)
            })?;
        } else if session.process_frame()? {
            let conf = session.get_confidence();
            if conf > 0.9 {
                info!("Correct!");
                session.reset();
                last_result = conf;
                cooldown_until = Instant::now() + Duration::from_secs(3);
            } else if conf < 0.1 {
                info!("Incorrect!");
                session.reset();
                last_result = conf;
                cooldown_until = Instant::now() + Duration::from_secs(3);
            }
        }
    }
}
