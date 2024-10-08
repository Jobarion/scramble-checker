use clap::Parser;
use log::LevelFilter;
use opencv::prelude::*;
use serde::Deserialize;
use simple_logger::SimpleLogger;

use scramble_checker::{init_face_model, init_facelet_model, run_detection, Args, Mode, YOLOv8};

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let facelet_model = init_facelet_model()?;
    let face_model = init_face_model()?;

    if args.mode == Mode::Test {
        SimpleLogger::new()
            .with_level(LevelFilter::Off)
            .init()?;
        run_tests(&face_model, &facelet_model)?;
    } else {
        SimpleLogger::new()
            .with_module_level("timings", LevelFilter::Info)
            .with_level(LevelFilter::Debug)
            .init()?;
        run_detection(&face_model, &facelet_model, &args)?;
    }
    Ok(())
}

#[derive(Debug, Clone, Deserialize)]
struct LengthTestCase {
    puzzle: String,
    scramble: String,
    file: String,
    correct: bool
}

fn run_tests(face_model: &YOLOv8, facelet_model: &YOLOv8) -> anyhow::Result<()>{
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path("tests/tests.csv")
        .unwrap();
    for result in reader.deserialize() {
        let record: LengthTestCase = result.expect("A CSV record");
        println!("Testing {}", record.file);
        if run_test(&record, &face_model, &facelet_model)? {
            println!("Okay");
        } else {
            println!("Not okay");
        }
    }
    Ok(())
}

fn run_test(test: &LengthTestCase, face_model: &YOLOv8, facelet_model: &YOLOv8) -> anyhow::Result<bool> {
    let file = std::env::current_dir()?.join("tests").join(test.file.clone()).to_str().unwrap().to_string();
    let args = Args {
        puzzle: test.puzzle.clone(),
        scramble: test.scramble.clone(),
        file: Some(file),
        camera: 0,
        mode: Mode::Test,
    };
    let result = run_detection(face_model, facelet_model, &args)?;
    return Ok(result.is_some() && test.correct == result.unwrap())
}