use std::error::Error;
use std::time::Instant;
use log::{debug, LevelFilter, trace};
use opencv::{highgui, prelude::*, videoio};
use opencv::core::Rect;
use opencv::imgproc::{FILLED, LINE_8};
use simple_logger::SimpleLogger;
use scramble_checker::{OrtEP, Stopwatch, YOLOTask, YOLOv8};
use scramble_checker::detector::{CubePredictionNxN, PuzzleDetector};

fn main() -> Result<(), Box<dyn Error>> {
    SimpleLogger::new()
        .with_module_level("timings", LevelFilter::Info)
        .with_level(LevelFilter::Debug)
        .init()?;

    let window_out = "video output";
    let debug_out = "dbg";
    highgui::named_window(window_out, highgui::WINDOW_AUTOSIZE)?;
    highgui::named_window(debug_out, highgui::WINDOW_AUTOSIZE)?;
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?; // 0 is the default camera

    let opened = videoio::VideoCapture::is_opened(&cam)?;
    if !opened {
        panic!("Unable to open default camera!");
    }

    let mut facelet_model = YOLOv8::new(OrtEP::Cuda(0), "facelet.onnx".to_string(), YOLOTask::Detect, 0.3, 0.45)?;
    facelet_model.summary();
    let mut face_model = YOLOv8::new(OrtEP::Cuda(0), "face.onnx".to_string(), YOLOTask::Segment, 0.3, 0.45)?;
    face_model.summary();

    let mut cube = CubePredictionNxN::<3>::default();

    let mut s = Stopwatch::default();
    loop {
        highgui::wait_key(1)?;
        s.reset();
        let mut frame = Mat::default();
        cam.read(&mut frame)?;
        debug!(target: "timings", "[Timing] Frame read: {:?}", s.elapsed_and_reset());

        let face_result = &face_model.run(&frame)?[0];
        debug!(target: "timings", "[Timing] Face model run total: {:?}", s.elapsed_and_reset());
        if face_result.bboxes.is_none() {
            debug!("No face detected");
            highgui::imshow(window_out, &frame)?;
            continue
        }
        debug!("{face_result:?}");
        let detected_face = cube.find_face(&frame, face_result)?;
        debug!(target: "timings", "[Timing] Face result processing: {:?}", s.elapsed_and_reset());
        if detected_face.is_none() {
            debug!("No face detected");
            highgui::imshow(window_out, &frame)?;
            continue
        }
        let detected_face = detected_face.unwrap();
        let facelet_result = &facelet_model.run(&frame)?[0];
        debug!(target: "timings", "[Timing] Facelet model run total: {:?}", s.elapsed_and_reset());

        cube.ingest_frame(&frame, detected_face, facelet_result)?;

        for b in facelet_result.bboxes.iter().flat_map(|x|x.iter()).cloned() {
            let rect = Rect::new(b.xmin() as i32, b.ymin() as i32, b.width() as i32, b.height() as i32);
            opencv::imgproc::rectangle_def(&mut frame, rect, (0, 255, 0).into())?;
        }
        for b in face_result.bboxes.iter().flat_map(|x|x.iter()).cloned() {
            let rect = Rect::new(b.xmin() as i32, b.ymin() as i32, b.width() as i32, b.height() as i32);
            opencv::imgproc::rectangle_def(&mut frame, rect, (0, 0, 255).into())?;
        }
        print_cube_state(&mut frame, &cube);

        debug!(target: "timings", "[Timing] Draw result: {:?}", s.elapsed_and_reset());

        highgui::imshow(window_out, &frame)?;
    }

    Ok(())
}

fn print_cube_state<const N: usize>(mut frame: &mut Mat, cube: &CubePredictionNxN<N>) -> anyhow::Result<()> {
    let colors = [(255, 0, 0), (0, 255, 0), (0, 180, 255), (0, 0, 255), (255, 255, 255),
        (0, 255, 255)];
    for i in 0..6 {
        let offset_x = 0;
        let offset_y = 0;
        let tile_size = 10;
        let face_spacing = 5;

        for x in 0..cube.get_n() {
            for y in 0..cube.get_n() {
                let (cid, conf) = cube.faces[i].facelets[x][y].get_best();
                let color = if conf > 0.7 {
                    colors[cid]
                } else {
                    (200, 200, 200)
                };
                let x_start = offset_x + tile_size * x;
                let y_start = offset_y + tile_size * (i * cube.get_n() + y) + face_spacing * i;

                let rect = Rect::new(x_start as i32, y_start as i32, tile_size as i32, tile_size as i32);
                opencv::imgproc::rectangle(&mut frame, rect, color.into(), FILLED, LINE_8, 0)?;
            }
        }
    }
    Ok(())
}
