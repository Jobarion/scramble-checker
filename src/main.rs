use std::error::Error;

use opencv::{highgui, prelude::*, videoio};
use opencv::core::Rect;

use scramble_checker::{OrtEP, YOLOv8};

fn main() -> Result<(), Box<dyn Error>> {
    let window_out = "video output";
    highgui::named_window(window_out, highgui::WINDOW_AUTOSIZE)?;
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?; // 0 is the default camera

    let opened = videoio::VideoCapture::is_opened(&cam)?;
    if !opened {
        panic!("Unable to open default camera!");
    }


    let mut model = YOLOv8::new(OrtEP::Cuda(0), "best.onnx".to_string(), 0.3, 0.45)?;
    model.summary(); // model info

    loop {
        let mut frame = Mat::default();
        cam.read(&mut frame)?;
        let result = model.run(&frame)?;

        for b in result.get(0).unwrap().iter().cloned() {
            let rect = Rect::new(b.xmin() as i32, b.ymin() as i32, b.width() as i32, b.height() as i32);
            opencv::imgproc::rectangle_def(&mut frame, rect, (0, 255, 0).into())?;
        }

        highgui::imshow(window_out, &frame)?;
        highgui::wait_key(1)?;
    }

    Ok(())
}
