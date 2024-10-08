use log::{debug, trace};
use opencv::core::{Mat, Rect};
use opencv::highgui;
use opencv::videoio::{VideoCapture, VideoCaptureTrait, VideoWriter, VideoWriterTrait};
use crate::detector::PuzzleDetector;
use anyhow::Result;
use crate::util::Stopwatch;
use crate::{Frame, YOLOv8};

pub struct DetectionSession<'a, 'b> {
    pub video_input: VideoCapture,
    pub face_model: &'a YOLOv8,
    pub facelet_model: &'b YOLOv8,
    pub detector: Box<dyn PuzzleDetector>,
    pub outputs: Vec<Output>,
}

pub enum Output {
    Window(String),
    File(VideoWriter)
}

impl Output {
    pub fn write_frame(&mut self, frame: &Mat) -> Result<()> {
        match self {
            Output::Window(name) => highgui::imshow(name.as_str(), frame)?,
            Output::File(vout) => vout.write(frame)?,
        };
        Ok(())
    }
}

impl <'a, 'b> DetectionSession<'a, 'b> {

    const WRONG_QUICK_REJECT_THRESHOLD: f32 = 0.15;
    const UNKNOWN_THRESHOLD: f32 = 0.15;
    const WRONG_REJECT_THRESHOLD: f32 = 0.05;

    pub fn reset(&mut self) {
        self.detector.reset()
    }

    pub fn read_frame(&mut self) -> Result<Option<Mat>> {
        let mut frame = Mat::default();
        if self.video_input.read(&mut frame)? {
            Ok(Some(frame))
        } else {
            Ok(None)
        }
    }

    pub fn draw_frame(&mut self, func: impl FnOnce() -> Result<Mat>) -> Result<()> {
        if !self.outputs.is_empty() {
            let frame = func()?;
            for out in self.outputs.iter_mut() {
                out.write_frame(&frame)?;
            }
        }
        Ok(())
    }

    pub fn process_frame(&mut self) -> Result<(bool, Frame)> {
        let mut s = Stopwatch::default();
        let mut frame = Mat::default();

        self.video_input.read(&mut frame)?;
        debug!(target: "timings", "[Timing] Frame read: {:?}", s.elapsed_and_reset());

        let mut frame = Frame::new(frame);

        let face_result = &self.face_model.run(&frame.input)?[0];
        debug!(target: "timings", "[Timing] Face model run total: {:?}", s.elapsed_and_reset());
        if face_result.bboxes.is_none() {
            debug!("No face detected");
            return Ok((false, frame))
        }

        let detected_face = self.detector.find_face(&mut frame, face_result)?;
        debug!(target: "timings", "[Timing] Face result processing: {:?}", s.elapsed_and_reset());
        if detected_face.is_none() {
            debug!("No face detected");
            return Ok((false, frame))
        }

        let detected_face = detected_face.unwrap();
        let facelet_result = &self.facelet_model.run(&frame.input)?[0];
        debug!(target: "timings", "[Timing] Facelet model run total: {:?}", s.elapsed_and_reset());

        self.detector.ingest_frame(&mut frame, detected_face, facelet_result)?;
        debug!(target: "timings", "[Timing] Ingest frame: {:?}", s.elapsed_and_reset());

        if !self.outputs.is_empty() {
            for b in facelet_result.bboxes.iter().flat_map(|x|x.iter()).cloned() {
                let rect = Rect::new(b.xmin() as i32, b.ymin() as i32, b.width() as i32, b.height() as i32);
                opencv::imgproc::rectangle_def(&mut frame.output, rect, (0, 255, 0).into())?;
            }
            for b in face_result.bboxes.iter().flat_map(|x|x.iter()).cloned() {
                let rect = Rect::new(b.xmin() as i32, b.ymin() as i32, b.width() as i32, b.height() as i32);
                opencv::imgproc::rectangle_def(&mut frame.output, rect, (0, 0, 255).into())?;
            }
        }
        debug!(target: "timings", "[Timing] Draw result: {:?}", s.elapsed_and_reset());
        Ok((true, frame))
    }

    pub fn get_confidence(&self) -> f32 {
        let mut s = Stopwatch::default();
        let (r, w, u) = self.detector.compare_to_puzzle();
        debug!(target: "timings", "[Timing] Comparing puzzle: {:?}", s.elapsed_and_reset());

        let total = r + w + u;
        let w = w as f32 / total as f32;
        let u = u as f32 / total as f32;

        // We don't have logic to calculate confidence values, so we just output 100%, 50% or 0%
        if u <= Self::UNKNOWN_THRESHOLD {
            if w > Self::WRONG_REJECT_THRESHOLD {
                0.0
            } else {
                1.0
            }
        } else if w > Self::WRONG_QUICK_REJECT_THRESHOLD {
            0.0
        } else {
            0.5
        }
    }
}