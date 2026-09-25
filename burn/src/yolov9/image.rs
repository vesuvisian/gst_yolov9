use gst_video::{VideoFormat, VideoFrameRef, prelude::*};
use image::{ImageBuffer, Rgb};

/// A 32-bit floating point RGB image. Values are normalized to [0, 1].
pub type Rgb32FImage = ImageBuffer<Rgb<f32>, Vec<f32>>;

/// Convert an RGB video frame plane into a normalized f32 RGB image.
pub fn convert_video_frame_to_image(
    frame: &VideoFrameRef<&mut gst::BufferRef>,
) -> Result<Rgb32FImage, String> {
    let width = frame.width();
    let height = frame.height();
    let format = frame.format();
    if format != VideoFormat::Rgb {
        return Err(format!("unsupported pixel format: {format:?}"));
    }

    let stride = frame.plane_stride()[0] as usize;
    let data = frame
        .plane_data(0)
        .map_err(|_| "missing frame plane data".to_string())?;

    let row_bytes = width as usize * 3;
    let mut out = Vec::with_capacity(row_bytes * height as usize);
    for y in 0..height as usize {
        let row = &data[y * stride..y * stride + row_bytes];
        for &v in row {
            out.push(v as f32 / 255.0);
        }
    }

    ImageBuffer::from_raw(width, height, out)
        .ok_or_else(|| "failed to build RGB f32 image".to_string())
}
