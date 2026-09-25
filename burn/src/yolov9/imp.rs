use std::sync::{LazyLock, Mutex};

use burn::prelude::Device;
use burn::tensor::DeviceKind;
use gst::glib;
use gst::subclass::prelude::*;
use gst_video::prelude::*;
use gst_video::subclass::prelude::*;
use image::{ImageBuffer, Rgb};
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;

use super::image::convert_video_frame_to_image;
use super::inference_engine::InferenceEngine;

static DEVICE: LazyLock<Device> = LazyLock::new(|| {
    #[cfg(target_vendor = "apple")]
    {
        Device::metal(DeviceKind::DefaultDevice)
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        Device::flex()
    }
});

static CAT: LazyLock<gst::DebugCategory> = LazyLock::new(|| {
    gst::DebugCategory::new(
        "yolov9",
        gst::DebugColorFlags::empty(),
        Some("YOLOv9"),
    )
});

const BASE_COLORS: [(u8, u8, u8); 6] = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
];

#[derive(Default)]
struct State {
    // Box keeps YOLOv9c's 64-byte alignment off the GObject instance (glib max is 16).
    ie: Option<Box<InferenceEngine>>,
}

#[derive(Default)]
pub struct Yolov9 {
    state: Mutex<State>,
}

#[glib::object_subclass]
impl ObjectSubclass for Yolov9 {
    const NAME: &'static str = "GstYolov9";
    type Type = super::Yolov9;
    type ParentType = gst_video::VideoFilter;
}

impl ObjectImpl for Yolov9 {}

impl GstObjectImpl for Yolov9 {}

impl ElementImpl for Yolov9 {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static ELEMENT_METADATA: LazyLock<gst::subclass::ElementMetadata> = LazyLock::new(|| {
            gst::subclass::ElementMetadata::new(
                "YOLOv9",
                "Filter/Video",
                "Run YOLOv9 inference and overlay bounding boxes",
                "Andrew Martin",
            )
        });
        Some(&*ELEMENT_METADATA)
    }

    fn pad_templates() -> &'static [gst::PadTemplate] {
        static PAD_TEMPLATES: LazyLock<Vec<gst::PadTemplate>> = LazyLock::new(|| {
            let video_caps = gst_video::VideoCapsBuilder::new()
                .format(gst_video::VideoFormat::Rgb)
                .build();

            let sink = gst::PadTemplate::new(
                "sink",
                gst::PadDirection::Sink,
                gst::PadPresence::Always,
                &video_caps,
            )
            .unwrap();
            let src = gst::PadTemplate::new(
                "src",
                gst::PadDirection::Src,
                gst::PadPresence::Always,
                &video_caps,
            )
            .unwrap();
            vec![sink, src]
        });
        PAD_TEMPLATES.as_ref()
    }
}

impl BaseTransformImpl for Yolov9 {
    const MODE: gst_base::subclass::BaseTransformMode =
        gst_base::subclass::BaseTransformMode::AlwaysInPlace;
    const PASSTHROUGH_ON_SAME_CAPS: bool = false;
    const TRANSFORM_IP_ON_PASSTHROUGH: bool = true;

    fn start(&self) -> Result<(), gst::ErrorMessage> {
        let mut state = self.state.lock().unwrap();
        if state.ie.is_none() {
            gst::info!(CAT, imp = self, "Loading model");
            state.ie = Some(Box::new(InferenceEngine::new(&*DEVICE)));
        }
        Ok(())
    }
}

impl VideoFilterImpl for Yolov9 {
    fn transform_frame_ip(
        &self,
        frame: &mut gst_video::VideoFrameRef<&mut gst::BufferRef>,
    ) -> Result<gst::FlowSuccess, gst::FlowError> {
        let bboxes = {
            let mut state = self.state.lock().unwrap();
            let Some(ie) = state.ie.as_mut() else {
                return Ok(gst::FlowSuccess::Ok);
            };

            let img = match convert_video_frame_to_image(frame) {
                Ok(img) => img,
                Err(e) => {
                    gst::error!(CAT, imp = self, "Failed to convert frame: {}", e);
                    return Ok(gst::FlowSuccess::Ok);
                }
            };

            ie.infer(&img)
        };

        if bboxes.is_empty() {
            return Ok(gst::FlowSuccess::Ok);
        }

        let width = frame.width();
        let height = frame.height();
        let stride = frame.plane_stride()[0] as usize;
        let data = match frame.plane_data_mut(0) {
            Ok(d) => d,
            Err(_) => {
                gst::error!(CAT, imp = self, "Missing writable plane data");
                return Ok(gst::FlowSuccess::Ok);
            }
        };

        let row_bytes = width as usize * 3;
        let mut packed = if stride == row_bytes {
            data.to_vec()
        } else {
            let mut buf = Vec::with_capacity(row_bytes * height as usize);
            for y in 0..height as usize {
                buf.extend_from_slice(&data[y * stride..y * stride + row_bytes]);
            }
            buf
        };

        {
            let Some(mut img) =
                ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, packed.as_mut_slice())
            else {
                gst::error!(CAT, imp = self, "Failed to wrap frame as ImageBuffer");
                return Ok(gst::FlowSuccess::Ok);
            };

            for &(x, y, w, h, class_id) in &bboxes {
                if w == 0 || h == 0 {
                    continue;
                }
                let (r, g, b) = BASE_COLORS[class_id.unsigned_abs() as usize % BASE_COLORS.len()];
                draw_hollow_rect_mut(
                    &mut img,
                    Rect::at(x as i32, y as i32).of_size(w, h),
                    Rgb([r, g, b]),
                );
            }
        }

        if stride == row_bytes {
            data.copy_from_slice(&packed);
        } else {
            for y in 0..height as usize {
                data[y * stride..y * stride + row_bytes]
                    .copy_from_slice(&packed[y * row_bytes..(y + 1) * row_bytes]);
            }
        }

        Ok(gst::FlowSuccess::Ok)
    }
}
