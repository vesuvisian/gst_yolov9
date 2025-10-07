use std::sync::{LazyLock, Mutex};

// Note: Candle and Wgpu backends have been attempted as well
use burn::backend::{ndarray::NdArrayDevice, NdArray};
use burn::prelude::Module;
use burn::record::Recorder;
use gst::glib;
use gst::prelude::*;
use gst::subclass::prelude::*;
use gst_video::subclass::prelude::*;

use super::yolov9c::YOLOv9c;

type MyBackend = NdArray<f32, i32>;
static DEVICE: LazyLock<NdArrayDevice> = LazyLock::new(|| NdArrayDevice::default());

const DEFAULT_MODEL_PATH: &str = "burn/src/yolov9/model.mpk";

#[derive(Debug, Clone)]
struct Settings {
    model_path: String,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            model_path: DEFAULT_MODEL_PATH.to_string(),
        }
    }
}

// #[derive(Default)]
// struct State {}

#[derive(Default)]
pub struct Yolov9 {
    settings: Mutex<Settings>,
    // state: Arc<Mutex<State>>,
}

// impl Yolov9 {}

#[glib::object_subclass]
impl ObjectSubclass for Yolov9 {
    const NAME: &'static str = "GstYolov9";
    type Type = super::Yolov9;
    type ParentType = gst_video::VideoFilter;
}

impl ObjectImpl for Yolov9 {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPERTIES: LazyLock<Vec<glib::ParamSpec>> = LazyLock::new(|| {
            vec![glib::ParamSpecString::builder("model-path")
                .nick("Model path")
                .blurb("Path to the model .mpk file")
                .default_value(DEFAULT_MODEL_PATH)
                .build()]
        });
        PROPERTIES.as_ref()
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        match pspec.name() {
            "model-path" => {
                let mut settings = self.settings.lock().unwrap();
                settings.model_path = value.get().expect("type checked upstream");
            }
            _ => unimplemented!(),
        }
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        let settings = self.settings.lock().unwrap();
        match pspec.name() {
            "model-path" => settings.model_path.to_value(),
            _ => unimplemented!(),
        }
    }
}

impl GstObjectImpl for Yolov9 {}

impl ElementImpl for Yolov9 {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static ELEMENT_METADATA: LazyLock<gst::subclass::ElementMetadata> = LazyLock::new(|| {
            gst::subclass::ElementMetadata::new(
                "YOLOv9",
                "Filter/Video",
                "Perform inference with a YOLOv9 model using Burn",
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

            let video_pad_template = gst::PadTemplate::new(
                "sink",
                gst::PadDirection::Sink,
                gst::PadPresence::Always,
                &video_caps,
            )
            .unwrap();

            let src_pad_template = gst::PadTemplate::new(
                "src",
                gst::PadDirection::Src,
                gst::PadPresence::Always,
                &video_caps,
            )
            .unwrap();

            vec![video_pad_template, src_pad_template]
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
        // Note, loading the model has also been attempted above in set_property and below in transform_frame_ip

        let settings = self.settings.lock().unwrap().clone();

        let path = &settings.model_path;
        assert!(
            std::path::Path::new(&path).exists(),
            "File does not exist: {}",
            path
        );
        let device = DEVICE.clone();

        let model = YOLOv9c::<MyBackend>::new(&device);
        println!("model: {:p}", &model);
        let recorder =
            burn::record::NamedMpkBytesRecorder::<burn::record::FullPrecisionSettings>::new();
        println!("recorder: {:p}", &recorder);

        // Non-working attempt to load the model; leads to SIGBUS
        // Note, embedding the model file in the binary with include_bytes! and loading with the BinFileRecorder has also been attempted
        let record: super::yolov9c::YOLOv9cRecord<MyBackend> = recorder
            .load(path.as_str().into(), &device)
            .expect("Record file to exist.");
        println!("record: {:p}", &record);

        // Haven't gotten this far yet
        let model = model.load_record(record);
        println!("model: {:p}", &model);

        // If the model were to successfully load, it would be stored in state

        Ok(())
    }
}

impl VideoFilterImpl for Yolov9 {
    fn transform_frame_ip(
        &self,
        _frame: &mut gst_video::VideoFrameRef<&mut gst::BufferRef>,
    ) -> Result<gst::FlowSuccess, gst::FlowError> {
        // Here is where the inference would be performed on the frame data
        Ok(gst::FlowSuccess::Ok)
    }
}
