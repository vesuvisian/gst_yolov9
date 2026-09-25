use gst::glib;
use gst::prelude::*;

mod image;
mod imp;
mod inference_engine;
mod yolov9c;

glib::wrapper! {
    pub struct Yolov9(ObjectSubclass<imp::Yolov9>) @extends gst_video::VideoFilter, gst_base::BaseTransform, gst::Element, gst::Object;
}

pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(
        Some(plugin),
        "yolov9",
        gst::Rank::NONE,
        Yolov9::static_type(),
    )
}
