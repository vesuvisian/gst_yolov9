#![allow(clippy::non_send_fields_in_send_ty, unused_doc_comments)]
#![recursion_limit = "256"]

use gst::glib;

mod yolov9;

fn plugin_init(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    yolov9::register(plugin)?;
    Ok(())
}

gst::plugin_define!(
    burn,
    env!("CARGO_PKG_DESCRIPTION"),
    plugin_init,
    concat!(env!("CARGO_PKG_VERSION")), //, "-", env!("COMMIT_ID")),
    "Proprietary",
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_REPOSITORY")
);
