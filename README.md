# GStreamer Plugin w/ Burn
This package contains a minimal GStreamer plugin with an element that loads and runs a YOLOv9c object detection model with Burn. Its structure is based on that of [gst-plugins-rs](https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs).

## Plugins

This package currently provides a single plugin and element.

- `burn`: Elements that use Burn
  - `yolov9`: Run a yolov9 model

## Development

Development can be done either locally if you have Rust available, or in a Docker container.

To build the plugin, you must install cargo-c:

```bash
cargo install cargo-c
```

Then, you can the plugin with either of the following commands:

```bash
cargo cbuild
cargo cbuild -p gst-burn
```

This builds it in 'debug' mode. To build it in 'release' mode, which takes longer but results in more optimized binaries, add the `--release` flag.

After being built, the plugin is located in `target/<system architecture>/debug` or `target/<system architecture>/release`. You can either move it to where your other GStreamer plugins are installed or add this location to your path with something like

```bash
export GST_PLUGIN_PATH=target/aarch64-apple-darwin/debug:$GST_PLUGIN_PATH
```

The element can be used like normal:

```bash
gst-inspect-1.0 yolov9
gst-launch-1.0 avfvideosrc ! videoconvert ! yolov9 ! autovideosink sync=0
```
