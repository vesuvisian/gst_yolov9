# GStreamer Plugin w/ Burn
This package contains a non-working GStreamer plugin that attempts to load a Burn model. This is a minimal example for debugging purposes. Its structure is based on that of [gst-plugins-rs](https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs).

## Plugins

This package currently provides a single plugin and element.

- `burn`: Elements that use Burn
  - `yolov9`: Load a yolov9 model

## Development

Development can be done either locally if you have Rust available, or in a Docker container.

To build the plugins, you must install cargo-c:

```bash
cargo install cargo-c
```

Then, you can build all of the plugins or individual ones, respectively, with the following commands:

```bash
cargo cbuild
cargo cbuild -p gst-burn
```

This builds them in 'debug' mode. To build them in 'release' mode, which takes longer but results in more optimized binaries, add the `--release` flag.

After being built, the plugins are located in `target/<system architecture>/debug` or `target/<system architecture>/release`. You can either move them to where your other GStreamer plugins are installed or add this location to your path with something like

```bash
export GST_PLUGIN_PATH=target/aarch64-apple-darwin/debug:$GST_PLUGIN_PATH
```

Then they can be used like normal:

```bash
gst-inspect-1.0 yolov9
gst-launch-1.0 videotestsrc ! yolov9 ! fakesink
```
