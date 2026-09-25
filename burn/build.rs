fn main() {
    gst_plugin_version_helper::info();

    // CubeCL's buildid crate references __mh_execute_header, which only exists in
    // executables. This plugin is a dylib, so alias to the dylib Mach-O header.
    #[cfg(target_os = "macos")]
    {
        println!(
            "cargo:rustc-link-arg-cdylib=-Wl,-alias,__mh_dylib_header,__mh_execute_header"
        );
    }
}
