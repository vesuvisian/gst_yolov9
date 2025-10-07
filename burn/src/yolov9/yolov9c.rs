use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::nn::conv::Conv3d;
use burn::nn::conv::Conv3dConfig;
use burn::nn::interpolate::Interpolate2d;
use burn::nn::interpolate::Interpolate2dConfig;
use burn::nn::interpolate::InterpolateMode;
use burn::nn::pool::AvgPool2d;
use burn::nn::pool::AvgPool2dConfig;
use burn::nn::pool::MaxPool2d;
use burn::nn::pool::MaxPool2dConfig;
use burn::nn::PaddingConfig2d;
use burn::nn::PaddingConfig3d;
use burn::prelude::*;
use burn::record::FullPrecisionSettings;
use burn::record::Recorder;

#[derive(Module, Debug)]
pub struct YOLOv9c<B: Backend> {
    constant1: burn::module::Param<Tensor<B, 2>>,
    conv2d1: Conv2d<B>,
    conv2d2: Conv2d<B>,
    conv2d3: Conv2d<B>,
    conv2d4: Conv2d<B>,
    conv2d5: Conv2d<B>,
    conv2d6: Conv2d<B>,
    conv2d7: Conv2d<B>,
    conv2d8: Conv2d<B>,
    conv2d9: Conv2d<B>,
    conv2d10: Conv2d<B>,
    conv2d11: Conv2d<B>,
    conv2d12: Conv2d<B>,
    conv2d13: Conv2d<B>,
    conv2d14: Conv2d<B>,
    conv2d15: Conv2d<B>,
    conv2d16: Conv2d<B>,
    conv2d17: Conv2d<B>,
    conv2d18: Conv2d<B>,
    averagepool2d1: AvgPool2d,
    conv2d19: Conv2d<B>,
    maxpool2d1: MaxPool2d,
    conv2d20: Conv2d<B>,
    conv2d21: Conv2d<B>,
    conv2d22: Conv2d<B>,
    conv2d23: Conv2d<B>,
    conv2d24: Conv2d<B>,
    conv2d25: Conv2d<B>,
    conv2d26: Conv2d<B>,
    conv2d27: Conv2d<B>,
    conv2d28: Conv2d<B>,
    conv2d29: Conv2d<B>,
    conv2d30: Conv2d<B>,
    conv2d31: Conv2d<B>,
    conv2d32: Conv2d<B>,
    conv2d33: Conv2d<B>,
    conv2d34: Conv2d<B>,
    conv2d35: Conv2d<B>,
    conv2d36: Conv2d<B>,
    averagepool2d2: AvgPool2d,
    conv2d37: Conv2d<B>,
    maxpool2d2: MaxPool2d,
    conv2d38: Conv2d<B>,
    conv2d39: Conv2d<B>,
    conv2d40: Conv2d<B>,
    conv2d41: Conv2d<B>,
    conv2d42: Conv2d<B>,
    conv2d43: Conv2d<B>,
    conv2d44: Conv2d<B>,
    conv2d45: Conv2d<B>,
    conv2d46: Conv2d<B>,
    conv2d47: Conv2d<B>,
    conv2d48: Conv2d<B>,
    conv2d49: Conv2d<B>,
    conv2d50: Conv2d<B>,
    conv2d51: Conv2d<B>,
    conv2d52: Conv2d<B>,
    conv2d53: Conv2d<B>,
    conv2d54: Conv2d<B>,
    averagepool2d3: AvgPool2d,
    conv2d55: Conv2d<B>,
    maxpool2d3: MaxPool2d,
    conv2d56: Conv2d<B>,
    conv2d57: Conv2d<B>,
    conv2d58: Conv2d<B>,
    conv2d59: Conv2d<B>,
    conv2d60: Conv2d<B>,
    conv2d61: Conv2d<B>,
    conv2d62: Conv2d<B>,
    conv2d63: Conv2d<B>,
    conv2d64: Conv2d<B>,
    conv2d65: Conv2d<B>,
    conv2d66: Conv2d<B>,
    conv2d67: Conv2d<B>,
    conv2d68: Conv2d<B>,
    conv2d69: Conv2d<B>,
    conv2d70: Conv2d<B>,
    conv2d71: Conv2d<B>,
    conv2d72: Conv2d<B>,
    conv2d73: Conv2d<B>,
    maxpool2d4: MaxPool2d,
    maxpool2d5: MaxPool2d,
    maxpool2d6: MaxPool2d,
    conv2d74: Conv2d<B>,
    resize1: Interpolate2d,
    conv2d75: Conv2d<B>,
    conv2d76: Conv2d<B>,
    conv2d77: Conv2d<B>,
    conv2d78: Conv2d<B>,
    conv2d79: Conv2d<B>,
    conv2d80: Conv2d<B>,
    conv2d81: Conv2d<B>,
    conv2d82: Conv2d<B>,
    conv2d83: Conv2d<B>,
    conv2d84: Conv2d<B>,
    conv2d85: Conv2d<B>,
    conv2d86: Conv2d<B>,
    conv2d87: Conv2d<B>,
    conv2d88: Conv2d<B>,
    conv2d89: Conv2d<B>,
    conv2d90: Conv2d<B>,
    resize2: Interpolate2d,
    conv2d91: Conv2d<B>,
    conv2d92: Conv2d<B>,
    conv2d93: Conv2d<B>,
    conv2d94: Conv2d<B>,
    conv2d95: Conv2d<B>,
    conv2d96: Conv2d<B>,
    conv2d97: Conv2d<B>,
    conv2d98: Conv2d<B>,
    conv2d99: Conv2d<B>,
    conv2d100: Conv2d<B>,
    conv2d101: Conv2d<B>,
    conv2d102: Conv2d<B>,
    conv2d103: Conv2d<B>,
    conv2d104: Conv2d<B>,
    conv2d105: Conv2d<B>,
    conv2d106: Conv2d<B>,
    averagepool2d4: AvgPool2d,
    conv2d107: Conv2d<B>,
    maxpool2d7: MaxPool2d,
    conv2d108: Conv2d<B>,
    conv2d109: Conv2d<B>,
    conv2d110: Conv2d<B>,
    conv2d111: Conv2d<B>,
    conv2d112: Conv2d<B>,
    conv2d113: Conv2d<B>,
    conv2d114: Conv2d<B>,
    conv2d115: Conv2d<B>,
    conv2d116: Conv2d<B>,
    conv2d117: Conv2d<B>,
    conv2d118: Conv2d<B>,
    conv2d119: Conv2d<B>,
    conv2d120: Conv2d<B>,
    conv2d121: Conv2d<B>,
    conv2d122: Conv2d<B>,
    conv2d123: Conv2d<B>,
    conv2d124: Conv2d<B>,
    averagepool2d5: AvgPool2d,
    conv2d125: Conv2d<B>,
    maxpool2d8: MaxPool2d,
    conv2d126: Conv2d<B>,
    conv2d127: Conv2d<B>,
    conv2d128: Conv2d<B>,
    conv2d129: Conv2d<B>,
    conv2d130: Conv2d<B>,
    conv2d131: Conv2d<B>,
    conv2d132: Conv2d<B>,
    conv2d133: Conv2d<B>,
    conv2d134: Conv2d<B>,
    conv2d135: Conv2d<B>,
    conv2d136: Conv2d<B>,
    conv2d137: Conv2d<B>,
    conv2d138: Conv2d<B>,
    conv2d139: Conv2d<B>,
    conv2d140: Conv2d<B>,
    conv2d141: Conv2d<B>,
    conv2d142: Conv2d<B>,
    conv2d143: Conv2d<B>,
    conv2d144: Conv2d<B>,
    conv2d145: Conv2d<B>,
    conv2d146: Conv2d<B>,
    conv2d147: Conv2d<B>,
    conv2d148: Conv2d<B>,
    conv3d1: Conv3d<B>,
    conv2d149: Conv2d<B>,
    conv2d150: Conv2d<B>,
    conv2d151: Conv2d<B>,
    conv2d152: Conv2d<B>,
    conv2d153: Conv2d<B>,
    conv2d154: Conv2d<B>,
    conv3d2: Conv3d<B>,
    conv2d155: Conv2d<B>,
    conv2d156: Conv2d<B>,
    conv2d157: Conv2d<B>,
    conv2d158: Conv2d<B>,
    conv2d159: Conv2d<B>,
    conv2d160: Conv2d<B>,
    conv3d3: Conv3d<B>,
    constant94: burn::module::Param<Tensor<B, 3>>,
    constant101: burn::module::Param<Tensor<B, 2>>,
    phantom: core::marker::PhantomData<B>,
    device: burn::module::Ignored<B::Device>,
}

impl<B: Backend> YOLOv9c<B> {
    pub fn from_file(file: &str, device: &B::Device) -> Self {
        let record = burn::record::NamedMpkFileRecorder::<FullPrecisionSettings>::new()
            .load(file.into(), device)
            .expect("Record file to exist.");
        Self::new(device).load_record(record)
    }

    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let constant1: burn::module::Param<Tensor<B, 2>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 2>::zeros([8400, 2], device),
            device.clone(),
            false,
        );
        let conv2d1 = Conv2dConfig::new([3, 64], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d2 = Conv2dConfig::new([64, 128], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d3 = Conv2dConfig::new([128, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d4 = Conv2dConfig::new([64, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d5 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d6 = Conv2dConfig::new([32, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d7 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d8 = Conv2dConfig::new([64, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d9 = Conv2dConfig::new([64, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d10 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d11 = Conv2dConfig::new([64, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d12 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d13 = Conv2dConfig::new([32, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d14 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d15 = Conv2dConfig::new([64, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d16 = Conv2dConfig::new([64, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d17 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d18 = Conv2dConfig::new([256, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let averagepool2d1 = AvgPool2dConfig::new([2, 2])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_count_include_pad(true)
            .init();
        let conv2d19 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d1 = MaxPool2dConfig::new([3, 3])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .init();
        let conv2d20 = Conv2dConfig::new([128, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d21 = Conv2dConfig::new([256, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d22 = Conv2dConfig::new([128, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d23 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d24 = Conv2dConfig::new([64, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d25 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d26 = Conv2dConfig::new([128, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d27 = Conv2dConfig::new([128, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d28 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d29 = Conv2dConfig::new([128, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d30 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d31 = Conv2dConfig::new([64, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d32 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d33 = Conv2dConfig::new([128, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d34 = Conv2dConfig::new([128, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d35 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d36 = Conv2dConfig::new([512, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let averagepool2d2 = AvgPool2dConfig::new([2, 2])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_count_include_pad(true)
            .init();
        let conv2d37 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d2 = MaxPool2dConfig::new([3, 3])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .init();
        let conv2d38 = Conv2dConfig::new([256, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d39 = Conv2dConfig::new([512, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d40 = Conv2dConfig::new([256, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d41 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d42 = Conv2dConfig::new([128, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d43 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d44 = Conv2dConfig::new([256, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d45 = Conv2dConfig::new([256, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d46 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d47 = Conv2dConfig::new([256, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d48 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d49 = Conv2dConfig::new([128, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d50 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d51 = Conv2dConfig::new([256, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d52 = Conv2dConfig::new([256, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d53 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d54 = Conv2dConfig::new([1024, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let averagepool2d3 = AvgPool2dConfig::new([2, 2])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_count_include_pad(true)
            .init();
        let conv2d55 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d3 = MaxPool2dConfig::new([3, 3])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .init();
        let conv2d56 = Conv2dConfig::new([256, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d57 = Conv2dConfig::new([512, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d58 = Conv2dConfig::new([256, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d59 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d60 = Conv2dConfig::new([128, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d61 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d62 = Conv2dConfig::new([256, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d63 = Conv2dConfig::new([256, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d64 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d65 = Conv2dConfig::new([256, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d66 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d67 = Conv2dConfig::new([128, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d68 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d69 = Conv2dConfig::new([256, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d70 = Conv2dConfig::new([256, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d71 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d72 = Conv2dConfig::new([1024, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d73 = Conv2dConfig::new([512, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d4 = MaxPool2dConfig::new([5, 5])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2))
            .with_dilation([1, 1])
            .init();
        let maxpool2d5 = MaxPool2dConfig::new([5, 5])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2))
            .with_dilation([1, 1])
            .init();
        let maxpool2d6 = MaxPool2dConfig::new([5, 5])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2))
            .with_dilation([1, 1])
            .init();
        let conv2d74 = Conv2dConfig::new([1024, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let resize1 = Interpolate2dConfig::new()
            .with_output_size(None)
            .with_scale_factor(Some([2.0, 2.0]))
            .with_mode(InterpolateMode::Nearest)
            .init();
        let conv2d75 = Conv2dConfig::new([1024, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d76 = Conv2dConfig::new([256, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d77 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d78 = Conv2dConfig::new([128, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d79 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d80 = Conv2dConfig::new([256, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d81 = Conv2dConfig::new([256, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d82 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d83 = Conv2dConfig::new([256, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d84 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d85 = Conv2dConfig::new([128, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d86 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d87 = Conv2dConfig::new([256, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d88 = Conv2dConfig::new([256, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d89 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d90 = Conv2dConfig::new([1024, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let resize2 = Interpolate2dConfig::new()
            .with_output_size(None)
            .with_scale_factor(Some([2.0, 2.0]))
            .with_mode(InterpolateMode::Nearest)
            .init();
        let conv2d91 = Conv2dConfig::new([1024, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d92 = Conv2dConfig::new([128, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d93 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d94 = Conv2dConfig::new([64, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d95 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d96 = Conv2dConfig::new([128, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d97 = Conv2dConfig::new([128, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d98 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d99 = Conv2dConfig::new([128, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d100 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d101 = Conv2dConfig::new([64, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d102 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d103 = Conv2dConfig::new([128, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d104 = Conv2dConfig::new([128, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d105 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d106 = Conv2dConfig::new([512, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let averagepool2d4 = AvgPool2dConfig::new([2, 2])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_count_include_pad(true)
            .init();
        let conv2d107 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d7 = MaxPool2dConfig::new([3, 3])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .init();
        let conv2d108 = Conv2dConfig::new([128, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d109 = Conv2dConfig::new([768, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d110 = Conv2dConfig::new([256, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d111 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d112 = Conv2dConfig::new([128, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d113 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d114 = Conv2dConfig::new([256, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d115 = Conv2dConfig::new([256, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d116 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d117 = Conv2dConfig::new([256, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d118 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d119 = Conv2dConfig::new([128, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d120 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d121 = Conv2dConfig::new([256, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d122 = Conv2dConfig::new([256, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d123 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d124 = Conv2dConfig::new([1024, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let averagepool2d5 = AvgPool2dConfig::new([2, 2])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_count_include_pad(true)
            .init();
        let conv2d125 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d8 = MaxPool2dConfig::new([3, 3])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .init();
        let conv2d126 = Conv2dConfig::new([256, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d127 = Conv2dConfig::new([1024, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d128 = Conv2dConfig::new([256, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d129 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d130 = Conv2dConfig::new([128, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d131 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d132 = Conv2dConfig::new([256, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d133 = Conv2dConfig::new([256, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d134 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d135 = Conv2dConfig::new([256, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d136 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d137 = Conv2dConfig::new([128, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d138 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d139 = Conv2dConfig::new([256, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d140 = Conv2dConfig::new([256, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d141 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d142 = Conv2dConfig::new([1024, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d143 = Conv2dConfig::new([256, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d144 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(4)
            .with_bias(true)
            .init(device);
        let conv2d145 = Conv2dConfig::new([64, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(4)
            .with_bias(true)
            .init(device);
        let conv2d146 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d147 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d148 = Conv2dConfig::new([256, 1], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d1 = Conv3dConfig::new([16, 1], [1, 1, 1])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Valid)
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let conv2d149 = Conv2dConfig::new([512, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d150 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(4)
            .with_bias(true)
            .init(device);
        let conv2d151 = Conv2dConfig::new([64, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(4)
            .with_bias(true)
            .init(device);
        let conv2d152 = Conv2dConfig::new([512, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d153 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d154 = Conv2dConfig::new([256, 1], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d2 = Conv3dConfig::new([16, 1], [1, 1, 1])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Valid)
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let conv2d155 = Conv2dConfig::new([512, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d156 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(4)
            .with_bias(true)
            .init(device);
        let conv2d157 = Conv2dConfig::new([64, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(4)
            .with_bias(true)
            .init(device);
        let conv2d158 = Conv2dConfig::new([512, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d159 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d160 = Conv2dConfig::new([256, 1], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d3 = Conv3dConfig::new([16, 1], [1, 1, 1])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Valid)
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant94: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 8400, 1], device),
            device.clone(),
            false,
        );
        let constant101: burn::module::Param<Tensor<B, 2>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 2>::zeros([8400, 2], device),
            device.clone(),
            false,
        );
        Self {
            constant1,
            conv2d1,
            conv2d2,
            conv2d3,
            conv2d4,
            conv2d5,
            conv2d6,
            conv2d7,
            conv2d8,
            conv2d9,
            conv2d10,
            conv2d11,
            conv2d12,
            conv2d13,
            conv2d14,
            conv2d15,
            conv2d16,
            conv2d17,
            conv2d18,
            averagepool2d1,
            conv2d19,
            maxpool2d1,
            conv2d20,
            conv2d21,
            conv2d22,
            conv2d23,
            conv2d24,
            conv2d25,
            conv2d26,
            conv2d27,
            conv2d28,
            conv2d29,
            conv2d30,
            conv2d31,
            conv2d32,
            conv2d33,
            conv2d34,
            conv2d35,
            conv2d36,
            averagepool2d2,
            conv2d37,
            maxpool2d2,
            conv2d38,
            conv2d39,
            conv2d40,
            conv2d41,
            conv2d42,
            conv2d43,
            conv2d44,
            conv2d45,
            conv2d46,
            conv2d47,
            conv2d48,
            conv2d49,
            conv2d50,
            conv2d51,
            conv2d52,
            conv2d53,
            conv2d54,
            averagepool2d3,
            conv2d55,
            maxpool2d3,
            conv2d56,
            conv2d57,
            conv2d58,
            conv2d59,
            conv2d60,
            conv2d61,
            conv2d62,
            conv2d63,
            conv2d64,
            conv2d65,
            conv2d66,
            conv2d67,
            conv2d68,
            conv2d69,
            conv2d70,
            conv2d71,
            conv2d72,
            conv2d73,
            maxpool2d4,
            maxpool2d5,
            maxpool2d6,
            conv2d74,
            resize1,
            conv2d75,
            conv2d76,
            conv2d77,
            conv2d78,
            conv2d79,
            conv2d80,
            conv2d81,
            conv2d82,
            conv2d83,
            conv2d84,
            conv2d85,
            conv2d86,
            conv2d87,
            conv2d88,
            conv2d89,
            conv2d90,
            resize2,
            conv2d91,
            conv2d92,
            conv2d93,
            conv2d94,
            conv2d95,
            conv2d96,
            conv2d97,
            conv2d98,
            conv2d99,
            conv2d100,
            conv2d101,
            conv2d102,
            conv2d103,
            conv2d104,
            conv2d105,
            conv2d106,
            averagepool2d4,
            conv2d107,
            maxpool2d7,
            conv2d108,
            conv2d109,
            conv2d110,
            conv2d111,
            conv2d112,
            conv2d113,
            conv2d114,
            conv2d115,
            conv2d116,
            conv2d117,
            conv2d118,
            conv2d119,
            conv2d120,
            conv2d121,
            conv2d122,
            conv2d123,
            conv2d124,
            averagepool2d5,
            conv2d125,
            maxpool2d8,
            conv2d126,
            conv2d127,
            conv2d128,
            conv2d129,
            conv2d130,
            conv2d131,
            conv2d132,
            conv2d133,
            conv2d134,
            conv2d135,
            conv2d136,
            conv2d137,
            conv2d138,
            conv2d139,
            conv2d140,
            conv2d141,
            conv2d142,
            conv2d143,
            conv2d144,
            conv2d145,
            conv2d146,
            conv2d147,
            conv2d148,
            conv3d1,
            conv2d149,
            conv2d150,
            conv2d151,
            conv2d152,
            conv2d153,
            conv2d154,
            conv3d2,
            conv2d155,
            conv2d156,
            conv2d157,
            conv2d158,
            conv2d159,
            conv2d160,
            conv3d3,
            constant94,
            constant101,
            phantom: core::marker::PhantomData,
            device: burn::module::Ignored(device.clone()),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, input1: Tensor<B, 4>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let constant1_out1 = self.constant1.val();
        let conv2d1_out1 = self.conv2d1.forward(input1);
        let sigmoid1_out1 = burn::tensor::activation::sigmoid(conv2d1_out1.clone());
        let mul1_out1 = conv2d1_out1.mul(sigmoid1_out1);
        let conv2d2_out1 = self.conv2d2.forward(mul1_out1);
        let sigmoid2_out1 = burn::tensor::activation::sigmoid(conv2d2_out1.clone());
        let mul2_out1 = conv2d2_out1.mul(sigmoid2_out1);
        let conv2d3_out1 = self.conv2d3.forward(mul2_out1);
        let sigmoid3_out1 = burn::tensor::activation::sigmoid(conv2d3_out1.clone());
        let mul3_out1 = conv2d3_out1.mul(sigmoid3_out1);
        let shape1_out1: [i64; 4] = mul3_out1.clone().dims()[0..4]
            .iter()
            .map(|&x| x as i64)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let constant4_out1: [i64; 1] = [1i64];
        let gather1_out1: [i64; 1usize] = constant4_out1
            .iter()
            .map(|&idx| {
                let actual_idx = if idx < 0 {
                    (shape1_out1.len() as i64 + idx) as usize
                } else {
                    idx as usize
                };
                shape1_out1[actual_idx]
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let constant6_out1: [i64; 1] = [1i64];
        let add1_out1 = {
            let mut result = gather1_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant6_out1.iter()) {
                *result_item = result_item.saturating_add(*rhs_item);
            }
            result
        };
        let constant7_out1: [i64; 1] = [2i64];
        let div1_out1 = {
            let mut result = add1_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant7_out1.iter()) {
                *result_item = if *rhs_item != 0 {
                    *result_item / *rhs_item
                } else {
                    *result_item
                };
            }
            result
        };
        let constant8_out1: [i64; 1] = [1i64];
        let mul4_out1 = {
            let mut result = div1_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant8_out1.iter()) {
                *result_item = result_item.saturating_mul(*rhs_item);
            }
            result
        };
        let slice1_out1 = mul3_out1.clone().slice(s![.., 0..mul4_out1[0], .., ..]);
        let constant9_out1: [i64; 1] = [2i64];
        let mul5_out1 = {
            let mut result = div1_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant9_out1.iter()) {
                *result_item = result_item.saturating_mul(*rhs_item);
            }
            result
        };
        let slice2_out1 = mul3_out1.slice(s![.., mul4_out1[0]..mul5_out1[0], .., ..]);
        let conv2d4_out1 = self.conv2d4.forward(slice2_out1.clone());
        let sigmoid4_out1 = burn::tensor::activation::sigmoid(conv2d4_out1.clone());
        let mul6_out1 = conv2d4_out1.mul(sigmoid4_out1);
        let conv2d5_out1 = self.conv2d5.forward(mul6_out1.clone());
        let conv2d6_out1 = self.conv2d6.forward(mul6_out1.clone());
        let add2_out1 = conv2d5_out1.add(conv2d6_out1);
        let sigmoid5_out1 = burn::tensor::activation::sigmoid(add2_out1.clone());
        let mul7_out1 = add2_out1.mul(sigmoid5_out1);
        let conv2d7_out1 = self.conv2d7.forward(mul7_out1);
        let sigmoid6_out1 = burn::tensor::activation::sigmoid(conv2d7_out1.clone());
        let mul8_out1 = conv2d7_out1.mul(sigmoid6_out1);
        let add3_out1 = mul6_out1.add(mul8_out1);
        let conv2d8_out1 = self.conv2d8.forward(slice2_out1.clone());
        let sigmoid7_out1 = burn::tensor::activation::sigmoid(conv2d8_out1.clone());
        let mul9_out1 = conv2d8_out1.mul(sigmoid7_out1);
        let concat1_out1 = burn::tensor::Tensor::cat([add3_out1, mul9_out1].into(), 1);
        let conv2d9_out1 = self.conv2d9.forward(concat1_out1);
        let sigmoid8_out1 = burn::tensor::activation::sigmoid(conv2d9_out1.clone());
        let mul10_out1 = conv2d9_out1.mul(sigmoid8_out1);
        let conv2d10_out1 = self.conv2d10.forward(mul10_out1);
        let sigmoid9_out1 = burn::tensor::activation::sigmoid(conv2d10_out1.clone());
        let mul11_out1 = conv2d10_out1.mul(sigmoid9_out1);
        let conv2d11_out1 = self.conv2d11.forward(mul11_out1.clone());
        let sigmoid10_out1 = burn::tensor::activation::sigmoid(conv2d11_out1.clone());
        let mul12_out1 = conv2d11_out1.mul(sigmoid10_out1);
        let conv2d12_out1 = self.conv2d12.forward(mul12_out1.clone());
        let conv2d13_out1 = self.conv2d13.forward(mul12_out1.clone());
        let add4_out1 = conv2d12_out1.add(conv2d13_out1);
        let sigmoid11_out1 = burn::tensor::activation::sigmoid(add4_out1.clone());
        let mul13_out1 = add4_out1.mul(sigmoid11_out1);
        let conv2d14_out1 = self.conv2d14.forward(mul13_out1);
        let sigmoid12_out1 = burn::tensor::activation::sigmoid(conv2d14_out1.clone());
        let mul14_out1 = conv2d14_out1.mul(sigmoid12_out1);
        let add5_out1 = mul12_out1.add(mul14_out1);
        let conv2d15_out1 = self.conv2d15.forward(mul11_out1.clone());
        let sigmoid13_out1 = burn::tensor::activation::sigmoid(conv2d15_out1.clone());
        let mul15_out1 = conv2d15_out1.mul(sigmoid13_out1);
        let concat2_out1 = burn::tensor::Tensor::cat([add5_out1, mul15_out1].into(), 1);
        let conv2d16_out1 = self.conv2d16.forward(concat2_out1);
        let sigmoid14_out1 = burn::tensor::activation::sigmoid(conv2d16_out1.clone());
        let mul16_out1 = conv2d16_out1.mul(sigmoid14_out1);
        let conv2d17_out1 = self.conv2d17.forward(mul16_out1);
        let sigmoid15_out1 = burn::tensor::activation::sigmoid(conv2d17_out1.clone());
        let mul17_out1 = conv2d17_out1.mul(sigmoid15_out1);
        let concat3_out1 =
            burn::tensor::Tensor::cat([slice1_out1, slice2_out1, mul11_out1, mul17_out1].into(), 1);
        let conv2d18_out1 = self.conv2d18.forward(concat3_out1);
        let sigmoid16_out1 = burn::tensor::activation::sigmoid(conv2d18_out1.clone());
        let mul18_out1 = conv2d18_out1.mul(sigmoid16_out1);
        let averagepool2d1_out1 = self.averagepool2d1.forward(mul18_out1);
        let shape2_out1: [i64; 4] = averagepool2d1_out1.clone().dims()[0..4]
            .iter()
            .map(|&x| x as i64)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let constant10_out1: [i64; 1] = [1i64];
        let gather2_out1: [i64; 1usize] = constant10_out1
            .iter()
            .map(|&idx| {
                let actual_idx = if idx < 0 {
                    (shape2_out1.len() as i64 + idx) as usize
                } else {
                    idx as usize
                };
                shape2_out1[actual_idx]
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let constant12_out1: [i64; 1] = [1i64];
        let add6_out1 = {
            let mut result = gather2_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant12_out1.iter()) {
                *result_item = result_item.saturating_add(*rhs_item);
            }
            result
        };
        let constant13_out1: [i64; 1] = [2i64];
        let div2_out1 = {
            let mut result = add6_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant13_out1.iter()) {
                *result_item = if *rhs_item != 0 {
                    *result_item / *rhs_item
                } else {
                    *result_item
                };
            }
            result
        };
        let constant14_out1: [i64; 1] = [1i64];
        let mul19_out1 = {
            let mut result = div2_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant14_out1.iter()) {
                *result_item = result_item.saturating_mul(*rhs_item);
            }
            result
        };
        let slice3_out1 = averagepool2d1_out1
            .clone()
            .slice(s![.., 0..mul19_out1[0], .., ..]);
        let constant15_out1: [i64; 1] = [2i64];
        let mul20_out1 = {
            let mut result = div2_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant15_out1.iter()) {
                *result_item = result_item.saturating_mul(*rhs_item);
            }
            result
        };
        let slice4_out1 = averagepool2d1_out1.slice(s![.., mul19_out1[0]..mul20_out1[0], .., ..]);
        let conv2d19_out1 = self.conv2d19.forward(slice3_out1);
        let sigmoid17_out1 = burn::tensor::activation::sigmoid(conv2d19_out1.clone());
        let mul21_out1 = conv2d19_out1.mul(sigmoid17_out1);
        let maxpool2d1_out1 = self.maxpool2d1.forward(slice4_out1);
        let conv2d20_out1 = self.conv2d20.forward(maxpool2d1_out1);
        let sigmoid18_out1 = burn::tensor::activation::sigmoid(conv2d20_out1.clone());
        let mul22_out1 = conv2d20_out1.mul(sigmoid18_out1);
        let concat4_out1 = burn::tensor::Tensor::cat([mul21_out1, mul22_out1].into(), 1);
        let conv2d21_out1 = self.conv2d21.forward(concat4_out1);
        let sigmoid19_out1 = burn::tensor::activation::sigmoid(conv2d21_out1.clone());
        let mul23_out1 = conv2d21_out1.mul(sigmoid19_out1);
        let shape3_out1: [i64; 4] = mul23_out1.clone().dims()[0..4]
            .iter()
            .map(|&x| x as i64)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let constant16_out1: [i64; 1] = [1i64];
        let gather3_out1: [i64; 1usize] = constant16_out1
            .iter()
            .map(|&idx| {
                let actual_idx = if idx < 0 {
                    (shape3_out1.len() as i64 + idx) as usize
                } else {
                    idx as usize
                };
                shape3_out1[actual_idx]
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let constant18_out1: [i64; 1] = [1i64];
        let add7_out1 = {
            let mut result = gather3_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant18_out1.iter()) {
                *result_item = result_item.saturating_add(*rhs_item);
            }
            result
        };
        let constant19_out1: [i64; 1] = [2i64];
        let div3_out1 = {
            let mut result = add7_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant19_out1.iter()) {
                *result_item = if *rhs_item != 0 {
                    *result_item / *rhs_item
                } else {
                    *result_item
                };
            }
            result
        };
        let constant20_out1: [i64; 1] = [1i64];
        let mul24_out1 = {
            let mut result = div3_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant20_out1.iter()) {
                *result_item = result_item.saturating_mul(*rhs_item);
            }
            result
        };
        let slice5_out1 = mul23_out1.clone().slice(s![.., 0..mul24_out1[0], .., ..]);
        let constant21_out1: [i64; 1] = [2i64];
        let mul25_out1 = {
            let mut result = div3_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant21_out1.iter()) {
                *result_item = result_item.saturating_mul(*rhs_item);
            }
            result
        };
        let slice6_out1 = mul23_out1.slice(s![.., mul24_out1[0]..mul25_out1[0], .., ..]);
        let conv2d22_out1 = self.conv2d22.forward(slice6_out1.clone());
        let sigmoid20_out1 = burn::tensor::activation::sigmoid(conv2d22_out1.clone());
        let mul26_out1 = conv2d22_out1.mul(sigmoid20_out1);
        let conv2d23_out1 = self.conv2d23.forward(mul26_out1.clone());
        let conv2d24_out1 = self.conv2d24.forward(mul26_out1.clone());
        let add8_out1 = conv2d23_out1.add(conv2d24_out1);
        let sigmoid21_out1 = burn::tensor::activation::sigmoid(add8_out1.clone());
        let mul27_out1 = add8_out1.mul(sigmoid21_out1);
        let conv2d25_out1 = self.conv2d25.forward(mul27_out1);
        let sigmoid22_out1 = burn::tensor::activation::sigmoid(conv2d25_out1.clone());
        let mul28_out1 = conv2d25_out1.mul(sigmoid22_out1);
        let add9_out1 = mul26_out1.add(mul28_out1);
        let conv2d26_out1 = self.conv2d26.forward(slice6_out1.clone());
        let sigmoid23_out1 = burn::tensor::activation::sigmoid(conv2d26_out1.clone());
        let mul29_out1 = conv2d26_out1.mul(sigmoid23_out1);
        let concat5_out1 = burn::tensor::Tensor::cat([add9_out1, mul29_out1].into(), 1);
        let conv2d27_out1 = self.conv2d27.forward(concat5_out1);
        let sigmoid24_out1 = burn::tensor::activation::sigmoid(conv2d27_out1.clone());
        let mul30_out1 = conv2d27_out1.mul(sigmoid24_out1);
        let conv2d28_out1 = self.conv2d28.forward(mul30_out1);
        let sigmoid25_out1 = burn::tensor::activation::sigmoid(conv2d28_out1.clone());
        let mul31_out1 = conv2d28_out1.mul(sigmoid25_out1);
        let conv2d29_out1 = self.conv2d29.forward(mul31_out1.clone());
        let sigmoid26_out1 = burn::tensor::activation::sigmoid(conv2d29_out1.clone());
        let mul32_out1 = conv2d29_out1.mul(sigmoid26_out1);
        let conv2d30_out1 = self.conv2d30.forward(mul32_out1.clone());
        let conv2d31_out1 = self.conv2d31.forward(mul32_out1.clone());
        let add10_out1 = conv2d30_out1.add(conv2d31_out1);
        let sigmoid27_out1 = burn::tensor::activation::sigmoid(add10_out1.clone());
        let mul33_out1 = add10_out1.mul(sigmoid27_out1);
        let conv2d32_out1 = self.conv2d32.forward(mul33_out1);
        let sigmoid28_out1 = burn::tensor::activation::sigmoid(conv2d32_out1.clone());
        let mul34_out1 = conv2d32_out1.mul(sigmoid28_out1);
        let add11_out1 = mul32_out1.add(mul34_out1);
        let conv2d33_out1 = self.conv2d33.forward(mul31_out1.clone());
        let sigmoid29_out1 = burn::tensor::activation::sigmoid(conv2d33_out1.clone());
        let mul35_out1 = conv2d33_out1.mul(sigmoid29_out1);
        let concat6_out1 = burn::tensor::Tensor::cat([add11_out1, mul35_out1].into(), 1);
        let conv2d34_out1 = self.conv2d34.forward(concat6_out1);
        let sigmoid30_out1 = burn::tensor::activation::sigmoid(conv2d34_out1.clone());
        let mul36_out1 = conv2d34_out1.mul(sigmoid30_out1);
        let conv2d35_out1 = self.conv2d35.forward(mul36_out1);
        let sigmoid31_out1 = burn::tensor::activation::sigmoid(conv2d35_out1.clone());
        let mul37_out1 = conv2d35_out1.mul(sigmoid31_out1);
        let concat7_out1 =
            burn::tensor::Tensor::cat([slice5_out1, slice6_out1, mul31_out1, mul37_out1].into(), 1);
        let conv2d36_out1 = self.conv2d36.forward(concat7_out1);
        let sigmoid32_out1 = burn::tensor::activation::sigmoid(conv2d36_out1.clone());
        let mul38_out1 = conv2d36_out1.mul(sigmoid32_out1);
        let averagepool2d2_out1 = self.averagepool2d2.forward(mul38_out1.clone());
        let shape4_out1: [i64; 4] = averagepool2d2_out1.clone().dims()[0..4]
            .iter()
            .map(|&x| x as i64)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let constant22_out1: [i64; 1] = [1i64];
        let gather4_out1: [i64; 1usize] = constant22_out1
            .iter()
            .map(|&idx| {
                let actual_idx = if idx < 0 {
                    (shape4_out1.len() as i64 + idx) as usize
                } else {
                    idx as usize
                };
                shape4_out1[actual_idx]
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let constant24_out1: [i64; 1] = [1i64];
        let add12_out1 = {
            let mut result = gather4_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant24_out1.iter()) {
                *result_item = result_item.saturating_add(*rhs_item);
            }
            result
        };
        let constant25_out1: [i64; 1] = [2i64];
        let div4_out1 = {
            let mut result = add12_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant25_out1.iter()) {
                *result_item = if *rhs_item != 0 {
                    *result_item / *rhs_item
                } else {
                    *result_item
                };
            }
            result
        };
        let constant26_out1: [i64; 1] = [1i64];
        let mul39_out1 = {
            let mut result = div4_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant26_out1.iter()) {
                *result_item = result_item.saturating_mul(*rhs_item);
            }
            result
        };
        let slice7_out1 = averagepool2d2_out1
            .clone()
            .slice(s![.., 0..mul39_out1[0], .., ..]);
        let constant27_out1: [i64; 1] = [2i64];
        let mul40_out1 = {
            let mut result = div4_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant27_out1.iter()) {
                *result_item = result_item.saturating_mul(*rhs_item);
            }
            result
        };
        let slice8_out1 = averagepool2d2_out1.slice(s![.., mul39_out1[0]..mul40_out1[0], .., ..]);
        let conv2d37_out1 = self.conv2d37.forward(slice7_out1);
        let sigmoid33_out1 = burn::tensor::activation::sigmoid(conv2d37_out1.clone());
        let mul41_out1 = conv2d37_out1.mul(sigmoid33_out1);
        let maxpool2d2_out1 = self.maxpool2d2.forward(slice8_out1);
        let conv2d38_out1 = self.conv2d38.forward(maxpool2d2_out1);
        let sigmoid34_out1 = burn::tensor::activation::sigmoid(conv2d38_out1.clone());
        let mul42_out1 = conv2d38_out1.mul(sigmoid34_out1);
        let concat8_out1 = burn::tensor::Tensor::cat([mul41_out1, mul42_out1].into(), 1);
        let conv2d39_out1 = self.conv2d39.forward(concat8_out1);
        let sigmoid35_out1 = burn::tensor::activation::sigmoid(conv2d39_out1.clone());
        let mul43_out1 = conv2d39_out1.mul(sigmoid35_out1);
        let shape5_out1: [i64; 4] = mul43_out1.clone().dims()[0..4]
            .iter()
            .map(|&x| x as i64)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let constant28_out1: [i64; 1] = [1i64];
        let gather5_out1: [i64; 1usize] = constant28_out1
            .iter()
            .map(|&idx| {
                let actual_idx = if idx < 0 {
                    (shape5_out1.len() as i64 + idx) as usize
                } else {
                    idx as usize
                };
                shape5_out1[actual_idx]
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let constant30_out1: [i64; 1] = [1i64];
        let add13_out1 = {
            let mut result = gather5_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant30_out1.iter()) {
                *result_item = result_item.saturating_add(*rhs_item);
            }
            result
        };
        let constant31_out1: [i64; 1] = [2i64];
        let div5_out1 = {
            let mut result = add13_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant31_out1.iter()) {
                *result_item = if *rhs_item != 0 {
                    *result_item / *rhs_item
                } else {
                    *result_item
                };
            }
            result
        };
        let constant32_out1: [i64; 1] = [1i64];
        let mul44_out1 = {
            let mut result = div5_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant32_out1.iter()) {
                *result_item = result_item.saturating_mul(*rhs_item);
            }
            result
        };
        let slice9_out1 = mul43_out1.clone().slice(s![.., 0..mul44_out1[0], .., ..]);
        let constant33_out1: [i64; 1] = [2i64];
        let mul45_out1 = {
            let mut result = div5_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant33_out1.iter()) {
                *result_item = result_item.saturating_mul(*rhs_item);
            }
            result
        };
        let slice10_out1 = mul43_out1.slice(s![.., mul44_out1[0]..mul45_out1[0], .., ..]);
        let conv2d40_out1 = self.conv2d40.forward(slice10_out1.clone());
        let sigmoid36_out1 = burn::tensor::activation::sigmoid(conv2d40_out1.clone());
        let mul46_out1 = conv2d40_out1.mul(sigmoid36_out1);
        let conv2d41_out1 = self.conv2d41.forward(mul46_out1.clone());
        let conv2d42_out1 = self.conv2d42.forward(mul46_out1.clone());
        let add14_out1 = conv2d41_out1.add(conv2d42_out1);
        let sigmoid37_out1 = burn::tensor::activation::sigmoid(add14_out1.clone());
        let mul47_out1 = add14_out1.mul(sigmoid37_out1);
        let conv2d43_out1 = self.conv2d43.forward(mul47_out1);
        let sigmoid38_out1 = burn::tensor::activation::sigmoid(conv2d43_out1.clone());
        let mul48_out1 = conv2d43_out1.mul(sigmoid38_out1);
        let add15_out1 = mul46_out1.add(mul48_out1);
        let conv2d44_out1 = self.conv2d44.forward(slice10_out1.clone());
        let sigmoid39_out1 = burn::tensor::activation::sigmoid(conv2d44_out1.clone());
        let mul49_out1 = conv2d44_out1.mul(sigmoid39_out1);
        let concat9_out1 = burn::tensor::Tensor::cat([add15_out1, mul49_out1].into(), 1);
        let conv2d45_out1 = self.conv2d45.forward(concat9_out1);
        let sigmoid40_out1 = burn::tensor::activation::sigmoid(conv2d45_out1.clone());
        let mul50_out1 = conv2d45_out1.mul(sigmoid40_out1);
        let conv2d46_out1 = self.conv2d46.forward(mul50_out1);
        let sigmoid41_out1 = burn::tensor::activation::sigmoid(conv2d46_out1.clone());
        let mul51_out1 = conv2d46_out1.mul(sigmoid41_out1);
        let conv2d47_out1 = self.conv2d47.forward(mul51_out1.clone());
        let sigmoid42_out1 = burn::tensor::activation::sigmoid(conv2d47_out1.clone());
        let mul52_out1 = conv2d47_out1.mul(sigmoid42_out1);
        let conv2d48_out1 = self.conv2d48.forward(mul52_out1.clone());
        let conv2d49_out1 = self.conv2d49.forward(mul52_out1.clone());
        let add16_out1 = conv2d48_out1.add(conv2d49_out1);
        let sigmoid43_out1 = burn::tensor::activation::sigmoid(add16_out1.clone());
        let mul53_out1 = add16_out1.mul(sigmoid43_out1);
        let conv2d50_out1 = self.conv2d50.forward(mul53_out1);
        let sigmoid44_out1 = burn::tensor::activation::sigmoid(conv2d50_out1.clone());
        let mul54_out1 = conv2d50_out1.mul(sigmoid44_out1);
        let add17_out1 = mul52_out1.add(mul54_out1);
        let conv2d51_out1 = self.conv2d51.forward(mul51_out1.clone());
        let sigmoid45_out1 = burn::tensor::activation::sigmoid(conv2d51_out1.clone());
        let mul55_out1 = conv2d51_out1.mul(sigmoid45_out1);
        let concat10_out1 = burn::tensor::Tensor::cat([add17_out1, mul55_out1].into(), 1);
        let conv2d52_out1 = self.conv2d52.forward(concat10_out1);
        let sigmoid46_out1 = burn::tensor::activation::sigmoid(conv2d52_out1.clone());
        let mul56_out1 = conv2d52_out1.mul(sigmoid46_out1);
        let conv2d53_out1 = self.conv2d53.forward(mul56_out1);
        let sigmoid47_out1 = burn::tensor::activation::sigmoid(conv2d53_out1.clone());
        let mul57_out1 = conv2d53_out1.mul(sigmoid47_out1);
        let concat11_out1 = burn::tensor::Tensor::cat(
            [slice9_out1, slice10_out1, mul51_out1, mul57_out1].into(),
            1,
        );
        let conv2d54_out1 = self.conv2d54.forward(concat11_out1);
        let sigmoid48_out1 = burn::tensor::activation::sigmoid(conv2d54_out1.clone());
        let mul58_out1 = conv2d54_out1.mul(sigmoid48_out1);
        let averagepool2d3_out1 = self.averagepool2d3.forward(mul58_out1.clone());
        let shape6_out1: [i64; 4] = averagepool2d3_out1.clone().dims()[0..4]
            .iter()
            .map(|&x| x as i64)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let constant34_out1: [i64; 1] = [1i64];
        let gather6_out1: [i64; 1usize] = constant34_out1
            .iter()
            .map(|&idx| {
                let actual_idx = if idx < 0 {
                    (shape6_out1.len() as i64 + idx) as usize
                } else {
                    idx as usize
                };
                shape6_out1[actual_idx]
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let constant36_out1: [i64; 1] = [1i64];
        let add18_out1 = {
            let mut result = gather6_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant36_out1.iter()) {
                *result_item = result_item.saturating_add(*rhs_item);
            }
            result
        };
        let constant37_out1: [i64; 1] = [2i64];
        let div6_out1 = {
            let mut result = add18_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant37_out1.iter()) {
                *result_item = if *rhs_item != 0 {
                    *result_item / *rhs_item
                } else {
                    *result_item
                };
            }
            result
        };
        let constant38_out1: [i64; 1] = [1i64];
        let mul59_out1 = {
            let mut result = div6_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant38_out1.iter()) {
                *result_item = result_item.saturating_mul(*rhs_item);
            }
            result
        };
        let slice11_out1 = averagepool2d3_out1
            .clone()
            .slice(s![.., 0..mul59_out1[0], .., ..]);
        let constant39_out1: [i64; 1] = [2i64];
        let mul60_out1 = {
            let mut result = div6_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant39_out1.iter()) {
                *result_item = result_item.saturating_mul(*rhs_item);
            }
            result
        };
        let slice12_out1 = averagepool2d3_out1.slice(s![.., mul59_out1[0]..mul60_out1[0], .., ..]);
        let conv2d55_out1 = self.conv2d55.forward(slice11_out1);
        let sigmoid49_out1 = burn::tensor::activation::sigmoid(conv2d55_out1.clone());
        let mul61_out1 = conv2d55_out1.mul(sigmoid49_out1);
        let maxpool2d3_out1 = self.maxpool2d3.forward(slice12_out1);
        let conv2d56_out1 = self.conv2d56.forward(maxpool2d3_out1);
        let sigmoid50_out1 = burn::tensor::activation::sigmoid(conv2d56_out1.clone());
        let mul62_out1 = conv2d56_out1.mul(sigmoid50_out1);
        let concat12_out1 = burn::tensor::Tensor::cat([mul61_out1, mul62_out1].into(), 1);
        let conv2d57_out1 = self.conv2d57.forward(concat12_out1);
        let sigmoid51_out1 = burn::tensor::activation::sigmoid(conv2d57_out1.clone());
        let mul63_out1 = conv2d57_out1.mul(sigmoid51_out1);
        let shape7_out1: [i64; 4] = mul63_out1.clone().dims()[0..4]
            .iter()
            .map(|&x| x as i64)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let constant40_out1: [i64; 1] = [1i64];
        let gather7_out1: [i64; 1usize] = constant40_out1
            .iter()
            .map(|&idx| {
                let actual_idx = if idx < 0 {
                    (shape7_out1.len() as i64 + idx) as usize
                } else {
                    idx as usize
                };
                shape7_out1[actual_idx]
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let constant42_out1: [i64; 1] = [1i64];
        let add19_out1 = {
            let mut result = gather7_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant42_out1.iter()) {
                *result_item = result_item.saturating_add(*rhs_item);
            }
            result
        };
        let constant43_out1: [i64; 1] = [2i64];
        let div7_out1 = {
            let mut result = add19_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant43_out1.iter()) {
                *result_item = if *rhs_item != 0 {
                    *result_item / *rhs_item
                } else {
                    *result_item
                };
            }
            result
        };
        let constant44_out1: [i64; 1] = [1i64];
        let mul64_out1 = {
            let mut result = div7_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant44_out1.iter()) {
                *result_item = result_item.saturating_mul(*rhs_item);
            }
            result
        };
        let slice13_out1 = mul63_out1.clone().slice(s![.., 0..mul64_out1[0], .., ..]);
        let constant45_out1: [i64; 1] = [2i64];
        let mul65_out1 = {
            let mut result = div7_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant45_out1.iter()) {
                *result_item = result_item.saturating_mul(*rhs_item);
            }
            result
        };
        let slice14_out1 = mul63_out1.slice(s![.., mul64_out1[0]..mul65_out1[0], .., ..]);
        let conv2d58_out1 = self.conv2d58.forward(slice14_out1.clone());
        let sigmoid52_out1 = burn::tensor::activation::sigmoid(conv2d58_out1.clone());
        let mul66_out1 = conv2d58_out1.mul(sigmoid52_out1);
        let conv2d59_out1 = self.conv2d59.forward(mul66_out1.clone());
        let conv2d60_out1 = self.conv2d60.forward(mul66_out1.clone());
        let add20_out1 = conv2d59_out1.add(conv2d60_out1);
        let sigmoid53_out1 = burn::tensor::activation::sigmoid(add20_out1.clone());
        let mul67_out1 = add20_out1.mul(sigmoid53_out1);
        let conv2d61_out1 = self.conv2d61.forward(mul67_out1);
        let sigmoid54_out1 = burn::tensor::activation::sigmoid(conv2d61_out1.clone());
        let mul68_out1 = conv2d61_out1.mul(sigmoid54_out1);
        let add21_out1 = mul66_out1.add(mul68_out1);
        let conv2d62_out1 = self.conv2d62.forward(slice14_out1.clone());
        let sigmoid55_out1 = burn::tensor::activation::sigmoid(conv2d62_out1.clone());
        let mul69_out1 = conv2d62_out1.mul(sigmoid55_out1);
        let concat13_out1 = burn::tensor::Tensor::cat([add21_out1, mul69_out1].into(), 1);
        let conv2d63_out1 = self.conv2d63.forward(concat13_out1);
        let sigmoid56_out1 = burn::tensor::activation::sigmoid(conv2d63_out1.clone());
        let mul70_out1 = conv2d63_out1.mul(sigmoid56_out1);
        let conv2d64_out1 = self.conv2d64.forward(mul70_out1);
        let sigmoid57_out1 = burn::tensor::activation::sigmoid(conv2d64_out1.clone());
        let mul71_out1 = conv2d64_out1.mul(sigmoid57_out1);
        let conv2d65_out1 = self.conv2d65.forward(mul71_out1.clone());
        let sigmoid58_out1 = burn::tensor::activation::sigmoid(conv2d65_out1.clone());
        let mul72_out1 = conv2d65_out1.mul(sigmoid58_out1);
        let conv2d66_out1 = self.conv2d66.forward(mul72_out1.clone());
        let conv2d67_out1 = self.conv2d67.forward(mul72_out1.clone());
        let add22_out1 = conv2d66_out1.add(conv2d67_out1);
        let sigmoid59_out1 = burn::tensor::activation::sigmoid(add22_out1.clone());
        let mul73_out1 = add22_out1.mul(sigmoid59_out1);
        let conv2d68_out1 = self.conv2d68.forward(mul73_out1);
        let sigmoid60_out1 = burn::tensor::activation::sigmoid(conv2d68_out1.clone());
        let mul74_out1 = conv2d68_out1.mul(sigmoid60_out1);
        let add23_out1 = mul72_out1.add(mul74_out1);
        let conv2d69_out1 = self.conv2d69.forward(mul71_out1.clone());
        let sigmoid61_out1 = burn::tensor::activation::sigmoid(conv2d69_out1.clone());
        let mul75_out1 = conv2d69_out1.mul(sigmoid61_out1);
        let concat14_out1 = burn::tensor::Tensor::cat([add23_out1, mul75_out1].into(), 1);
        let conv2d70_out1 = self.conv2d70.forward(concat14_out1);
        let sigmoid62_out1 = burn::tensor::activation::sigmoid(conv2d70_out1.clone());
        let mul76_out1 = conv2d70_out1.mul(sigmoid62_out1);
        let conv2d71_out1 = self.conv2d71.forward(mul76_out1);
        let sigmoid63_out1 = burn::tensor::activation::sigmoid(conv2d71_out1.clone());
        let mul77_out1 = conv2d71_out1.mul(sigmoid63_out1);
        let concat15_out1 = burn::tensor::Tensor::cat(
            [slice13_out1, slice14_out1, mul71_out1, mul77_out1].into(),
            1,
        );
        let conv2d72_out1 = self.conv2d72.forward(concat15_out1);
        let sigmoid64_out1 = burn::tensor::activation::sigmoid(conv2d72_out1.clone());
        let mul78_out1 = conv2d72_out1.mul(sigmoid64_out1);
        let conv2d73_out1 = self.conv2d73.forward(mul78_out1);
        let sigmoid65_out1 = burn::tensor::activation::sigmoid(conv2d73_out1.clone());
        let mul79_out1 = conv2d73_out1.mul(sigmoid65_out1);
        let maxpool2d4_out1 = self.maxpool2d4.forward(mul79_out1.clone());
        let maxpool2d5_out1 = self.maxpool2d5.forward(maxpool2d4_out1.clone());
        let maxpool2d6_out1 = self.maxpool2d6.forward(maxpool2d5_out1.clone());
        let concat16_out1 = burn::tensor::Tensor::cat(
            [
                mul79_out1,
                maxpool2d4_out1,
                maxpool2d5_out1,
                maxpool2d6_out1,
            ]
            .into(),
            1,
        );
        let conv2d74_out1 = self.conv2d74.forward(concat16_out1);
        let sigmoid66_out1 = burn::tensor::activation::sigmoid(conv2d74_out1.clone());
        let mul80_out1 = conv2d74_out1.mul(sigmoid66_out1);
        let resize1_out1 = self.resize1.forward(mul80_out1.clone());
        let concat17_out1 = burn::tensor::Tensor::cat([resize1_out1, mul58_out1].into(), 1);
        let conv2d75_out1 = self.conv2d75.forward(concat17_out1);
        let sigmoid67_out1 = burn::tensor::activation::sigmoid(conv2d75_out1.clone());
        let mul81_out1 = conv2d75_out1.mul(sigmoid67_out1);
        let shape8_out1: [i64; 4] = mul81_out1.clone().dims()[0..4]
            .iter()
            .map(|&x| x as i64)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let constant47_out1: [i64; 1] = [1i64];
        let gather8_out1: [i64; 1usize] = constant47_out1
            .iter()
            .map(|&idx| {
                let actual_idx = if idx < 0 {
                    (shape8_out1.len() as i64 + idx) as usize
                } else {
                    idx as usize
                };
                shape8_out1[actual_idx]
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let constant49_out1: [i64; 1] = [1i64];
        let add24_out1 = {
            let mut result = gather8_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant49_out1.iter()) {
                *result_item = result_item.saturating_add(*rhs_item);
            }
            result
        };
        let constant50_out1: [i64; 1] = [2i64];
        let div8_out1 = {
            let mut result = add24_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant50_out1.iter()) {
                *result_item = if *rhs_item != 0 {
                    *result_item / *rhs_item
                } else {
                    *result_item
                };
            }
            result
        };
        let constant51_out1: [i64; 1] = [1i64];
        let mul82_out1 = {
            let mut result = div8_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant51_out1.iter()) {
                *result_item = result_item.saturating_mul(*rhs_item);
            }
            result
        };
        let slice15_out1 = mul81_out1.clone().slice(s![.., 0..mul82_out1[0], .., ..]);
        let constant52_out1: [i64; 1] = [2i64];
        let mul83_out1 = {
            let mut result = div8_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant52_out1.iter()) {
                *result_item = result_item.saturating_mul(*rhs_item);
            }
            result
        };
        let slice16_out1 = mul81_out1.slice(s![.., mul82_out1[0]..mul83_out1[0], .., ..]);
        let conv2d76_out1 = self.conv2d76.forward(slice16_out1.clone());
        let sigmoid68_out1 = burn::tensor::activation::sigmoid(conv2d76_out1.clone());
        let mul84_out1 = conv2d76_out1.mul(sigmoid68_out1);
        let conv2d77_out1 = self.conv2d77.forward(mul84_out1.clone());
        let conv2d78_out1 = self.conv2d78.forward(mul84_out1.clone());
        let add25_out1 = conv2d77_out1.add(conv2d78_out1);
        let sigmoid69_out1 = burn::tensor::activation::sigmoid(add25_out1.clone());
        let mul85_out1 = add25_out1.mul(sigmoid69_out1);
        let conv2d79_out1 = self.conv2d79.forward(mul85_out1);
        let sigmoid70_out1 = burn::tensor::activation::sigmoid(conv2d79_out1.clone());
        let mul86_out1 = conv2d79_out1.mul(sigmoid70_out1);
        let add26_out1 = mul84_out1.add(mul86_out1);
        let conv2d80_out1 = self.conv2d80.forward(slice16_out1.clone());
        let sigmoid71_out1 = burn::tensor::activation::sigmoid(conv2d80_out1.clone());
        let mul87_out1 = conv2d80_out1.mul(sigmoid71_out1);
        let concat18_out1 = burn::tensor::Tensor::cat([add26_out1, mul87_out1].into(), 1);
        let conv2d81_out1 = self.conv2d81.forward(concat18_out1);
        let sigmoid72_out1 = burn::tensor::activation::sigmoid(conv2d81_out1.clone());
        let mul88_out1 = conv2d81_out1.mul(sigmoid72_out1);
        let conv2d82_out1 = self.conv2d82.forward(mul88_out1);
        let sigmoid73_out1 = burn::tensor::activation::sigmoid(conv2d82_out1.clone());
        let mul89_out1 = conv2d82_out1.mul(sigmoid73_out1);
        let conv2d83_out1 = self.conv2d83.forward(mul89_out1.clone());
        let sigmoid74_out1 = burn::tensor::activation::sigmoid(conv2d83_out1.clone());
        let mul90_out1 = conv2d83_out1.mul(sigmoid74_out1);
        let conv2d84_out1 = self.conv2d84.forward(mul90_out1.clone());
        let conv2d85_out1 = self.conv2d85.forward(mul90_out1.clone());
        let add27_out1 = conv2d84_out1.add(conv2d85_out1);
        let sigmoid75_out1 = burn::tensor::activation::sigmoid(add27_out1.clone());
        let mul91_out1 = add27_out1.mul(sigmoid75_out1);
        let conv2d86_out1 = self.conv2d86.forward(mul91_out1);
        let sigmoid76_out1 = burn::tensor::activation::sigmoid(conv2d86_out1.clone());
        let mul92_out1 = conv2d86_out1.mul(sigmoid76_out1);
        let add28_out1 = mul90_out1.add(mul92_out1);
        let conv2d87_out1 = self.conv2d87.forward(mul89_out1.clone());
        let sigmoid77_out1 = burn::tensor::activation::sigmoid(conv2d87_out1.clone());
        let mul93_out1 = conv2d87_out1.mul(sigmoid77_out1);
        let concat19_out1 = burn::tensor::Tensor::cat([add28_out1, mul93_out1].into(), 1);
        let conv2d88_out1 = self.conv2d88.forward(concat19_out1);
        let sigmoid78_out1 = burn::tensor::activation::sigmoid(conv2d88_out1.clone());
        let mul94_out1 = conv2d88_out1.mul(sigmoid78_out1);
        let conv2d89_out1 = self.conv2d89.forward(mul94_out1);
        let sigmoid79_out1 = burn::tensor::activation::sigmoid(conv2d89_out1.clone());
        let mul95_out1 = conv2d89_out1.mul(sigmoid79_out1);
        let concat20_out1 = burn::tensor::Tensor::cat(
            [slice15_out1, slice16_out1, mul89_out1, mul95_out1].into(),
            1,
        );
        let conv2d90_out1 = self.conv2d90.forward(concat20_out1);
        let sigmoid80_out1 = burn::tensor::activation::sigmoid(conv2d90_out1.clone());
        let mul96_out1 = conv2d90_out1.mul(sigmoid80_out1);
        let resize2_out1 = self.resize2.forward(mul96_out1.clone());
        let concat21_out1 = burn::tensor::Tensor::cat([resize2_out1, mul38_out1].into(), 1);
        let conv2d91_out1 = self.conv2d91.forward(concat21_out1);
        let sigmoid81_out1 = burn::tensor::activation::sigmoid(conv2d91_out1.clone());
        let mul97_out1 = conv2d91_out1.mul(sigmoid81_out1);
        let shape9_out1: [i64; 4] = mul97_out1.clone().dims()[0..4]
            .iter()
            .map(|&x| x as i64)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let constant54_out1: [i64; 1] = [1i64];
        let gather9_out1: [i64; 1usize] = constant54_out1
            .iter()
            .map(|&idx| {
                let actual_idx = if idx < 0 {
                    (shape9_out1.len() as i64 + idx) as usize
                } else {
                    idx as usize
                };
                shape9_out1[actual_idx]
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let constant56_out1: [i64; 1] = [1i64];
        let add29_out1 = {
            let mut result = gather9_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant56_out1.iter()) {
                *result_item = result_item.saturating_add(*rhs_item);
            }
            result
        };
        let constant57_out1: [i64; 1] = [2i64];
        let div9_out1 = {
            let mut result = add29_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant57_out1.iter()) {
                *result_item = if *rhs_item != 0 {
                    *result_item / *rhs_item
                } else {
                    *result_item
                };
            }
            result
        };
        let constant58_out1: [i64; 1] = [1i64];
        let mul98_out1 = {
            let mut result = div9_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant58_out1.iter()) {
                *result_item = result_item.saturating_mul(*rhs_item);
            }
            result
        };
        let slice17_out1 = mul97_out1.clone().slice(s![.., 0..mul98_out1[0], .., ..]);
        let constant59_out1: [i64; 1] = [2i64];
        let mul99_out1 = {
            let mut result = div9_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant59_out1.iter()) {
                *result_item = result_item.saturating_mul(*rhs_item);
            }
            result
        };
        let slice18_out1 = mul97_out1.slice(s![.., mul98_out1[0]..mul99_out1[0], .., ..]);
        let conv2d92_out1 = self.conv2d92.forward(slice18_out1.clone());
        let sigmoid82_out1 = burn::tensor::activation::sigmoid(conv2d92_out1.clone());
        let mul100_out1 = conv2d92_out1.mul(sigmoid82_out1);
        let conv2d93_out1 = self.conv2d93.forward(mul100_out1.clone());
        let conv2d94_out1 = self.conv2d94.forward(mul100_out1.clone());
        let add30_out1 = conv2d93_out1.add(conv2d94_out1);
        let sigmoid83_out1 = burn::tensor::activation::sigmoid(add30_out1.clone());
        let mul101_out1 = add30_out1.mul(sigmoid83_out1);
        let conv2d95_out1 = self.conv2d95.forward(mul101_out1);
        let sigmoid84_out1 = burn::tensor::activation::sigmoid(conv2d95_out1.clone());
        let mul102_out1 = conv2d95_out1.mul(sigmoid84_out1);
        let add31_out1 = mul100_out1.add(mul102_out1);
        let conv2d96_out1 = self.conv2d96.forward(slice18_out1.clone());
        let sigmoid85_out1 = burn::tensor::activation::sigmoid(conv2d96_out1.clone());
        let mul103_out1 = conv2d96_out1.mul(sigmoid85_out1);
        let concat22_out1 = burn::tensor::Tensor::cat([add31_out1, mul103_out1].into(), 1);
        let conv2d97_out1 = self.conv2d97.forward(concat22_out1);
        let sigmoid86_out1 = burn::tensor::activation::sigmoid(conv2d97_out1.clone());
        let mul104_out1 = conv2d97_out1.mul(sigmoid86_out1);
        let conv2d98_out1 = self.conv2d98.forward(mul104_out1);
        let sigmoid87_out1 = burn::tensor::activation::sigmoid(conv2d98_out1.clone());
        let mul105_out1 = conv2d98_out1.mul(sigmoid87_out1);
        let conv2d99_out1 = self.conv2d99.forward(mul105_out1.clone());
        let sigmoid88_out1 = burn::tensor::activation::sigmoid(conv2d99_out1.clone());
        let mul106_out1 = conv2d99_out1.mul(sigmoid88_out1);
        let conv2d100_out1 = self.conv2d100.forward(mul106_out1.clone());
        let conv2d101_out1 = self.conv2d101.forward(mul106_out1.clone());
        let add32_out1 = conv2d100_out1.add(conv2d101_out1);
        let sigmoid89_out1 = burn::tensor::activation::sigmoid(add32_out1.clone());
        let mul107_out1 = add32_out1.mul(sigmoid89_out1);
        let conv2d102_out1 = self.conv2d102.forward(mul107_out1);
        let sigmoid90_out1 = burn::tensor::activation::sigmoid(conv2d102_out1.clone());
        let mul108_out1 = conv2d102_out1.mul(sigmoid90_out1);
        let add33_out1 = mul106_out1.add(mul108_out1);
        let conv2d103_out1 = self.conv2d103.forward(mul105_out1.clone());
        let sigmoid91_out1 = burn::tensor::activation::sigmoid(conv2d103_out1.clone());
        let mul109_out1 = conv2d103_out1.mul(sigmoid91_out1);
        let concat23_out1 = burn::tensor::Tensor::cat([add33_out1, mul109_out1].into(), 1);
        let conv2d104_out1 = self.conv2d104.forward(concat23_out1);
        let sigmoid92_out1 = burn::tensor::activation::sigmoid(conv2d104_out1.clone());
        let mul110_out1 = conv2d104_out1.mul(sigmoid92_out1);
        let conv2d105_out1 = self.conv2d105.forward(mul110_out1);
        let sigmoid93_out1 = burn::tensor::activation::sigmoid(conv2d105_out1.clone());
        let mul111_out1 = conv2d105_out1.mul(sigmoid93_out1);
        let concat24_out1 = burn::tensor::Tensor::cat(
            [slice17_out1, slice18_out1, mul105_out1, mul111_out1].into(),
            1,
        );
        let conv2d106_out1 = self.conv2d106.forward(concat24_out1);
        let sigmoid94_out1 = burn::tensor::activation::sigmoid(conv2d106_out1.clone());
        let mul112_out1 = conv2d106_out1.mul(sigmoid94_out1);
        let averagepool2d4_out1 = self.averagepool2d4.forward(mul112_out1.clone());
        let shape10_out1: [i64; 4] = averagepool2d4_out1.clone().dims()[0..4]
            .iter()
            .map(|&x| x as i64)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let constant60_out1: [i64; 1] = [1i64];
        let gather10_out1: [i64; 1usize] = constant60_out1
            .iter()
            .map(|&idx| {
                let actual_idx = if idx < 0 {
                    (shape10_out1.len() as i64 + idx) as usize
                } else {
                    idx as usize
                };
                shape10_out1[actual_idx]
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let constant62_out1: [i64; 1] = [1i64];
        let add34_out1 = {
            let mut result = gather10_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant62_out1.iter()) {
                *result_item = result_item.saturating_add(*rhs_item);
            }
            result
        };
        let constant63_out1: [i64; 1] = [2i64];
        let div10_out1 = {
            let mut result = add34_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant63_out1.iter()) {
                *result_item = if *rhs_item != 0 {
                    *result_item / *rhs_item
                } else {
                    *result_item
                };
            }
            result
        };
        let constant64_out1: [i64; 1] = [1i64];
        let mul113_out1 = {
            let mut result = div10_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant64_out1.iter()) {
                *result_item = result_item.saturating_mul(*rhs_item);
            }
            result
        };
        let slice19_out1 = averagepool2d4_out1
            .clone()
            .slice(s![.., 0..mul113_out1[0], .., ..]);
        let constant65_out1: [i64; 1] = [2i64];
        let mul114_out1 = {
            let mut result = div10_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant65_out1.iter()) {
                *result_item = result_item.saturating_mul(*rhs_item);
            }
            result
        };
        let slice20_out1 =
            averagepool2d4_out1.slice(s![.., mul113_out1[0]..mul114_out1[0], .., ..]);
        let conv2d107_out1 = self.conv2d107.forward(slice19_out1);
        let sigmoid95_out1 = burn::tensor::activation::sigmoid(conv2d107_out1.clone());
        let mul115_out1 = conv2d107_out1.mul(sigmoid95_out1);
        let maxpool2d7_out1 = self.maxpool2d7.forward(slice20_out1);
        let conv2d108_out1 = self.conv2d108.forward(maxpool2d7_out1);
        let sigmoid96_out1 = burn::tensor::activation::sigmoid(conv2d108_out1.clone());
        let mul116_out1 = conv2d108_out1.mul(sigmoid96_out1);
        let concat25_out1 =
            burn::tensor::Tensor::cat([mul115_out1, mul116_out1, mul96_out1].into(), 1);
        let conv2d109_out1 = self.conv2d109.forward(concat25_out1);
        let sigmoid97_out1 = burn::tensor::activation::sigmoid(conv2d109_out1.clone());
        let mul117_out1 = conv2d109_out1.mul(sigmoid97_out1);
        let shape11_out1: [i64; 4] = mul117_out1.clone().dims()[0..4]
            .iter()
            .map(|&x| x as i64)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let constant66_out1: [i64; 1] = [1i64];
        let gather11_out1: [i64; 1usize] = constant66_out1
            .iter()
            .map(|&idx| {
                let actual_idx = if idx < 0 {
                    (shape11_out1.len() as i64 + idx) as usize
                } else {
                    idx as usize
                };
                shape11_out1[actual_idx]
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let constant68_out1: [i64; 1] = [1i64];
        let add35_out1 = {
            let mut result = gather11_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant68_out1.iter()) {
                *result_item = result_item.saturating_add(*rhs_item);
            }
            result
        };
        let constant69_out1: [i64; 1] = [2i64];
        let div11_out1 = {
            let mut result = add35_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant69_out1.iter()) {
                *result_item = if *rhs_item != 0 {
                    *result_item / *rhs_item
                } else {
                    *result_item
                };
            }
            result
        };
        let constant70_out1: [i64; 1] = [1i64];
        let mul118_out1 = {
            let mut result = div11_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant70_out1.iter()) {
                *result_item = result_item.saturating_mul(*rhs_item);
            }
            result
        };
        let slice21_out1 = mul117_out1.clone().slice(s![.., 0..mul118_out1[0], .., ..]);
        let constant71_out1: [i64; 1] = [2i64];
        let mul119_out1 = {
            let mut result = div11_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant71_out1.iter()) {
                *result_item = result_item.saturating_mul(*rhs_item);
            }
            result
        };
        let slice22_out1 = mul117_out1.slice(s![.., mul118_out1[0]..mul119_out1[0], .., ..]);
        let conv2d110_out1 = self.conv2d110.forward(slice22_out1.clone());
        let sigmoid98_out1 = burn::tensor::activation::sigmoid(conv2d110_out1.clone());
        let mul120_out1 = conv2d110_out1.mul(sigmoid98_out1);
        let conv2d111_out1 = self.conv2d111.forward(mul120_out1.clone());
        let conv2d112_out1 = self.conv2d112.forward(mul120_out1.clone());
        let add36_out1 = conv2d111_out1.add(conv2d112_out1);
        let sigmoid99_out1 = burn::tensor::activation::sigmoid(add36_out1.clone());
        let mul121_out1 = add36_out1.mul(sigmoid99_out1);
        let conv2d113_out1 = self.conv2d113.forward(mul121_out1);
        let sigmoid100_out1 = burn::tensor::activation::sigmoid(conv2d113_out1.clone());
        let mul122_out1 = conv2d113_out1.mul(sigmoid100_out1);
        let add37_out1 = mul120_out1.add(mul122_out1);
        let conv2d114_out1 = self.conv2d114.forward(slice22_out1.clone());
        let sigmoid101_out1 = burn::tensor::activation::sigmoid(conv2d114_out1.clone());
        let mul123_out1 = conv2d114_out1.mul(sigmoid101_out1);
        let concat26_out1 = burn::tensor::Tensor::cat([add37_out1, mul123_out1].into(), 1);
        let conv2d115_out1 = self.conv2d115.forward(concat26_out1);
        let sigmoid102_out1 = burn::tensor::activation::sigmoid(conv2d115_out1.clone());
        let mul124_out1 = conv2d115_out1.mul(sigmoid102_out1);
        let conv2d116_out1 = self.conv2d116.forward(mul124_out1);
        let sigmoid103_out1 = burn::tensor::activation::sigmoid(conv2d116_out1.clone());
        let mul125_out1 = conv2d116_out1.mul(sigmoid103_out1);
        let conv2d117_out1 = self.conv2d117.forward(mul125_out1.clone());
        let sigmoid104_out1 = burn::tensor::activation::sigmoid(conv2d117_out1.clone());
        let mul126_out1 = conv2d117_out1.mul(sigmoid104_out1);
        let conv2d118_out1 = self.conv2d118.forward(mul126_out1.clone());
        let conv2d119_out1 = self.conv2d119.forward(mul126_out1.clone());
        let add38_out1 = conv2d118_out1.add(conv2d119_out1);
        let sigmoid105_out1 = burn::tensor::activation::sigmoid(add38_out1.clone());
        let mul127_out1 = add38_out1.mul(sigmoid105_out1);
        let conv2d120_out1 = self.conv2d120.forward(mul127_out1);
        let sigmoid106_out1 = burn::tensor::activation::sigmoid(conv2d120_out1.clone());
        let mul128_out1 = conv2d120_out1.mul(sigmoid106_out1);
        let add39_out1 = mul126_out1.add(mul128_out1);
        let conv2d121_out1 = self.conv2d121.forward(mul125_out1.clone());
        let sigmoid107_out1 = burn::tensor::activation::sigmoid(conv2d121_out1.clone());
        let mul129_out1 = conv2d121_out1.mul(sigmoid107_out1);
        let concat27_out1 = burn::tensor::Tensor::cat([add39_out1, mul129_out1].into(), 1);
        let conv2d122_out1 = self.conv2d122.forward(concat27_out1);
        let sigmoid108_out1 = burn::tensor::activation::sigmoid(conv2d122_out1.clone());
        let mul130_out1 = conv2d122_out1.mul(sigmoid108_out1);
        let conv2d123_out1 = self.conv2d123.forward(mul130_out1);
        let sigmoid109_out1 = burn::tensor::activation::sigmoid(conv2d123_out1.clone());
        let mul131_out1 = conv2d123_out1.mul(sigmoid109_out1);
        let concat28_out1 = burn::tensor::Tensor::cat(
            [slice21_out1, slice22_out1, mul125_out1, mul131_out1].into(),
            1,
        );
        let conv2d124_out1 = self.conv2d124.forward(concat28_out1);
        let sigmoid110_out1 = burn::tensor::activation::sigmoid(conv2d124_out1.clone());
        let mul132_out1 = conv2d124_out1.mul(sigmoid110_out1);
        let averagepool2d5_out1 = self.averagepool2d5.forward(mul132_out1.clone());
        let shape12_out1: [i64; 4] = averagepool2d5_out1.clone().dims()[0..4]
            .iter()
            .map(|&x| x as i64)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let constant72_out1: [i64; 1] = [1i64];
        let gather12_out1: [i64; 1usize] = constant72_out1
            .iter()
            .map(|&idx| {
                let actual_idx = if idx < 0 {
                    (shape12_out1.len() as i64 + idx) as usize
                } else {
                    idx as usize
                };
                shape12_out1[actual_idx]
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let constant74_out1: [i64; 1] = [1i64];
        let add40_out1 = {
            let mut result = gather12_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant74_out1.iter()) {
                *result_item = result_item.saturating_add(*rhs_item);
            }
            result
        };
        let constant75_out1: [i64; 1] = [2i64];
        let div12_out1 = {
            let mut result = add40_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant75_out1.iter()) {
                *result_item = if *rhs_item != 0 {
                    *result_item / *rhs_item
                } else {
                    *result_item
                };
            }
            result
        };
        let constant76_out1: [i64; 1] = [1i64];
        let mul133_out1 = {
            let mut result = div12_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant76_out1.iter()) {
                *result_item = result_item.saturating_mul(*rhs_item);
            }
            result
        };
        let slice23_out1 = averagepool2d5_out1
            .clone()
            .slice(s![.., 0..mul133_out1[0], .., ..]);
        let constant77_out1: [i64; 1] = [2i64];
        let mul134_out1 = {
            let mut result = div12_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant77_out1.iter()) {
                *result_item = result_item.saturating_mul(*rhs_item);
            }
            result
        };
        let slice24_out1 =
            averagepool2d5_out1.slice(s![.., mul133_out1[0]..mul134_out1[0], .., ..]);
        let conv2d125_out1 = self.conv2d125.forward(slice23_out1);
        let sigmoid111_out1 = burn::tensor::activation::sigmoid(conv2d125_out1.clone());
        let mul135_out1 = conv2d125_out1.mul(sigmoid111_out1);
        let maxpool2d8_out1 = self.maxpool2d8.forward(slice24_out1);
        let conv2d126_out1 = self.conv2d126.forward(maxpool2d8_out1);
        let sigmoid112_out1 = burn::tensor::activation::sigmoid(conv2d126_out1.clone());
        let mul136_out1 = conv2d126_out1.mul(sigmoid112_out1);
        let concat29_out1 =
            burn::tensor::Tensor::cat([mul135_out1, mul136_out1, mul80_out1].into(), 1);
        let conv2d127_out1 = self.conv2d127.forward(concat29_out1);
        let sigmoid113_out1 = burn::tensor::activation::sigmoid(conv2d127_out1.clone());
        let mul137_out1 = conv2d127_out1.mul(sigmoid113_out1);
        let shape13_out1: [i64; 4] = mul137_out1.clone().dims()[0..4]
            .iter()
            .map(|&x| x as i64)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let constant78_out1: [i64; 1] = [1i64];
        let gather13_out1: [i64; 1usize] = constant78_out1
            .iter()
            .map(|&idx| {
                let actual_idx = if idx < 0 {
                    (shape13_out1.len() as i64 + idx) as usize
                } else {
                    idx as usize
                };
                shape13_out1[actual_idx]
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let constant80_out1: [i64; 1] = [1i64];
        let add41_out1 = {
            let mut result = gather13_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant80_out1.iter()) {
                *result_item = result_item.saturating_add(*rhs_item);
            }
            result
        };
        let constant81_out1: [i64; 1] = [2i64];
        let div13_out1 = {
            let mut result = add41_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant81_out1.iter()) {
                *result_item = if *rhs_item != 0 {
                    *result_item / *rhs_item
                } else {
                    *result_item
                };
            }
            result
        };
        let constant82_out1: [i64; 1] = [1i64];
        let mul138_out1 = {
            let mut result = div13_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant82_out1.iter()) {
                *result_item = result_item.saturating_mul(*rhs_item);
            }
            result
        };
        let slice25_out1 = mul137_out1.clone().slice(s![.., 0..mul138_out1[0], .., ..]);
        let constant83_out1: [i64; 1] = [2i64];
        let mul139_out1 = {
            let mut result = div13_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant83_out1.iter()) {
                *result_item = result_item.saturating_mul(*rhs_item);
            }
            result
        };
        let slice26_out1 = mul137_out1.slice(s![.., mul138_out1[0]..mul139_out1[0], .., ..]);
        let conv2d128_out1 = self.conv2d128.forward(slice26_out1.clone());
        let sigmoid114_out1 = burn::tensor::activation::sigmoid(conv2d128_out1.clone());
        let mul140_out1 = conv2d128_out1.mul(sigmoid114_out1);
        let conv2d129_out1 = self.conv2d129.forward(mul140_out1.clone());
        let conv2d130_out1 = self.conv2d130.forward(mul140_out1.clone());
        let add42_out1 = conv2d129_out1.add(conv2d130_out1);
        let sigmoid115_out1 = burn::tensor::activation::sigmoid(add42_out1.clone());
        let mul141_out1 = add42_out1.mul(sigmoid115_out1);
        let conv2d131_out1 = self.conv2d131.forward(mul141_out1);
        let sigmoid116_out1 = burn::tensor::activation::sigmoid(conv2d131_out1.clone());
        let mul142_out1 = conv2d131_out1.mul(sigmoid116_out1);
        let add43_out1 = mul140_out1.add(mul142_out1);
        let conv2d132_out1 = self.conv2d132.forward(slice26_out1.clone());
        let sigmoid117_out1 = burn::tensor::activation::sigmoid(conv2d132_out1.clone());
        let mul143_out1 = conv2d132_out1.mul(sigmoid117_out1);
        let concat30_out1 = burn::tensor::Tensor::cat([add43_out1, mul143_out1].into(), 1);
        let conv2d133_out1 = self.conv2d133.forward(concat30_out1);
        let sigmoid118_out1 = burn::tensor::activation::sigmoid(conv2d133_out1.clone());
        let mul144_out1 = conv2d133_out1.mul(sigmoid118_out1);
        let conv2d134_out1 = self.conv2d134.forward(mul144_out1);
        let sigmoid119_out1 = burn::tensor::activation::sigmoid(conv2d134_out1.clone());
        let mul145_out1 = conv2d134_out1.mul(sigmoid119_out1);
        let conv2d135_out1 = self.conv2d135.forward(mul145_out1.clone());
        let sigmoid120_out1 = burn::tensor::activation::sigmoid(conv2d135_out1.clone());
        let mul146_out1 = conv2d135_out1.mul(sigmoid120_out1);
        let conv2d136_out1 = self.conv2d136.forward(mul146_out1.clone());
        let conv2d137_out1 = self.conv2d137.forward(mul146_out1.clone());
        let add44_out1 = conv2d136_out1.add(conv2d137_out1);
        let sigmoid121_out1 = burn::tensor::activation::sigmoid(add44_out1.clone());
        let mul147_out1 = add44_out1.mul(sigmoid121_out1);
        let conv2d138_out1 = self.conv2d138.forward(mul147_out1);
        let sigmoid122_out1 = burn::tensor::activation::sigmoid(conv2d138_out1.clone());
        let mul148_out1 = conv2d138_out1.mul(sigmoid122_out1);
        let add45_out1 = mul146_out1.add(mul148_out1);
        let conv2d139_out1 = self.conv2d139.forward(mul145_out1.clone());
        let sigmoid123_out1 = burn::tensor::activation::sigmoid(conv2d139_out1.clone());
        let mul149_out1 = conv2d139_out1.mul(sigmoid123_out1);
        let concat31_out1 = burn::tensor::Tensor::cat([add45_out1, mul149_out1].into(), 1);
        let conv2d140_out1 = self.conv2d140.forward(concat31_out1);
        let sigmoid124_out1 = burn::tensor::activation::sigmoid(conv2d140_out1.clone());
        let mul150_out1 = conv2d140_out1.mul(sigmoid124_out1);
        let conv2d141_out1 = self.conv2d141.forward(mul150_out1);
        let sigmoid125_out1 = burn::tensor::activation::sigmoid(conv2d141_out1.clone());
        let mul151_out1 = conv2d141_out1.mul(sigmoid125_out1);
        let concat32_out1 = burn::tensor::Tensor::cat(
            [slice25_out1, slice26_out1, mul145_out1, mul151_out1].into(),
            1,
        );
        let conv2d142_out1 = self.conv2d142.forward(concat32_out1);
        let sigmoid126_out1 = burn::tensor::activation::sigmoid(conv2d142_out1.clone());
        let mul152_out1 = conv2d142_out1.mul(sigmoid126_out1);
        let conv2d143_out1 = self.conv2d143.forward(mul112_out1.clone());
        let sigmoid127_out1 = burn::tensor::activation::sigmoid(conv2d143_out1.clone());
        let mul153_out1 = conv2d143_out1.mul(sigmoid127_out1);
        let conv2d144_out1 = self.conv2d144.forward(mul153_out1);
        let sigmoid128_out1 = burn::tensor::activation::sigmoid(conv2d144_out1.clone());
        let mul154_out1 = conv2d144_out1.mul(sigmoid128_out1);
        let conv2d145_out1 = self.conv2d145.forward(mul154_out1);
        let conv2d146_out1 = self.conv2d146.forward(mul112_out1);
        let sigmoid129_out1 = burn::tensor::activation::sigmoid(conv2d146_out1.clone());
        let mul155_out1 = conv2d146_out1.mul(sigmoid129_out1);
        let conv2d147_out1 = self.conv2d147.forward(mul155_out1);
        let sigmoid130_out1 = burn::tensor::activation::sigmoid(conv2d147_out1.clone());
        let mul156_out1 = conv2d147_out1.mul(sigmoid130_out1);
        let conv2d148_out1 = self.conv2d148.forward(mul156_out1);
        let constant84_out1: i64 = 0i64;
        let reshape1_out1 = conv2d145_out1.reshape([1, 4, 16, 80, 80]);
        let transpose1_out1 = reshape1_out1.permute([0, 2, 1, 3, 4]);
        let softmax1_out1 = burn::tensor::activation::softmax(transpose1_out1, 1);
        let conv3d1_out1 = self.conv3d1.forward(softmax1_out1);
        let sliced = conv3d1_out1.slice(s![
            ..,
            (constant84_out1 as usize)..((constant84_out1 as usize) + 1),
            ..,
            ..,
            ..
        ]);
        let gather14_out1 = sliced.squeeze_dim::<4usize>(1);
        let conv2d149_out1 = self.conv2d149.forward(mul132_out1.clone());
        let sigmoid131_out1 = burn::tensor::activation::sigmoid(conv2d149_out1.clone());
        let mul157_out1 = conv2d149_out1.mul(sigmoid131_out1);
        let conv2d150_out1 = self.conv2d150.forward(mul157_out1);
        let sigmoid132_out1 = burn::tensor::activation::sigmoid(conv2d150_out1.clone());
        let mul158_out1 = conv2d150_out1.mul(sigmoid132_out1);
        let conv2d151_out1 = self.conv2d151.forward(mul158_out1);
        let conv2d152_out1 = self.conv2d152.forward(mul132_out1);
        let sigmoid133_out1 = burn::tensor::activation::sigmoid(conv2d152_out1.clone());
        let mul159_out1 = conv2d152_out1.mul(sigmoid133_out1);
        let conv2d153_out1 = self.conv2d153.forward(mul159_out1);
        let sigmoid134_out1 = burn::tensor::activation::sigmoid(conv2d153_out1.clone());
        let mul160_out1 = conv2d153_out1.mul(sigmoid134_out1);
        let conv2d154_out1 = self.conv2d154.forward(mul160_out1);
        let reshape2_out1 = conv2d151_out1.reshape([1, 4, 16, 40, 40]);
        let transpose2_out1 = reshape2_out1.permute([0, 2, 1, 3, 4]);
        let softmax2_out1 = burn::tensor::activation::softmax(transpose2_out1, 1);
        let conv3d2_out1 = self.conv3d2.forward(softmax2_out1);
        let sliced = conv3d2_out1.slice(s![
            ..,
            (constant84_out1 as usize)..((constant84_out1 as usize) + 1),
            ..,
            ..,
            ..
        ]);
        let gather15_out1 = sliced.squeeze_dim::<4usize>(1);
        let conv2d155_out1 = self.conv2d155.forward(mul152_out1.clone());
        let sigmoid135_out1 = burn::tensor::activation::sigmoid(conv2d155_out1.clone());
        let mul161_out1 = conv2d155_out1.mul(sigmoid135_out1);
        let conv2d156_out1 = self.conv2d156.forward(mul161_out1);
        let sigmoid136_out1 = burn::tensor::activation::sigmoid(conv2d156_out1.clone());
        let mul162_out1 = conv2d156_out1.mul(sigmoid136_out1);
        let conv2d157_out1 = self.conv2d157.forward(mul162_out1);
        let conv2d158_out1 = self.conv2d158.forward(mul152_out1);
        let sigmoid137_out1 = burn::tensor::activation::sigmoid(conv2d158_out1.clone());
        let mul163_out1 = conv2d158_out1.mul(sigmoid137_out1);
        let conv2d159_out1 = self.conv2d159.forward(mul163_out1);
        let sigmoid138_out1 = burn::tensor::activation::sigmoid(conv2d159_out1.clone());
        let mul164_out1 = conv2d159_out1.mul(sigmoid138_out1);
        let conv2d160_out1 = self.conv2d160.forward(mul164_out1);
        let reshape3_out1 = conv2d157_out1.reshape([1, 4, 16, 20, 20]);
        let transpose3_out1 = reshape3_out1.permute([0, 2, 1, 3, 4]);
        let softmax3_out1 = burn::tensor::activation::softmax(transpose3_out1, 1);
        let conv3d3_out1 = self.conv3d3.forward(softmax3_out1);
        let sliced = conv3d3_out1.slice(s![
            ..,
            (constant84_out1 as usize)..((constant84_out1 as usize) + 1),
            ..,
            ..,
            ..
        ]);
        let gather16_out1 = sliced.squeeze_dim::<4usize>(1);
        let transpose4_out1 = conv2d148_out1.permute([0, 2, 3, 1]);
        let reshape4_out1 = transpose4_out1.reshape([1, 6400, 1]);
        let transpose5_out1 = gather14_out1.permute([0, 2, 3, 1]);
        let reshape5_out1 = transpose5_out1.reshape([1, 6400, 4]);
        let transpose6_out1 = conv2d154_out1.permute([0, 2, 3, 1]);
        let reshape6_out1 = transpose6_out1.reshape([1, 1600, 1]);
        let transpose7_out1 = gather15_out1.permute([0, 2, 3, 1]);
        let reshape7_out1 = transpose7_out1.reshape([1, 1600, 4]);
        let transpose8_out1 = conv2d160_out1.permute([0, 2, 3, 1]);
        let reshape8_out1 = transpose8_out1.reshape([1, 400, 1]);
        let transpose9_out1 = gather16_out1.permute([0, 2, 3, 1]);
        let reshape9_out1 = transpose9_out1.reshape([1, 400, 4]);
        let concat33_out1 =
            burn::tensor::Tensor::cat([reshape4_out1, reshape6_out1, reshape8_out1].into(), 1);
        let concat34_out1 =
            burn::tensor::Tensor::cat([reshape5_out1, reshape7_out1, reshape9_out1].into(), 1);
        let constant94_out1 = self.constant94.val();
        let mul165_out1 = concat34_out1.mul(constant94_out1);
        let shape14_out1: [i64; 3] = mul165_out1.clone().dims()[0..3]
            .iter()
            .map(|&x| x as i64)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let constant95_out1: [i64; 1] = [-1i64];
        let gather17_out1: [i64; 1usize] = constant95_out1
            .iter()
            .map(|&idx| {
                let actual_idx = if idx < 0 {
                    (shape14_out1.len() as i64 + idx) as usize
                } else {
                    idx as usize
                };
                shape14_out1[actual_idx]
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let constant97_out1: [i64; 1] = [1i64];
        let add46_out1 = {
            let mut result = gather17_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant97_out1.iter()) {
                *result_item = result_item.saturating_add(*rhs_item);
            }
            result
        };
        let constant98_out1: [i64; 1] = [2i64];
        let div14_out1 = {
            let mut result = add46_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant98_out1.iter()) {
                *result_item = if *rhs_item != 0 {
                    *result_item / *rhs_item
                } else {
                    *result_item
                };
            }
            result
        };
        let constant99_out1: [i64; 1] = [1i64];
        let mul166_out1 = {
            let mut result = div14_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant99_out1.iter()) {
                *result_item = result_item.saturating_mul(*rhs_item);
            }
            result
        };
        let slice27_out1 = mul165_out1.clone().slice(s![.., .., 0..mul166_out1[0]]);
        let constant100_out1: [i64; 1] = [2i64];
        let mul167_out1 = {
            let mut result = div14_out1;
            for (result_item, rhs_item) in result.iter_mut().zip(constant100_out1.iter()) {
                *result_item = result_item.saturating_mul(*rhs_item);
            }
            result
        };
        let slice28_out1 = mul165_out1.slice(s![.., .., mul166_out1[0]..mul167_out1[0]]);
        let constant101_out1 = self.constant101.val();
        let sub1_out1 = constant101_out1.unsqueeze_dims(&[0isize]).sub(slice27_out1);
        let add47_out1 = constant1_out1.unsqueeze_dims(&[0isize]).add(slice28_out1);
        let concat35_out1 = burn::tensor::Tensor::cat([sub1_out1, add47_out1].into(), 2);
        (concat33_out1, concat35_out1)
    }
}
