use std::collections::HashSet;

use burn::prelude::*;
use fast_image_resize::images::{Image, ImageRef};
use fast_image_resize::{PixelType, Resizer};
use image::{GenericImage, GenericImageView, ImageBuffer, Rgb};

use super::image::Rgb32FImage;
use super::yolov9m::YOLOv9m;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/yolov9/model.bpk");
const INPUT_H: usize = 640;
const INPUT_W: usize = 640;
const CONF_THRESH: f32 = 0.25;
const IOU_THRESH: f32 = 0.5;
const PADVAL: f32 = 117. / 255.;

/// (x, y, w, h, class_id)
pub type BBox = (u32, u32, u32, u32, i32);

fn logit(p: f32) -> f32 {
    (p / (1.0 - p)).ln()
}

struct SizeParams {
    scale: f32,
    new_w: u32,
    new_h: u32,
    x_offset: u32,
    y_offset: u32,
    canvas: ImageBuffer<Rgb<f32>, Vec<f32>>,
    resizer: Resizer,
    input_buffer: Vec<f32>,
}

pub struct InferenceEngine {
    model: YOLOv9m,
    device: Device,
    conf_thresh_logit: f32,
    size_params: Option<SizeParams>,
}

impl InferenceEngine {
    pub fn new(device: &Device) -> Self {
        assert!(
            std::path::Path::new(MODEL_PATH).exists(),
            "model not found: {MODEL_PATH}"
        );
        let model = YOLOv9m::from_file(MODEL_PATH, device);
        Self {
            model,
            device: device.clone(),
            conf_thresh_logit: logit(CONF_THRESH),
            size_params: None,
        }
    }

    fn set_size_params(&mut self, img_width: usize, img_height: usize) {
        let scale_x = INPUT_W as f32 / img_width as f32;
        let scale_y = INPUT_H as f32 / img_height as f32;
        let scale = scale_x.min(scale_y);
        let new_w = (img_width as f32 * scale).round() as u32;
        let new_h = (img_height as f32 * scale).round() as u32;

        self.size_params = Some(SizeParams {
            scale,
            new_w,
            new_h,
            x_offset: (INPUT_W as u32 - new_w) / 2,
            y_offset: (INPUT_H as u32 - new_h) / 2,
            canvas: ImageBuffer::from_pixel(INPUT_W as u32, INPUT_H as u32, Rgb([PADVAL; 3])),
            resizer: Resizer::new(),
            input_buffer: vec![0.0f32; INPUT_W * INPUT_H * 3],
        });
    }

    pub fn infer(&mut self, img: &Rgb32FImage) -> Vec<BBox> {
        if self.size_params.is_none() {
            self.set_size_params(img.width() as usize, img.height() as usize);
        }

        let sp = self.size_params.as_mut().unwrap();

        for p in sp.canvas.pixels_mut() {
            *p = Rgb([PADVAL, PADVAL, PADVAL]);
        }
        let tile = img.view(0, 0, img.width(), img.height()).to_image();
        if tile.width() == sp.new_w && tile.height() == sp.new_h {
            sp.canvas
                .copy_from(&tile, sp.x_offset, sp.y_offset)
                .unwrap();
        } else {
            let tile_view = ImageRef::new(
                tile.width(),
                tile.height(),
                bytemuck::cast_slice(tile.as_raw()),
                PixelType::F32x3,
            )
            .unwrap();
            let mut resized_tile = Image::new(sp.new_w, sp.new_h, PixelType::F32x3);
            sp.resizer
                .resize(&tile_view, &mut resized_tile, None)
                .unwrap();
            let src = bytemuck::cast_slice::<u8, f32>(resized_tile.buffer());
            sp.input_buffer[..src.len()].copy_from_slice(src);
            let resized_tile_buffer: ImageBuffer<Rgb<f32>, Vec<f32>> =
                ImageBuffer::from_raw(sp.new_w, sp.new_h, sp.input_buffer.clone())
                    .expect("Failed to convert resized tile");
            sp.canvas
                .copy_from(&resized_tile_buffer, sp.x_offset, sp.y_offset)
                .unwrap();
        }

        let input = Tensor::<1>::from_data(sp.canvas.as_raw().as_slice(), &self.device)
            .reshape([1, INPUT_H as i32, INPUT_W as i32, 3])
            .permute([0, 3, 1, 2]);

        let (class_confs, bboxes) = self.model.forward(input);

        let class_ids = class_confs.clone().argmax(2);
        let confs = class_confs.max_dim(2);

        assert_eq!(class_ids.dims()[0], 1, "Batch size > 1 not supported");
        let bboxes = bboxes.slice(s![0]).squeeze_dim::<2>(0);
        let confs = confs.slice(s![0]).reshape([-1]);
        let class_ids = class_ids.slice(s![0]).reshape([-1]);

        let conf_mask = confs.clone().greater_elem(self.conf_thresh_logit);
        let indices = conf_mask.argwhere().reshape([-1]);
        if indices.dims()[0] == 0 {
            return Vec::new();
        }

        let bboxes = bboxes.select(0, indices.clone());
        let class_ids = class_ids.select(0, indices.clone());
        let confs = confs.select(0, indices);

        let unique_classes: HashSet<i32> = class_ids
            .clone()
            .try_to_vec_as::<i32>()
            .unwrap()
            .into_iter()
            .collect();

        let mut all = Vec::new();
        let x_offset_f32 = sp.x_offset as f32;
        let y_offset_f32 = sp.y_offset as f32;
        let scale = sp.scale;

        for class_id in unique_classes {
            let class_mask = class_ids.clone().equal_elem(class_id);
            let indices = class_mask.argwhere().reshape([-1]);
            let class_bboxes = bboxes.clone().select(0, indices.clone());
            let class_confs = confs.clone().select(0, indices);

            let (class_confs, indices) = class_confs.sort_descending_with_indices(0);
            let class_bboxes = class_bboxes.select(0, indices);

            let bboxes_data = class_bboxes.try_into_vec_as::<f32>().unwrap();
            let confs_data = class_confs.try_into_vec_as::<f32>().unwrap();

            let mut keep = Vec::new();
            let mut suppressed = vec![false; confs_data.len()];

            for i in 0..confs_data.len() {
                if suppressed[i] {
                    continue;
                }
                keep.push(i);
                let box_i = &bboxes_data[i * 4..(i + 1) * 4];
                let area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1]);

                for j in (i + 1)..confs_data.len() {
                    if suppressed[j] {
                        continue;
                    }
                    let box_j = &bboxes_data[j * 4..(j + 1) * 4];
                    let xx1 = box_i[0].max(box_j[0]);
                    let yy1 = box_i[1].max(box_j[1]);
                    let xx2 = box_i[2].min(box_j[2]);
                    let yy2 = box_i[3].min(box_j[3]);
                    let w = (xx2 - xx1).max(0.0);
                    let h = (yy2 - yy1).max(0.0);
                    let inter = w * h;
                    let area_j = (box_j[2] - box_j[0]) * (box_j[3] - box_j[1]);
                    let ovr = inter / (area_i + area_j - inter);
                    if ovr > IOU_THRESH {
                        suppressed[j] = true;
                    }
                }
            }

            for &idx in &keep {
                let bbox = &bboxes_data[idx * 4..(idx + 1) * 4];
                let x1 = ((bbox[0] - x_offset_f32) / scale).round().max(0.0) as u32;
                let y1 = ((bbox[1] - y_offset_f32) / scale).round().max(0.0) as u32;
                let x2 = ((bbox[2] - x_offset_f32) / scale).round().max(0.0) as u32;
                let y2 = ((bbox[3] - y_offset_f32) / scale).round().max(0.0) as u32;
                if x2 <= x1 || y2 <= y1 {
                    continue;
                }
                all.push((x1, y1, x2 - x1, y2 - y1, class_id));
            }
        }

        all
    }
}
