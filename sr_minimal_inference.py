import cv2
import glob
import os
import torch
import numpy as np
import torch.nn.functional as F
import argparse
import tqdm
import requests


class RealESRGANer:
    def __init__(self, jit_model_path, scale=2, half=False, pre_pad=0, device="cuda"):
        self.model = torch.jit.load(jit_model_path)
        self.model.eval()
        self.model.to(device)
        self.scale = scale
        self.half = half
        self.pre_pad = pre_pad
        self.device = device

    def pre_process(self, img):
        """Pre-process, such as pre-pad and mod pad, so that the images can be divisible"""
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        self.img = img.unsqueeze(0).to(self.device)
        if self.half:
            self.img = self.img.half()

        # pre_pad
        if self.pre_pad != 0:
            self.img = F.pad(self.img, (0, self.pre_pad, 0, self.pre_pad), "reflect")
        # mod pad for divisible borders
        self.mod_scale = None
        if self.scale == 2:
            self.mod_scale = 2
        elif self.scale == 1:
            self.mod_scale = 4
        if self.mod_scale is not None:
            self.mod_pad_h, self.mod_pad_w = 0, 0
            _, _, h, w = self.img.size()
            if h % self.mod_scale != 0:
                self.mod_pad_h = self.mod_scale - h % self.mod_scale
            if w % self.mod_scale != 0:
                self.mod_pad_w = self.mod_scale - w % self.mod_scale
            self.img = F.pad(
                self.img, (0, self.mod_pad_w, 0, self.mod_pad_h), "reflect"
            )

    def process(self):
        # model inference
        with torch.no_grad():
            self.output = self.model(self.img)

    def post_process(self):
        # remove extra pad
        if self.mod_scale is not None:
            _, _, h, w = self.output.size()
            self.output = self.output[
                :,
                :,
                0 : h - self.mod_pad_h * self.scale,
                0 : w - self.mod_pad_w * self.scale,
            ]
        # remove prepad
        if self.pre_pad != 0:
            _, _, h, w = self.output.size()
            self.output = self.output[
                :,
                :,
                0 : h - self.pre_pad * self.scale,
                0 : w - self.pre_pad * self.scale,
            ]
        return self.output.data.squeeze().float().cpu().clamp_(0, 1).numpy()

    def enhance(self, img):
        self.pre_process(img)
        self.process()
        return self.post_process()


def get_model_from_url(url, save_path="model.jit.tar"):
    print(f"Downloading model from {url} to {save_path}")
    response = requests.get(url)
    with open(save_path, "wb") as f:
        f.write(response.content)
    return save_path


def test_realesrgan():
    jit_model_path = "/root/workspace/git_proj/sr/RealESRGAN_x2-x2.0.scripts_jit.tar"
    upsampler = RealESRGANer(
        jit_model_path, scale=2, half=True, pre_pad=0, device="cuda"
    )
    img = cv2.imread("/workspace/git_proj/sr/Real-ESRGAN/inputs/0014.jpg") / 255.0
    print(f"img shape: {img.shape}", img.min(), img.max())
    output = upsampler.enhance(img)
    print(f"output shape: {output.shape}", output.min(), output.max())
    output = np.transpose(output, (1, 2, 0))
    output = (output * 255.0).astype(np.uint8)
    cv2.imwrite("output.png", output)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--half", type=bool, default=True)
    parser.add_argument("--model_name", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if os.path.isdir(args.input):
        paths = sorted(
            glob.glob(os.path.join(args.input, "*.png"))
            + glob.glob(os.path.join(args.input, "*.jpg"))
        )
        os.makedirs(args.output, exist_ok=True)
    else:
        paths = [args.input]
        os.makedirs(args.output, exist_ok=True)

    assert args.model_name != ""
    model_info = {
        "RealESRGAN_x2": {
            "url": "https://github.com/Holasyb918/REAL_ESRGAN-minimal-inference/releases/download/v1.0.0/RealESRGAN_x2.scripts_jit.tar",
            "scale": 2,
        },
        "RealESRGAN_x4plus_anime_6B": {
            "url": "https://github.com/Holasyb918/REAL_ESRGAN-minimal-inference/releases/download/v1.0.0/RealESRGAN_x4plus_anime_6B-x4.0.scripts_jit.tar",
            "scale": 4,
        },
    }
    model_path = os.path.basename(model_info[args.model_name]["url"])
    if not os.path.exists(model_path):
        get_model_from_url(model_info[args.model_name]["url"], model_path)
    model = RealESRGANer(
        model_path,
        scale=model_info[args.model_name]["scale"],
        half=True,
        pre_pad=0,
        device="cuda",
    )
    for path in tqdm.tqdm(paths):
        basename = os.path.basename(path)
        img = cv2.imread(path) / 255.0
        print("input shape: ", img.shape)
        h, w = img.shape[:2]
        output = model.enhance(img)
        output = np.transpose(output, (1, 2, 0))
        output = (output * 255.0).astype(np.uint8)
        target_h, target_w = h * args.scale, w * args.scale
        output = cv2.resize(output, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        print("output shape: ", output.shape)
        cv2.imwrite(os.path.join(args.output, basename), output)
        print("save to: ", os.path.join(args.output, basename))
