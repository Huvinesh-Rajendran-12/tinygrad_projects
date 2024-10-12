from PIL import Image
from tinygrad import Tensor, nn, TinyJit
from tinygrad.engine.graph import GlobalCounters
from tinygrad.helpers import trange
import numpy as np
import os
import random
# Assuming these constants are defined in your utils.constants
from constants import (
    IMAGE_FORMAT,
    DOWNSAMPLE_MODE,
    COLOR_CHANNELS,
    UPSCALING_FACTOR,
    HR_IMG_SIZE,
    LR_IMG_SIZE,
)

def mse_loss(pred: Tensor, target: Tensor):
    return ((pred - target) ** 2).mean()


class DIV2K_Dataset:
    def __init__(self, hr_image_folder: str, batch_size: int, set_type: str):
        self.batch_size = batch_size
        self.hr_image_folder = hr_image_folder
        self.image_fns = np.sort([
            x for x in os.listdir(hr_image_folder) if x.endswith(IMAGE_FORMAT)
        ])

        if set_type == "train":
            self.image_fns = self.image_fns[:-200]
        elif set_type == "val":
            self.image_fns = self.image_fns[-200:-100]
        else:
            self.image_fns = self.image_fns[-100:]

        self.set_type = set_type

    def __len__(self):
        return len(self.image_fns) // self.batch_size

    def on_epoch_end(self):
        self.image_fns = np.random.permutation(self.image_fns)

    def __iter__(self):
        return self

    def __next__(self):
        batch_image_fns = random.sample(list(self.image_fns), self.batch_size)
        batch_hr_images = []
        batch_lr_images = []

        for image_fn in batch_image_fns:
            hr_image_pil = Image.open(os.path.join(self.hr_image_folder, image_fn))
            hr_image = np.array(hr_image_pil)

            # Simple random crop
            if self.set_type in ["train", "val"]:
                x = random.randint(0, hr_image.shape[1] - HR_IMG_SIZE[0])
                y = random.randint(0, hr_image.shape[0] - HR_IMG_SIZE[1])
                hr_image = hr_image[y:y+HR_IMG_SIZE[1], x:x+HR_IMG_SIZE[0]]
            else:
                hr_image = hr_image[:HR_IMG_SIZE[1], :HR_IMG_SIZE[0]]

            hr_image_pil = Image.fromarray(hr_image)
            lr_image_pil = hr_image_pil.resize(LR_IMG_SIZE, resample=DOWNSAMPLE_MODE)

            hr_image = np.array(hr_image_pil).astype(np.float32) / 255.0
            lr_image = np.array(lr_image_pil).astype(np.float32) / 255.0

            batch_hr_images.append(hr_image)
            batch_lr_images.append(lr_image)

        batch_hr_images = np.array(batch_hr_images).transpose(0, 3, 1, 2)  # NHWC to NCHW
        batch_lr_images = np.array(batch_lr_images).transpose(0, 3, 1, 2)  # NHWC to NCHW

        return Tensor(batch_lr_images), Tensor(batch_hr_images)

class FSRCNN:
    def __init__(self, scale_f: int, d: int = 56, s: int = 12, m: int = 4) -> None:
        self.feature_extraction = nn.Conv2d(
            in_channels=3, out_channels=d, kernel_size=5, padding=2
        )
        self.shrinking = nn.Conv2d(in_channels=d, out_channels=s, kernel_size=1)
        self.mapping = [
            nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, padding=1)
            for _ in range(m)
        ]
        self.expanding = nn.Conv2d(in_channels=s, out_channels=d, kernel_size=1)
        self.deconvolution = nn.ConvTranspose2d(
            in_channels=d,
            out_channels=3,
            kernel_size=9,
            stride=scale_f,
            padding=4,
            output_padding=1,
        )

    def __call__(self, x) -> Tensor:
        x = self.feature_extraction(x).relu()
        x = self.shrinking(x).relu()
        x = x.sequential(self.mapping).relu()
        x = self.expanding(x).relu()
        x = self.deconvolution(x)
        return x

if __name__ == "__main__":

    n_epochs = 100
    train_dataset = DIV2K_Dataset(hr_image_folder="./data/DIV2K_HR/", batch_size=32, set_type="train")
    val_dataset = DIV2K_Dataset(hr_image_folder="./data/DIV2K_HR/", batch_size=32, set_type="val")
    model = FSRCNN(scale_f=2)
    optimizer = nn.optim.AdamW(nn.state.get_parameters(model))

    @TinyJit
    def train_step(x: Tensor, y: Tensor) -> Tensor:
        optimizer.zero_grad()
        pred = model(x)
        loss = mse_loss(pred, y).backward()
        optimizer.step()
        return loss

    for i in (t := trange(n_epochs)):
        GlobalCounters.reset()
        total_loss = 0
        batches = 0

        for batch_lr, batch_hr in train_dataset:
            with Tensor.train():
                loss = train_step(batch_lr, batch_hr)
            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / batches
        t.set_description(f"epoch: {i+1}/{n_epochs}, average mse loss: {avg_loss:6.2f}")
