from PIL import Image

IMAGE_FORMAT = ".png"
DOWNSAMPLE_MODE = Image.BICUBIC
COLOR_CHANNELS = 3

HR_IMG_SIZE = (160, 160) #size is selected beased on the smallest image in the dataset
UPSCALING_FACTOR = 2
LR_IMG_SIZE = (HR_IMG_SIZE[0] // UPSCALING_FACTOR, HR_IMG_SIZE[1] // UPSCALING_FACTOR)
