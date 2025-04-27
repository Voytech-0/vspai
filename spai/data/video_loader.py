import decord
import torch
from PIL import Image


def get_frame(filepath: str, index: int):
    try:
        with decord.VideoReader(filepath, ctx=decord.cpu(0)) as vr:
            image = vr[index].asnumpy()
            image = Image.fromarray(image)
    except Exception as e:
        print("Image not read properly: ", filepath)
        image = Image.new("RGB", (128, 128), (0, 0, 0))

    return image


def get_num_frames(filepath: str):
    with decord.VideoReader(filepath, ctx=decord.cpu(0)) as vr:
        len_vr = len(vr)

    return len_vr
