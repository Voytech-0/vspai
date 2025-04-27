import decord
import torch
from PIL import Image


def get_frame(filepath: str, index: int):
    try:
        vr = decord.VideoReader(filepath, ctx=decord.cpu(0))
        image = vr[index].asnumpy()
        image = Image.fromarray(image)
        del vr
    except Exception as e:
        print("Image not read properly: ", filepath)
        image = Image.new("RGB", (128, 128), (0, 0, 0))

    return image


def get_num_frames(filepath: str):
    vr = decord.VideoReader(filepath, ctx=decord.cpu(0))
    len_vr = len(vr)
    del vr

    return len_vr
