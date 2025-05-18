import decord
import math
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

def video_subsample(filepath: str):
    vr = decord.VideoReader(filepath, ctx=decord.cpu(0))
    num_frames = len(vr)

    if num_frames < 8:
        return [i for i in range(num_frames)]

    fps = vr.get_avg_fps()
    duration = num_frames / fps
    subsampled_idx = []

    # if longer than 4 seconds - only sample from first 4 seconds 
    if duration > 4:
        num_frames = math.floor(4 * fps)

    stride = max(1, num_frames // 8)
    for i in range(0, num_frames, stride):
        subsampled_idx.append(i)

    return subsampled_idx


        
