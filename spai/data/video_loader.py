import decord
import torch
from PIL import Image



def get_frame(filepath: str, index: int):
    # try catch
    try:
        vr = decord.VideoReader(filepath, ctx=decord.cpu(0))
        image = vr[index].asnumpy()
        image = Image.fromarray(image)
    except Exception as e:
        print("Image not read properly: ", filepath)
        image = Image.new("RGB", (128, 128), (0, 0, 0))

    # tensor = torch.as_tensor(vr[0].asnumpy())
    return image

def get_num_frames(filepath: str):
    vr = decord.VideoReader(filepath, ctx=decord.cpu(0))

    return len(vr)
    
