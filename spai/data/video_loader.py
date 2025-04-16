import decord
import torch
from PIL import Image



def get_frame(filepath: str, index: int):
    vr = decord.VideoReader(filepath, ctx=decord.cpu(0))
    # tensor = torch.as_tensor(vr[0].asnumpy())
    image = vr[index].asnumpy()
    image = Image.fromarray(image)
    return image

def get_num_frames(filepath: str):
    vr = decord.VideoReader(filepath, ctx=decord.cpu(0))

    return len(vr)
    
