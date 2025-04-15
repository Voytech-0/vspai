import decord
import torch
from PIL import Image



def get_frame(filepath: str):
    vr = decord.VideoReader(filepath, ctx=decord.cpu(0))
    # tensor = torch.as_tensor(vr[0].asnumpy())
    image = vr[0].asnumpy()
    image = Image.fromarray(image)
    return image
