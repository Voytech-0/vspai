import decord
import torch

def get_frame(filepath: str):

    vr = decord.VideoReader(filepath, ctx = decord.cpu(0))

    return torch.as_tensor(vr[0])

