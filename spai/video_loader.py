import decord
import torch

def get_frame(filepath: str):

    vr = decord.VideoReader(filepath, ctx = decord.cpu(0))

    tensor = torch.as_tensor(vr[0].asnumpy())

    tensor = tensor.permute(2, 0, 1)

    tensor = tensor.unsqueeze(0)

    tensor = tensor.float()

    return tensor

