import torch
from leffa.model import LeffaModel
model = LeffaModel(pretrained_model_name_or_path="runwayml/stable-diffusion-inpainting", pretrained_model="./ckpts/virtual_tryon.pth", dtype="float16").to("cuda")
print(list(model.unet_encoder.attn_processors.keys()))
