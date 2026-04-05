import torch
ckpt = torch.load("models/forecast_model.pt", map_location="cpu")
print(ckpt.keys())
if "model_state_dict" in ckpt:
    for k, v in list(ckpt["model_state_dict"].items())[:5]:
        print(k, v.shape)
