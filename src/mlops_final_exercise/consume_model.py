import torch
import wandb

from mlops_final_exercise.model import MyAwesomeModel

run = wandb.init()
artifact = run.use_artifact(
    "michaelfn-technical-university-of-denmark-org/wandb-registry-corrupt-mnist-model/corrupt-mnist-model:v0",
    type="model",
)
artifact_dir = artifact.download()
model = MyAwesomeModel()
# model.load_state_dict(torch.load("<artifact_dir>/model.ckpt"))
model.load_state_dict(torch.load(f"{artifact_dir}\model.pth"))
