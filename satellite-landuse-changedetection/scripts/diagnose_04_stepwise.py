"""
Diagnostic only -- runs the exact same steps as run_04_ucmerced.py, but
prints after every single stage, so whatever prints last (right before
a silent crash) tells us exactly which line is the problem.
Run from the project root with:
    python -u -m scripts.diagnose_04_stepwise
"""
import sys
def log(msg):
    print(msg)
    sys.stdout.flush()

log("Step 1: importing libraries...")
import yaml
import torch
from torch.utils.data import DataLoader
from src.data.datasets import UCMercedDataset
from src.data.transforms import get_eval_transform
from src.models.resnet_classifier import build_resnet18
from src.training.engine import load_checkpoint
from src.evaluation.class_mapping import build_ucm_to_eurosat_label_map
log("Step 1 OK: all imports succeeded.")

log("Step 2: loading config.yaml...")
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = cfg["data"]["eurosat_classes"]
log(f"Step 2 OK: device={device}")

log("Step 3: building UCMercedDataset (just listing files, no decoding yet)...")
ucm_ds = UCMercedDataset(f"{cfg['paths']['data_raw']}/ucmerced", transform=get_eval_transform(cfg["data"]["image_size"]))
log(f"Step 3 OK: {len(ucm_ds)} samples found across {len(ucm_ds.classes)} classes.")

log("Step 4: building DataLoader (lazy, shouldn't do anything yet)...")
ucm_loader = DataLoader(ucm_ds, batch_size=cfg["data"]["batch_size"], num_workers=cfg["data"]["num_workers"])
log("Step 4 OK.")

log("Step 5: building ResNet18 architecture...")
model = build_resnet18(num_classes=len(classes), pretrained=False)
log("Step 5 OK.")

log("Step 6: loading fine-tuned checkpoint weights from disk...")
model = load_checkpoint(model, f"{cfg['paths']['checkpoints']}/resnet18_eurosat_finetuned.pt", device).to(device)
log("Step 6 OK.")

log("Step 7: building EuroSAT<->UCMerced label map (pure Python, no I/O)...")
label_map = build_ucm_to_eurosat_label_map(classes, ucm_ds.classes)
log(f"Step 7 OK: {len(label_map)} mapped classes.")

log("Step 8: pulling the FIRST BATCH from the DataLoader (this is where real image decoding + transforms + batching all actually happen for the first time)...")
first_batch = next(iter(ucm_loader))
images, labels, paths = first_batch
log(f"Step 8 OK: got a batch of shape {images.shape}")

log("Step 9: running that one batch through the model on device...")
model.eval()
with torch.no_grad():
    images_on_device = images.to(device)
    log("Step 9a OK: moved batch to device.")
    output = model(images_on_device)
    log(f"Step 9b OK: model forward pass succeeded, output shape {output.shape}")

log("\nALL STEPS COMPLETED. If you see this, the crash is not in any of these individual steps -- it may only occur over the FULL dataset (all ~66 batches), not just the first one.")
