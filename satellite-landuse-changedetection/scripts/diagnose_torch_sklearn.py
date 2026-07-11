"""
Diagnostic only -- tests whether importing torch and scikit-learn
together (in that order) causes a crash, isolated from everything else
in the project.
Run from the project root with:
    python -u -m scripts.diagnose_torch_sklearn
"""
import sys
def log(msg):
    print(msg)
    sys.stdout.flush()

log("Importing torch...")
import torch
log("torch imported OK.")

log("Importing torchvision...")
import torchvision
log("torchvision imported OK.")

log("Importing sklearn.metrics...")
from sklearn.metrics import f1_score
log("sklearn.metrics imported OK.")

log("Importing scipy (used internally by sklearn)...")
import scipy
log("scipy imported OK.")

log("Importing cv2 (opencv)...")
import cv2
log("cv2 imported OK.")

log("Importing pytorch_grad_cam...")
import pytorch_grad_cam
log("pytorch_grad_cam imported OK.")

log("\nALL IMPORTS SUCCEEDED with no crash.")
