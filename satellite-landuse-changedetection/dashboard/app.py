"""
Streamlit dashboard — upload T1 (before) and T2 (after) images, get
classification, confidence, GradCAM, cosine similarity, risk indicator,
change explanation, heatmap, and a downloadable PDF report.

Run with:
    streamlit run dashboard/app.py
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

import yaml
import numpy as np
import torch
import streamlit as st
from PIL import Image

from src.data.transforms import get_eval_transform, IMAGENET_MEAN, IMAGENET_STD
from src.models.resnet_classifier import build_resnet18
from src.models.embedding_extractor import EmbeddingExtractor
from src.change_detection.similarity import cosine_similarity_single
from src.change_detection.heatmap import compute_patch_similarity_grid, overlay_heatmap_on_image
from src.explainability.gradcam import build_gradcam, generate_gradcam_overlay
from src.transition.transition_rules import generate_explanation
from src.utils.pdf_report import generate_pdf_report

st.set_page_config(page_title="Satellite Land-Use Change Detector", layout="wide")


@st.cache_resource
def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_model(_cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18(num_classes=len(_cfg["data"]["eurosat_classes"]), pretrained=False)
    ckpt_path = f"{_cfg['paths']['checkpoints']}/resnet18_eurosat_finetuned.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()
    embedder = EmbeddingExtractor(model).to(device)
    gradcam = build_gradcam(model)
    return model, embedder, gradcam, device


def classify(model, img_tensor, device, class_names):
    with torch.no_grad():
        logits = model(img_tensor.unsqueeze(0).to(device))
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()
        return class_names[pred_idx], probs[pred_idx].item(), pred_idx


def embed(embedder, img_tensor, device):
    with torch.no_grad():
        return embedder(img_tensor.unsqueeze(0).to(device))


def to_display_array(pil_img, size=64):
    return np.array(pil_img.resize((size, size)).convert("RGB"))


def main():
    cfg = load_config()
    class_names = cfg["data"]["eurosat_classes"]
    thresholds = cfg["dashboard"]["risk_thresholds"]

    st.title("🛰️ Satellite Land-Use Classifier & Change Detector")
    st.caption("Upload a before (T1) and after (T2) image of the same location.")

    col_upload1, col_upload2 = st.columns(2)
    with col_upload1:
        t1_file = st.file_uploader("Before image (T1)", type=["png", "jpg", "jpeg", "tif", "tiff"])
    with col_upload2:
        t2_file = st.file_uploader("After image (T2)", type=["png", "jpg", "jpeg", "tif", "tiff"])

    if not (t1_file and t2_file):
        st.info("Upload both T1 and T2 to run the analysis.")
        return

    try:
        model, embedder, gradcam, device = load_model(cfg)
    except FileNotFoundError:
        st.error(
            "No fine-tuned checkpoint found at "
            f"{cfg['paths']['checkpoints']}/resnet18_eurosat_finetuned.pt — "
            "run Phase 1 + Phase 2 training first (src/training/train_phase2.py)."
        )
        return

    t1_pil = Image.open(t1_file).convert("RGB")
    t2_pil = Image.open(t2_file).convert("RGB")

    transform = get_eval_transform(cfg["data"]["image_size"])
    t1_tensor = transform(t1_pil)
    t2_tensor = transform(t2_pil)

    class_t1, conf_t1, idx_t1 = classify(model, t1_tensor, device, class_names)
    class_t2, conf_t2, idx_t2 = classify(model, t2_tensor, device, class_names)

    emb_t1 = embed(embedder, t1_tensor, device)
    emb_t2 = embed(embedder, t2_tensor, device)
    similarity = cosine_similarity_single(emb_t1, emb_t2)

    explanation = generate_explanation(
        class_t1, class_t2, similarity,
        stable_min=thresholds["stable_min"], moderate_min=thresholds["moderate_min"],
    )

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.image(t1_pil, caption="T1 (Before)", use_container_width=True)
        st.write(f"**Predicted:** {class_t1}")
        st.progress(conf_t1, text=f"Confidence: {conf_t1:.1%}")
        gradcam_t1 = generate_gradcam_overlay(gradcam, t1_tensor, idx_t1, IMAGENET_MEAN, IMAGENET_STD)
        st.image(gradcam_t1, caption="GradCAM — T1", use_container_width=True)

    with col2:
        st.image(t2_pil, caption="T2 (After)", use_container_width=True)
        st.write(f"**Predicted:** {class_t2}")
        st.progress(conf_t2, text=f"Confidence: {conf_t2:.1%}")
        gradcam_t2 = generate_gradcam_overlay(gradcam, t2_tensor, idx_t2, IMAGENET_MEAN, IMAGENET_STD)
        st.image(gradcam_t2, caption="GradCAM — T2", use_container_width=True)

    st.divider()
    st.subheader(f"{explanation['risk_emoji']} {explanation['headline']}")
    st.metric("Cosine Similarity", f"{similarity:.3f}")
    st.write(f"**Previous:** {explanation['previous_class']}  →  **Current:** {explanation['current_class']}")
    st.write(f"**Interpretation:** {explanation['interpretation']}")
    st.caption("Interpretation is a rule-based, templated narrative — not model-generated text.")

    st.divider()
    st.subheader("Change Heatmap")
    sim_grid = compute_patch_similarity_grid(embedder, t1_tensor, t2_tensor, device, grid_size=8)
    t1_display = to_display_array(t1_pil, size=cfg["data"]["image_size"])
    overlay = overlay_heatmap_on_image(t1_display, sim_grid)
    st.image(overlay, caption="Red = higher change intensity", use_container_width=True)

    st.divider()
    if st.button("📄 Generate Analysis Report (PDF)"):
        output_path = "outputs/analysis_report.pdf"
        os.makedirs("outputs", exist_ok=True)
        generate_pdf_report(
            t1_pil, t2_pil, class_t1, conf_t1, class_t2, conf_t2,
            similarity, explanation["risk_level"], explanation["interpretation"],
            overlay, output_path,
        )
        with open(output_path, "rb") as f:
            st.download_button("Download PDF", f, file_name="analysis_report.pdf", mime="application/pdf")


if __name__ == "__main__":
    main()
