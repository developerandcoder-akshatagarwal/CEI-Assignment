"""
Dashboard "Generate Analysis Report" button — assembles the current
T1/T2 analysis into a downloadable PDF. Uses fpdf2 (pure Python, no
system dependencies), so this keeps working with no internet connection.
"""
import tempfile
from fpdf import FPDF
from PIL import Image


def _save_temp_image(pil_or_np_image, suffix=".png"):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    if isinstance(pil_or_np_image, Image.Image):
        pil_or_np_image.save(tmp.name)
    else:
        Image.fromarray(pil_or_np_image).save(tmp.name)
    return tmp.name


def generate_pdf_report(
    t1_image, t2_image, class_t1, conf_t1, class_t2, conf_t2,
    similarity, risk_level, explanation_text, heatmap_image, output_path
):
    """All image args are PIL.Image or numpy uint8 arrays.
    Returns output_path once written."""
    t1_path = _save_temp_image(t1_image)
    t2_path = _save_temp_image(t2_image)
    heatmap_path = _save_temp_image(heatmap_image)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Satellite Change Detection — Analysis Report", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.ln(4)

    # Side-by-side T1/T2 images
    pdf.image(t1_path, x=10, y=pdf.get_y(), w=85)
    pdf.image(t2_path, x=105, y=pdf.get_y(), w=85)
    pdf.ln(65)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(95, 8, "Before (T1)", align="C")
    pdf.cell(95, 8, "After (T2)", align="C", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"T1 Predicted Class: {class_t1}  (confidence: {conf_t1:.1%})", ln=True)
    pdf.cell(0, 8, f"T2 Predicted Class: {class_t2}  (confidence: {conf_t2:.1%})", ln=True)
    pdf.cell(0, 8, f"Cosine Similarity: {similarity:.3f}", ln=True)
    pdf.cell(0, 8, f"Risk Level: {risk_level.upper()}", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Change Interpretation (rule-based, templated):", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, explanation_text)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Change Heatmap:", ln=True)
    pdf.image(heatmap_path, x=55, y=pdf.get_y(), w=100)

    pdf.output(output_path)
    return output_path
