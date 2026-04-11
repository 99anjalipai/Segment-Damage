"""
pdf_generator.py

Generates a professional PDF claim report containing:
- Claimant and vehicle information
- Original and overlayed damage images
- The 4-section LLM-generated claim report
- Repair cost estimates
- Policy coverage summary
"""

from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import io
import tempfile

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.lib.colors import HexColor, black, white
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image as RLImage, PageBreak, HRFlowable,
    )
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# Brand colors
_BLUE = HexColor("#1E3A8A") if REPORTLAB_AVAILABLE else None
_LIGHT_BLUE = HexColor("#2563EB") if REPORTLAB_AVAILABLE else None
_GRAY = HexColor("#6B7280") if REPORTLAB_AVAILABLE else None
_LIGHT_GRAY = HexColor("#F3F4F6") if REPORTLAB_AVAILABLE else None
_DARK = HexColor("#1F2937") if REPORTLAB_AVAILABLE else None


def _get_styles():
    """Create custom paragraph styles for the PDF."""
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ClaimTitle",
        fontName="Helvetica-Bold",
        fontSize=18,
        textColor=_BLUE,
        spaceAfter=6,
        spaceBefore=12,
    ))
    styles.add(ParagraphStyle(
        name="SectionHeader",
        fontName="Helvetica-Bold",
        fontSize=13,
        textColor=_LIGHT_BLUE,
        spaceAfter=6,
        spaceBefore=16,
        borderWidth=0,
        borderPadding=0,
    ))
    styles.add(ParagraphStyle(
        name="SubHeader",
        fontName="Helvetica-Bold",
        fontSize=10,
        textColor=_DARK,
        spaceAfter=4,
        spaceBefore=8,
    ))
    styles.add(ParagraphStyle(
        name="ClaimBody",
        fontName="Helvetica",
        fontSize=9,
        textColor=black,
        spaceAfter=4,
        spaceBefore=2,
        leading=13,
    ))
    styles.add(ParagraphStyle(
        name="ClaimBodySmall",
        fontName="Helvetica",
        fontSize=8,
        textColor=_GRAY,
        spaceAfter=2,
        spaceBefore=1,
        leading=11,
    ))
    styles.add(ParagraphStyle(
        name="Footer",
        fontName="Helvetica",
        fontSize=7,
        textColor=_GRAY,
        alignment=TA_CENTER,
    ))
    return styles


def _pil_to_rl_image(pil_img, max_width=None, max_height=None):
    """Convert a PIL Image to a ReportLab Image flowable."""
    if max_width is None:
        max_width = 4.5 * inch
    if max_height is None:
        max_height = 3 * inch

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)

    img_w, img_h = pil_img.size
    aspect = img_w / img_h

    width = min(max_width, img_w)
    height = width / aspect
    if height > max_height:
        height = max_height
        width = height * aspect

    return RLImage(buf, width=width, height=height)


def _np_to_rl_image(np_img, max_width=None, max_height=None):
    """Convert a numpy array image to a ReportLab Image flowable."""
    from PIL import Image as PILImage
    pil_img = PILImage.fromarray(np_img)
    return _pil_to_rl_image(pil_img, max_width, max_height)


def _add_header_footer(canvas, doc):
    """Draw header and footer on every page."""
    canvas.saveState()

    # Header line
    canvas.setStrokeColor(_LIGHT_BLUE)
    canvas.setLineWidth(2)
    canvas.line(
        doc.leftMargin, letter[1] - 0.5 * inch,
        letter[0] - doc.rightMargin, letter[1] - 0.5 * inch,
    )
    canvas.setFont("Helvetica-Bold", 10)
    canvas.setFillColor(_BLUE)
    canvas.drawString(doc.leftMargin, letter[1] - 0.45 * inch, "AutoClaim AI")
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(_GRAY)
    canvas.drawRightString(
        letter[0] - doc.rightMargin,
        letter[1] - 0.45 * inch,
        f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
    )

    # Footer
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(_GRAY)
    canvas.drawCentredString(
        letter[0] / 2, 0.4 * inch,
        f"Page {doc.page} | AutoClaim AI - Vehicle Damage Assessment & Claims Assistant | Confidential",
    )

    canvas.restoreState()


def _text_to_paragraphs(text: str, style) -> list:
    """Split multiline text into Paragraph flowables, preserving line breaks."""
    elements = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            elements.append(Spacer(1, 4))
            continue
        # Escape HTML entities
        line = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        elements.append(Paragraph(line, style))
    return elements


def generate_claim_pdf(
    sections: Dict[str, str],
    claimant_info: Dict[str, str],
    vehicle_info: Dict[str, str],
    incident_info: Dict[str, str],
    insurance_info: Dict[str, str],
    original_images: Optional[List] = None,
    overlayed_images: Optional[List] = None,
    repair_estimate_text: Optional[str] = None,
    policy_summary: Optional[str] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    Generate a professional PDF claim report.

    Args:
        sections: Dict with keys 'claim_letter', 'damage_analysis',
                  'coverage_assessment', 'action_plan'.
        claimant_info: Dict with user_name, user_address, etc.
        vehicle_info: Dict with vehicle_year, vehicle_make, etc.
        incident_info: Dict with incident_date, incident_time, etc.
        insurance_info: Dict with insurance_company, policy_number.
        original_images: List of PIL Images (original vehicle photos).
        overlayed_images: List of numpy arrays (segmentation overlays).
        repair_estimate_text: Formatted repair estimate string.
        policy_summary: Formatted policy coverage summary.
        output_path: Where to save the PDF. If None, uses a temp file.

    Returns:
        Path to the generated PDF file.
    """
    if not REPORTLAB_AVAILABLE:
        raise ImportError(
            "reportlab is required for PDF generation. "
            "Install with: pip install reportlab"
        )

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(
            Path(tempfile.gettempdir()) / f"autoclaim_report_{timestamp}.pdf"
        )

    styles = _get_styles()
    elements = []

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
    )

    # ---- Title Page ----
    elements.append(Spacer(1, 0.5 * inch))
    elements.append(Paragraph("Insurance Claim Report", styles["ClaimTitle"]))
    elements.append(Spacer(1, 4))
    elements.append(HRFlowable(
        width="100%", thickness=1, color=_LIGHT_BLUE, spaceAfter=12,
    ))

    # Claimant summary table
    summary_data = [
        ["Claimant:", claimant_info.get("user_name", "[UNKNOWN]"),
         "Policy #:", insurance_info.get("policy_number", "[UNKNOWN]")],
        ["Phone:", claimant_info.get("user_phone", "[UNKNOWN]"),
         "Insurer:", insurance_info.get("insurance_company", "[UNKNOWN]")],
        ["Email:", claimant_info.get("user_email", "[UNKNOWN]"),
         "Date:", incident_info.get("incident_date", "[UNKNOWN]")],
        ["Address:", claimant_info.get("user_address", "[UNKNOWN]"),
         "Location:", incident_info.get("incident_location", "[UNKNOWN]")],
        ["Vehicle:", f"{vehicle_info.get('vehicle_year', '')} {vehicle_info.get('vehicle_make', '')} {vehicle_info.get('vehicle_model', '')}".strip(),
         "VIN:", vehicle_info.get("vehicle_vin", "[UNKNOWN]")],
    ]

    summary_table = Table(summary_data, colWidths=[0.8 * inch, 2.7 * inch, 0.8 * inch, 2.7 * inch])
    summary_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTNAME", (3, 0), (3, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), _DARK),
        ("TEXTCOLOR", (2, 0), (2, -1), _DARK),
        ("TEXTCOLOR", (1, 0), (1, -1), black),
        ("TEXTCOLOR", (3, 0), (3, -1), black),
        ("BACKGROUND", (0, 0), (-1, -1), _LIGHT_GRAY),
        ("BOX", (0, 0), (-1, -1), 0.5, _LIGHT_BLUE),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, HexColor("#E5E7EB")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 16))

    # ---- Damage Images ----
    if original_images or overlayed_images:
        elements.append(Paragraph("Vehicle Damage Evidence", styles["SectionHeader"]))
        elements.append(HRFlowable(
            width="100%", thickness=0.5, color=HexColor("#E5E7EB"), spaceAfter=8,
        ))

        num_images = max(
            len(original_images or []),
            len(overlayed_images or []),
        )

        for i in range(num_images):
            elements.append(Paragraph(f"Image {i + 1}", styles["SubHeader"]))
            image_row = []

            if original_images and i < len(original_images):
                img = original_images[i]
                rl_img = _pil_to_rl_image(img, max_width=3.2 * inch, max_height=2.4 * inch)
                image_row.append(rl_img)

            if overlayed_images and i < len(overlayed_images):
                ov = overlayed_images[i]
                rl_ov = _np_to_rl_image(ov, max_width=3.2 * inch, max_height=2.4 * inch)
                image_row.append(rl_ov)

            if image_row:
                img_table = Table([image_row], colWidths=[3.4 * inch] * len(image_row))
                img_table.setStyle(TableStyle([
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]))
                elements.append(img_table)

                label_row = []
                if original_images and i < len(original_images):
                    label_row.append(Paragraph("Original", styles["ClaimBodySmall"]))
                if overlayed_images and i < len(overlayed_images):
                    label_row.append(Paragraph("AI Damage Detection", styles["ClaimBodySmall"]))
                if label_row:
                    label_table = Table([label_row], colWidths=[3.4 * inch] * len(label_row))
                    label_table.setStyle(TableStyle([
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ]))
                    elements.append(label_table)

            elements.append(Spacer(1, 8))

    # ---- Section 1: Claim Letter ----
    elements.append(PageBreak())
    elements.append(Paragraph("Section 1: Insurance Claim Letter", styles["SectionHeader"]))
    elements.append(HRFlowable(
        width="100%", thickness=0.5, color=HexColor("#E5E7EB"), spaceAfter=8,
    ))
    elements.extend(_text_to_paragraphs(
        sections.get("claim_letter", "[Claim letter not generated]"),
        styles["ClaimBody"],
    ))

    # ---- Section 2: Damage Analysis ----
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Section 2: Damage Analysis", styles["SectionHeader"]))
    elements.append(HRFlowable(
        width="100%", thickness=0.5, color=HexColor("#E5E7EB"), spaceAfter=8,
    ))
    elements.extend(_text_to_paragraphs(
        sections.get("damage_analysis", "[Damage analysis not generated]"),
        styles["ClaimBody"],
    ))

    # ---- Repair Estimate (if available) ----
    if repair_estimate_text:
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Repair Cost Estimate", styles["SectionHeader"]))
        elements.append(HRFlowable(
            width="100%", thickness=0.5, color=HexColor("#E5E7EB"), spaceAfter=8,
        ))
        elements.extend(_text_to_paragraphs(
            repair_estimate_text, styles["ClaimBody"],
        ))

    # ---- Section 3: Coverage Assessment ----
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Section 3: Coverage Assessment", styles["SectionHeader"]))
    elements.append(HRFlowable(
        width="100%", thickness=0.5, color=HexColor("#E5E7EB"), spaceAfter=8,
    ))
    elements.extend(_text_to_paragraphs(
        sections.get("coverage_assessment", "[Coverage assessment not generated]"),
        styles["ClaimBody"],
    ))

    # ---- Policy Summary (if available) ----
    if policy_summary:
        elements.append(Spacer(1, 8))
        elements.append(Paragraph("Policy Reference", styles["SubHeader"]))
        elements.extend(_text_to_paragraphs(
            policy_summary, styles["ClaimBodySmall"],
        ))

    # ---- Section 4: Action Plan ----
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Section 4: Your Action Plan", styles["SectionHeader"]))
    elements.append(HRFlowable(
        width="100%", thickness=0.5, color=HexColor("#E5E7EB"), spaceAfter=8,
    ))
    elements.extend(_text_to_paragraphs(
        sections.get("action_plan", "[Action plan not generated]"),
        styles["ClaimBody"],
    ))

    # ---- Disclaimer ----
    elements.append(Spacer(1, 24))
    elements.append(HRFlowable(
        width="100%", thickness=0.5, color=HexColor("#E5E7EB"), spaceAfter=8,
    ))
    elements.append(Paragraph(
        "DISCLAIMER: This report was generated by AutoClaim AI using computer vision "
        "and language models. It is intended as a draft to assist in filing an insurance "
        "claim and does not constitute legal or financial advice. All damage assessments "
        "and repair cost estimates are approximate and should be verified by a licensed "
        "adjuster and certified body shop. Coverage determinations are based on the "
        "policy information provided and may not reflect all terms and conditions of "
        "your actual policy. Please review all content before submission.",
        styles["ClaimBodySmall"],
    ))

    # Build PDF
    doc.build(elements, onFirstPage=_add_header_footer, onLaterPages=_add_header_footer)
    return output_path