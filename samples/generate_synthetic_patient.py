"""Generate three synthetic outside-records PDFs for one fake patient.

Patient: Jane R. Doe, 68F, MRN 00831742, DOB 1957-03-14.
Conditions: HTN, HFrEF (EF 30%, ischemic), persistent AFib on apixaban,
CAD s/p PCI to LAD 2022, CKD stage 3, HLD on atorvastatin, T2DM.

The PDFs are intentionally formatted like outside records so the
Claude-vision extractor has to find facts in realistic context.
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib import colors

OUT_DIR = Path(__file__).parent / "outside_records"
OUT_DIR.mkdir(parents=True, exist_ok=True)

styles = getSampleStyleSheet()
H = styles["Heading2"]
H.spaceAfter = 4
SH = ParagraphStyle("SH", parent=styles["Heading4"], spaceAfter=2, textColor=colors.HexColor("#1a3a6b"))
B = ParagraphStyle("B", parent=styles["BodyText"], leading=12, fontSize=9.5)
SMALL = ParagraphStyle("SMALL", parent=styles["BodyText"], leading=10, fontSize=8.5, textColor=colors.grey)


def _header_table(facility: str, doc_type: str, encounter_date: str) -> Table:
    data = [
        [Paragraph(f"<b>{facility}</b>", B), Paragraph(f"<b>{doc_type}</b>", B)],
        [Paragraph("123 Main St, Springfield IL 62701  ·  (217) 555-0143", SMALL),
         Paragraph(f"Encounter date: {encounter_date}", SMALL)],
        [Paragraph("Patient: <b>Doe, Jane R.</b>  ·  MRN: <b>00831742</b>  ·  "
                   "DOB: 1957-03-14 (68F)  ·  Sex: Female", B), ""],
    ]
    t = Table(data, colWidths=[4.0 * inch, 3.0 * inch])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 2), (-1, 2), 0.5, colors.grey),
        ("BOTTOMPADDING", (0, 2), (-1, 2), 6),
    ]))
    return t


def _section(title: str, body_html: str) -> list:
    return [Paragraph(title, SH), Paragraph(body_html, B), Spacer(1, 4)]


def build_discharge_summary() -> None:
    doc = SimpleDocTemplate(
        str(OUT_DIR / "discharge_summary.pdf"),
        pagesize=LETTER,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
        topMargin=0.5 * inch, bottomMargin=0.5 * inch,
    )
    story = [
        _header_table("Mercy General Hospital", "HOSPITAL DISCHARGE SUMMARY",
                      "2026-04-22"),
        Spacer(1, 6),
    ]
    story += _section(
        "Admission / Discharge",
        "Admitted 2026-04-18 with acute decompensated heart failure. "
        "Discharged 2026-04-22 to home. Attending: Sarah Chen, MD (Cardiology)."
    )
    story += _section(
        "Reason for Admission",
        "68-year-old woman with HFrEF (LVEF 30%, ischemic), persistent atrial "
        "fibrillation, CAD s/p PCI to LAD in 2022, hypertension, type 2 diabetes, "
        "and CKD stage 3 presented with 5 days of progressive dyspnea on exertion, "
        "orthopnea, and 8-lb weight gain. She was diuresed with IV furosemide and "
        "transitioned back to oral therapy prior to discharge."
    )
    story += _section(
        "Past Medical History",
        "• Heart failure with reduced ejection fraction (HFrEF), LVEF 30% (ischemic)<br/>"
        "• Coronary artery disease, s/p PCI to LAD with DES, 2022<br/>"
        "• Atrial fibrillation, persistent, CHA₂DS₂-VASc score 5<br/>"
        "• Hypertension<br/>"
        "• Hyperlipidemia<br/>"
        "• Type 2 diabetes mellitus, on metformin<br/>"
        "• Chronic kidney disease, stage 3 (baseline eGFR ~45 mL/min/1.73m²)"
    )
    story += _section(
        "Past Surgical History",
        "Appendectomy (1985). PCI to LAD with drug-eluting stent (2022)."
    )
    story += _section(
        "Family History",
        "Father: MI at age 64. Mother: type 2 diabetes, stroke at age 78."
    )
    story += _section(
        "Social History",
        "Lives with husband. Former smoker, 20 pack-years, quit 2018. "
        "No alcohol. No illicit drug use. Walks 1 block before stopping for dyspnea."
    )
    story += _section(
        "Allergies",
        "No known drug allergies."
    )
    story += _section(
        "Discharge Medications",
        "1. Lisinopril 20 mg PO daily<br/>"
        "2. Metoprolol succinate 50 mg PO daily<br/>"
        "3. Furosemide 40 mg PO daily<br/>"
        "4. Apixaban 5 mg PO BID<br/>"
        "5. Atorvastatin 40 mg PO daily<br/>"
        "6. Aspirin 81 mg PO daily<br/>"
        "7. Metformin 1000 mg PO BID"
    )
    story += _section(
        "Vitals at Discharge",
        "BP 142/82 mm Hg · HR 76 (irregular) · RR 16 · SpO₂ 96% on room air · "
        "Weight 72.4 kg (dry weight 70.0 kg)."
    )
    story += _section(
        "Pertinent Labs (most recent during admission)",
        "Sodium 138 · Potassium 4.2 · Chloride 102 · CO₂ 24 · "
        "BUN 28 · Creatinine 1.4 · eGFR 45 mL/min/1.73m² · Glucose 168<br/>"
        "Hemoglobin 11.8 · WBC 6.8 · Platelets 218<br/>"
        "LDL-C 95 mg/dL · HDL 38 · Triglycerides 162 · Total cholesterol 168<br/>"
        "HbA1c 7.8% · NT-proBNP 4,820 pg/mL · Troponin I peak 0.04 ng/mL"
    )
    story += _section(
        "Imaging / Procedures",
        "TTE 2026-04-19: LVEF 30%, severe global LV systolic dysfunction, "
        "moderate functional mitral regurgitation. See separate report.<br/>"
        "Coronary angiography 2022: 90% mid-LAD stenosis, treated with DES."
    )
    story += _section(
        "Hospital Course",
        "Patient achieved net diuresis of 4.2 L over four days with IV furosemide, "
        "then transitioned to her home oral dose. Telemetry showed persistent AFib "
        "with controlled ventricular response on metoprolol. Apixaban was continued "
        "throughout the stay. No ischemic events on telemetry. Discharged in "
        "compensated state, ambulating without supplemental oxygen."
    )
    story += _section(
        "Discharge Diagnoses",
        "1. Acute decompensated heart failure with reduced ejection fraction (HFrEF)<br/>"
        "2. Persistent atrial fibrillation, anticoagulated<br/>"
        "3. Coronary artery disease, s/p PCI to LAD 2022<br/>"
        "4. Hypertension<br/>"
        "5. Hyperlipidemia<br/>"
        "6. Type 2 diabetes mellitus<br/>"
        "7. Chronic kidney disease, stage 3"
    )
    story += _section(
        "Follow-Up",
        "Cardiology clinic in 1 week (Dr. Chen). Primary care in 2 weeks. "
        "Daily weights; call if weight gain >3 lb in 2 days."
    )
    story += [Paragraph("Electronically signed: Sarah Chen, MD — 2026-04-22 14:21 CT", SMALL)]
    doc.build(story)


def build_clinic_note() -> None:
    doc = SimpleDocTemplate(
        str(OUT_DIR / "cardiology_clinic_note.pdf"),
        pagesize=LETTER,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
        topMargin=0.5 * inch, bottomMargin=0.5 * inch,
    )
    story = [
        _header_table("Springfield Cardiology Associates", "OUTPATIENT FOLLOW-UP NOTE",
                      "2026-04-30"),
        Spacer(1, 6),
    ]
    story += _section(
        "Subjective",
        "68F seen 8 days post-discharge from Mercy General after admission for "
        "acute decompensated HFrEF. Reports improved dyspnea, currently NYHA class II. "
        "Down 2 lb since discharge. No chest pain, no syncope. Adherent with all "
        "discharge medications."
    )
    story += _section(
        "Problem List",
        "• HFrEF, LVEF 30% (ischemic etiology)<br/>"
        "• CAD s/p PCI to LAD with DES (2022)<br/>"
        "• Persistent atrial fibrillation, CHA₂DS₂-VASc 5, on apixaban<br/>"
        "• Hypertension — currently above goal<br/>"
        "• Hyperlipidemia<br/>"
        "• Type 2 diabetes mellitus<br/>"
        "• Chronic kidney disease, stage 3"
    )
    story += _section(
        "Current Medications (verified today)",
        "Lisinopril 20 mg PO daily · Metoprolol succinate 50 mg PO daily · "
        "Furosemide 40 mg PO daily · Apixaban 5 mg PO BID · "
        "Atorvastatin 40 mg PO daily · Aspirin 81 mg PO daily · "
        "Metformin 1000 mg PO BID."
    )
    story += _section(
        "Vitals",
        "BP 138/84 mm Hg · HR 72 (irregular) · Weight 71.5 kg · BMI 27.4."
    )
    story += _section(
        "Labs (this visit)",
        "BMP: Na 139, K 4.1, Cr 1.4, eGFR 45 · NT-proBNP 1,640 pg/mL · "
        "LDL-C 95 mg/dL · HbA1c 7.6%."
    )
    story += _section(
        "Assessment and Plan",
        "<b>1. HFrEF (LVEF 30%, ischemic):</b> Patient is on lisinopril and "
        "metoprolol succinate, but is NOT on sacubitril/valsartan, an MRA, or "
        "an SGLT2 inhibitor. GDMT is incomplete. Discussed transition to "
        "sacubitril/valsartan, addition of spironolactone (K and eGFR permit), "
        "and starting dapagliflozin. Patient agreeable; will start MRA today and "
        "transition ACEi to ARNI after 36-hour washout next week. SGLT2i to be "
        "started concurrently.<br/><br/>"
        "<b>2. Atrial fibrillation:</b> Persistent, rate-controlled on metoprolol. "
        "Continue apixaban 5 mg BID for stroke prevention "
        "(CHA₂DS₂-VASc 5).<br/><br/>"
        "<b>3. CAD s/p PCI:</b> Continue aspirin 81 mg daily and atorvastatin 40 mg. "
        "LDL-C 95 on max-tolerated statin — discussed adding ezetimibe given very "
        "high-risk ASCVD status.<br/><br/>"
        "<b>4. Hypertension:</b> BP 138/84, above goal of <130/80. Will reassess after "
        "ARNI initiation, which often addresses BP simultaneously.<br/><br/>"
        "<b>5. T2DM:</b> HbA1c 7.6 on metformin. SGLT2i will provide additional "
        "glycemic benefit on top of HF benefit. Coordinate with PCP.<br/><br/>"
        "<b>6. CKD3:</b> Stable, eGFR 45. Will monitor with MRA and ARNI titration."
    )
    story += [Paragraph("Electronically signed: Sarah Chen, MD — 2026-04-30 11:08 CT", SMALL)]
    doc.build(story)


def build_echo_report() -> None:
    doc = SimpleDocTemplate(
        str(OUT_DIR / "echo_report.pdf"),
        pagesize=LETTER,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
        topMargin=0.5 * inch, bottomMargin=0.5 * inch,
    )
    story = [
        _header_table("Mercy General Hospital — Echocardiography Lab",
                      "TRANSTHORACIC ECHOCARDIOGRAM REPORT",
                      "2026-04-19"),
        Spacer(1, 6),
    ]
    story += _section(
        "Indication",
        "Acute decompensated heart failure; reassessment of LV function."
    )
    story += _section(
        "Study Details",
        "Date: 2026-04-19 · Sonographer: J. Patel, RDCS · Reading physician: "
        "M. Reyes, MD · Image quality: adequate · Rhythm during study: atrial fibrillation."
    )
    story += _section(
        "Measurements",
        "LV end-diastolic diameter 5.8 cm · LV end-systolic diameter 4.6 cm · "
        "LV ejection fraction (biplane Simpson's) 30% · "
        "LA volume index 42 mL/m² · "
        "RV size and systolic function normal · TAPSE 1.9 cm."
    )
    story += _section(
        "Findings",
        "<b>Left Ventricle:</b> Mildly dilated. Severe global LV systolic dysfunction. "
        "Estimated LVEF of 30%. Global hypokinesis with regional akinesis of the anterior wall and apex.<br/><br/>"
        "<b>Right Ventricle:</b> Normal size and systolic function.<br/><br/>"
        "<b>Atria:</b> Left atrium moderately dilated. Right atrium mildly dilated.<br/><br/>"
        "<b>Valves:</b> Mitral valve with moderate functional regurgitation due to annular "
        "dilation and tethering. Aortic valve trileaflet, no significant stenosis or regurgitation. "
        "Tricuspid valve with mild regurgitation, estimated PASP 38 mm Hg.<br/><br/>"
        "<b>Pericardium:</b> No effusion."
    )
    story += _section(
        "Conclusions",
        "1. LVEF 30% with severe global LV systolic dysfunction.<br/>"
        "2. Moderate functional mitral regurgitation.<br/>"
        "3. Moderate left atrial dilation, consistent with chronic atrial fibrillation.<br/>"
        "4. No pericardial effusion."
    )
    story += [Paragraph("Electronically signed: Miguel Reyes, MD — 2026-04-19 16:42 CT", SMALL)]
    doc.build(story)


if __name__ == "__main__":
    build_discharge_summary()
    build_clinic_note()
    build_echo_report()
    print(f"Wrote PDFs to {OUT_DIR}")
