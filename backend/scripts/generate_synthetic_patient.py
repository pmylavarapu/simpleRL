"""Generate a synthetic, obviously-fake patient record bundle for demo/testing.

Usage:
    python -m scripts.generate_synthetic_patient --out /tmp/demo_patient

Produces several PDFs (H&P, med list, lab report, echo, cath report) that exercise
the deterministic extractors and give the LLM enough surface to synthesize a useful
one-pager. All data is synthetic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def _write(path: Path, paragraphs: Iterable[str]) -> None:
    styles = getSampleStyleSheet()
    body = styles["BodyText"]
    h1 = styles["Heading1"]
    doc = SimpleDocTemplate(str(path), pagesize=LETTER)
    flow = []
    for p in paragraphs:
        if p.startswith("# "):
            flow.append(Paragraph(p[2:], h1))
        elif p == "":
            flow.append(Spacer(1, 8))
        else:
            flow.append(Paragraph(p, body))
    doc.build(flow)


HP = [
    "# History &amp; Physical — DEMO PATIENT (SYNTHETIC)",
    "Patient: Jane Demo  MRN: 000000  DOB: 01/01/1958  Date of Visit: 03/15/2026",
    "",
    "<b>Chief Complaint:</b> Worsening dyspnea on exertion and bilateral leg swelling over the last 3 weeks.",
    "",
    "<b>HPI:</b> 68-year-old woman with known coronary artery disease, hypertension, type 2 diabetes, "
    "and prior anterior MI in 2022 presents with progressive shortness of breath. She reports orthopnea "
    "requiring two pillows and a 6-pound weight gain. She denies chest pain or palpitations. "
    "Functional class has declined from NYHA II to NYHA III over the past month.",
    "",
    "<b>Past Medical History:</b> Coronary artery disease s/p anterior STEMI 2022, ischemic cardiomyopathy, "
    "hypertension, type 2 diabetes mellitus, hyperlipidemia, chronic kidney disease stage 3a, "
    "paroxysmal atrial fibrillation.",
    "",
    "<b>Past Surgical History:</b> Percutaneous coronary intervention with drug-eluting stent to "
    "proximal LAD in 2022. Cholecystectomy 2010. Right total knee arthroplasty 2018.",
    "",
    "<b>Family History:</b> Father died of myocardial infarction at age 62. Mother with type 2 diabetes "
    "and stroke at age 70. One sibling with hypertension.",
    "",
    "<b>Social History:</b> Former smoker, 30 pack-year history, quit 2022. Drinks alcohol socially "
    "(1-2 drinks per week). Retired schoolteacher. Lives with spouse.",
    "",
    "<b>Physical Exam:</b> BP 148/86 mmHg, HR 88 bpm irregular, RR 20, SpO2 94% on room air. "
    "JVP elevated at 10 cm. Bibasilar crackles. S3 gallop. 2+ pitting edema to the knees bilaterally.",
]

MEDS = [
    "# Outpatient Medication List — DEMO PATIENT (SYNTHETIC)",
    "Patient: Jane Demo  MRN: 000000",
    "",
    "lisinopril 20 mg PO daily",
    "metoprolol succinate 50 mg PO daily",
    "furosemide 40 mg PO daily",
    "atorvastatin 40 mg PO at bedtime",
    "aspirin 81 mg PO daily",
    "clopidogrel 75 mg PO daily",
    "apixaban 5 mg PO twice daily",
    "metformin 1000 mg PO twice daily",
    "empagliflozin 10 mg PO daily",
    "spironolactone 25 mg PO daily",
]

LABS = [
    "# Laboratory Report — DEMO PATIENT (SYNTHETIC)",
    "Patient: Jane Demo  MRN: 000000  Collected: 03/14/2026",
    "",
    "<b>Comprehensive Metabolic Panel</b>",
    "Sodium: 138 mmol/L",
    "Potassium: 4.2 mmol/L",
    "Creatinine: 1.4 mg/dL",
    "eGFR: 48 mL/min/1.73m2",
    "",
    "<b>Cardiac</b>",
    "NT-proBNP: 2450 pg/mL",
    "Troponin I: 0.02 ng/mL",
    "",
    "<b>Lipid Panel</b>",
    "Total cholesterol: 178 mg/dL",
    "LDL: 112 mg/dL",
    "HDL: 38 mg/dL",
    "Triglycerides: 165 mg/dL",
    "",
    "<b>Glycemic</b>",
    "HbA1c: 7.8 %",
    "",
    "<b>Complete Blood Count</b>",
    "Hemoglobin: 11.2 g/dL",
    "Platelets: 215 x10^9/L",
    "INR: 1.0",
]

ECHO = [
    "# Transthoracic Echocardiogram Report — DEMO PATIENT (SYNTHETIC)",
    "Patient: Jane Demo  MRN: 000000  Study Date: 03/14/2026",
    "",
    "<b>Indication:</b> Worsening heart failure symptoms.",
    "",
    "<b>Findings:</b>",
    "Left ventricular ejection fraction is 32 % (severely reduced). "
    "There is global hypokinesis with akinesis of the anterior wall and apex consistent with prior infarction.",
    "Left ventricle is moderately dilated. Left atrium is moderately dilated.",
    "Mitral valve: moderate mitral regurgitation, jet directed posteriorly.",
    "Tricuspid valve: mild tricuspid regurgitation. Pulmonary artery systolic pressure: 42 mmHg.",
    "Aortic valve: trileaflet, mild aortic stenosis with peak gradient 18 mmHg.",
    "",
    "<b>Conclusion:</b> Severely reduced LV systolic function (LVEF 32 %), ischemic pattern. "
    "Moderate MR. Mild AS. Elevated PASP suggesting elevated left-sided filling pressures.",
]

CATH = [
    "# Cardiac Catheterization Report — DEMO PATIENT (SYNTHETIC)",
    "Patient: Jane Demo  MRN: 000000  Procedure Date: 04/02/2022",
    "",
    "<b>Indication:</b> Acute anterior ST-elevation myocardial infarction.",
    "",
    "<b>Coronary Angiography:</b>",
    "Left main: no significant stenosis.",
    "LAD: proximal 99 % stenosis with TIMI 1 flow, culprit lesion.",
    "LCx: 40 % stenosis mid vessel.",
    "RCA: 50 % stenosis proximal segment, no intervention.",
    "",
    "<b>Intervention:</b> Primary PCI to proximal LAD with deployment of a 3.0 x 18 mm "
    "drug-eluting stent. Post-procedure TIMI 3 flow, 0 % residual stenosis.",
    "",
    "<b>Conclusion:</b> Successful primary PCI of culprit proximal LAD lesion. Moderate non-culprit "
    "RCA disease deferred for medical management.",
]


DOCS = {
    "h_and_p.pdf": HP,
    "medications.pdf": MEDS,
    "labs.pdf": LABS,
    "echo.pdf": ECHO,
    "cath.pdf": CATH,
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="Output directory")
    args = p.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, paragraphs in DOCS.items():
        path = outdir / name
        _write(path, paragraphs)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
