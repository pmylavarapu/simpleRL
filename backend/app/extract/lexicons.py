"""Small lexicons used by the deterministic extractors.

Kept intentionally short and cardiology-leaning — extend as needed.
"""

from __future__ import annotations

# Common cardiac & comorbidity medications. Lowercase, no dose info.
MEDICATIONS: set[str] = {
    # ACE / ARB / ARNI
    "lisinopril", "enalapril", "ramipril", "benazepril", "captopril",
    "losartan", "valsartan", "olmesartan", "irbesartan", "telmisartan", "candesartan",
    "sacubitril/valsartan", "sacubitril-valsartan", "entresto",
    # Beta blockers
    "metoprolol", "carvedilol", "bisoprolol", "atenolol", "nebivolol", "propranolol",
    # Diuretics
    "furosemide", "torsemide", "bumetanide", "spironolactone", "eplerenone",
    "hydrochlorothiazide", "chlorthalidone", "metolazone",
    # Calcium channel blockers
    "amlodipine", "diltiazem", "verapamil", "nifedipine", "felodipine",
    # Statins / lipid
    "atorvastatin", "rosuvastatin", "simvastatin", "pravastatin", "lovastatin",
    "ezetimibe", "evolocumab", "alirocumab",
    # Antiplatelets / anticoagulants
    "aspirin", "clopidogrel", "ticagrelor", "prasugrel",
    "warfarin", "apixaban", "rivaroxaban", "dabigatran", "edoxaban",
    # SGLT2 / GLP-1 / DM
    "empagliflozin", "dapagliflozin", "canagliflozin", "ertugliflozin",
    "semaglutide", "liraglutide", "dulaglutide", "tirzepatide",
    "metformin", "glipizide", "glyburide", "insulin", "sitagliptin",
    # Antiarrhythmics
    "amiodarone", "sotalol", "flecainide", "propafenone", "dofetilide", "digoxin",
    "ivabradine",
    # Nitrates / other
    "nitroglycerin", "isosorbide", "ranolazine", "hydralazine",
    # Pulmonary / misc
    "albuterol", "tiotropium", "fluticasone", "budesonide",
    "levothyroxine", "omeprazole", "pantoprazole",
}


# Lab analyte name → display name + units (canonical).
LAB_PATTERNS: list[tuple[str, str, str, str]] = [
    # (regex, canonical name, units, kind hint)
    (r"\b(?:hba1c|a1c|hemoglobin\s*a1c)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%?", "HbA1c", "%", "glycemic"),
    (r"\bldl(?:\s*[-]?\s*c|\s*cholesterol)?\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:mg/?dl)?", "LDL", "mg/dL", "lipid"),
    (r"\bhdl(?:\s*[-]?\s*c|\s*cholesterol)?\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:mg/?dl)?", "HDL", "mg/dL", "lipid"),
    (r"\btotal\s*cholesterol\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:mg/?dl)?", "Total cholesterol", "mg/dL", "lipid"),
    (r"\btriglycerides?\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:mg/?dl)?", "Triglycerides", "mg/dL", "lipid"),
    (r"\bnt[-\s]?probnp\s*[:=]?\s*(\d+(?:,?\d+)?)\s*(?:pg/?ml)?", "NT-proBNP", "pg/mL", "cardiac"),
    (r"(?<!nt[-\s])(?<!nt)\bbnp\s*[:=]?\s*(\d+(?:,?\d+)?)\s*(?:pg/?ml)?", "BNP", "pg/mL", "cardiac"),
    (r"\btroponin\s*[itn]?\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:ng/?ml|ng/?l)?", "Troponin", "ng/mL", "cardiac"),
    (r"\bcreatinine\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:mg/?dl)?", "Creatinine", "mg/dL", "renal"),
    (r"\begfr\s*[:=]?\s*(\d+(?:\.\d+)?)", "eGFR", "mL/min/1.73m^2", "renal"),
    (r"\bpotassium\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:mmol/?l|meq/?l)?", "Potassium", "mmol/L", "electrolyte"),
    (r"\bsodium\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:mmol/?l|meq/?l)?", "Sodium", "mmol/L", "electrolyte"),
    (r"\bhemoglobin\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:g/?dl)?", "Hemoglobin", "g/dL", "heme"),
    (r"\bplatelets?\s*[:=]?\s*(\d+(?:,?\d+)?)", "Platelets", "x10^9/L", "heme"),
    (r"\binr\s*[:=]?\s*(\d+(?:\.\d+)?)", "INR", "", "heme"),
]


# Echo findings (regex → field name).
ECHO_PATTERNS: list[tuple[str, str]] = [
    (r"(?:lv\s*)?ejection\s*fraction\s*(?:is\s*)?[:=]?\s*(\d+(?:\.\d+)?)\s*%", "LVEF"),
    (r"\bef\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%", "LVEF"),
    (r"\blvef\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%", "LVEF"),
    (r"(?:moderate|severe|mild)\s+aortic\s+stenosis", "Aortic stenosis"),
    (r"(?:moderate|severe|mild)\s+mitral\s+regurgitation", "Mitral regurgitation"),
    (r"(?:moderate|severe|mild)\s+tricuspid\s+regurgitation", "Tricuspid regurgitation"),
    (r"pulmonary\s+(?:artery\s+)?(?:systolic\s+)?pressure\s*[:=]?\s*(\d+(?:\.\d+)?)\s*mmhg", "PASP"),
]


# Coronary lesion patterns (vessel + stenosis %).
CATH_PATTERNS: list[tuple[str, str]] = [
    (r"\b(lad|left\s+anterior\s+descending)\b[^.\n]{0,80}?(\d{1,3})\s*%", "LAD"),
    (r"\b(lcx|left\s+circumflex)\b[^.\n]{0,80}?(\d{1,3})\s*%", "LCx"),
    (r"\b(rca|right\s+coronary\s+artery)\b[^.\n]{0,80}?(\d{1,3})\s*%", "RCA"),
    (r"\b(left\s+main|lm)\b[^.\n]{0,80}?(\d{1,3})\s*%", "Left main"),
]
