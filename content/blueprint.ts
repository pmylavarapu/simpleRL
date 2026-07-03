// ASCeXAM (ASE Board) Content Outline — 1/2025
// Source: National Board of Echocardiography (www.echoboards.org)
// This is the authoritative structure for the knowledge base and for card grouping.

export type Subtopic = {
  code: string;   // e.g. "II.A"
  slug: string;   // e.g. "aortic-valve-aorta-and-subvalvular-outflow-tract"
  title: string;
};

export type Section = {
  code: string;   // e.g. "II"
  slug: string;   // e.g. "valvular-heart-disease"
  title: string;
  subtopics: Subtopic[];
};

const s = (title: string) =>
  title
    .toLowerCase()
    .replace(/[^\w\s-]/g, "")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "");

const mk = (
  sectionCode: string,
  sectionTitle: string,
  items: [string, string][],
): Section => ({
  code: sectionCode,
  slug: s(sectionTitle),
  title: sectionTitle,
  subtopics: items.map(([letter, title]) => ({
    code: `${sectionCode}.${letter}`,
    slug: s(title),
    title,
  })),
});

export const BLUEPRINT: Section[] = [
  mk("I", "Physical Principles, Instrumentation, Examination Principles", [
    ["A", "Routine Doppler Examination"],
    ["B", "Transesophageal Echocardiography, Intraoperative Echocardiography, and Catheter-Based Echocardiography (ICE)"],
    ["C", "Physical Principles of Ultrasound"],
    ["D", "Cross-Sectional Echocardiographic Examination"],
    ["E", "Principles of Doppler Flow Measurement"],
    ["F", "Cross-Sectional Scanning: Technical Principles and Instrumentation"],
    ["G", "Standard Plane Positions – Standard Imaging Planes"],
    ["H", "Doppler Instrumentation"],
    ["I", "Principles of Flow"],
    ["J", "Principles of Color Flow Mapping"],
    ["K", "M-Mode Echocardiography"],
    ["L", "Digital Image Processing"],
    ["M", "Doppler Signal Processing, Tissue Characterization"],
    ["N", "Three-Dimensional Echocardiography"],
    ["O", "Place (Role) of Echocardiography"],
    ["P", "Hand-Held Echo"],
    ["Q", "Laboratory Accreditation"],
  ]),
  mk("II", "Valvular Heart Disease", [
    ["A", "Aortic Valve, Aorta, and Subvalvular Outflow Tract"],
    ["B", "Mitral Valve"],
    ["C", "Echo-Doppler Assessment of Prosthetic Heart Valves"],
    ["D", "Echocardiographic Findings in Infective Endocarditis"],
    ["E", "Fluid Dynamics of Regurgitant Jets"],
    ["F", "Tricuspid Valve"],
    ["G", "Pulmonic Valve"],
    ["H", "Pulmonary Hypertension"],
  ]),
  mk("III", "Chamber Size and Function", [
    ["A", "Coronary Artery Disease, Stress Echocardiography"],
    ["B", "General Considerations, Assessment of Chamber Size and Function"],
    ["C", "Echocardiographic Assessment of the Cardiomyopathies"],
    ["D", "Diastolic Function"],
    ["E", "Left Atrium, Pulmonary Veins, and Coronary Sinus"],
    ["F", "Right Ventricle"],
    ["G", "Right Atrium"],
    ["H", "Interatrial and Interventricular Septum"],
    ["I", "Inferior and Superior Vena Cava"],
    ["J", "Doppler Estimation of Volumetric Flow"],
    ["K", "Coronary Arteries"],
  ]),
  mk("IV", "Congenital Heart Disease", [
    ["A", "Complex Congenital Heart Disease"],
    ["B", "Aortic Valve, Aorta, and Subvalvular Outflow Tract"],
    ["C", "Tricuspid Valve Anomalies"],
    ["D", "Mitral Valve"],
    ["E", "Doppler Estimation of Volumetric Flow"],
    ["F", "Pulmonic Valve Anomalies"],
    ["G", "Coronary Arteries Anomalies"],
    ["H", "Fetal Echocardiography"],
    ["I", "Terminology and Anatomic and Physiologic Basis of CHD"],
    ["J", "Principles of Medical and Surgical Management"],
    ["K", "Echo Evaluation of Post-Op Congenital Heart Disease"],
  ]),
  mk("V", "Cardiac Masses, Pericardial Disease, Contrast and New Applications", [
    ["A", "Pericardial Disease"],
    ["B", "Cardiac Tumors and Masses"],
    ["C", "Contrast Echocardiography"],
    ["D", "Assessment of Myocardial Perfusion with Contrast"],
    ["E", "Echocardiography in Disorders of Cardiac Rhythm and Conduction"],
    ["F", "Echocardiography in Cardiac Transplantation"],
  ]),
  mk("VI", "Miscellaneous Topics (Role of Echo)", [
    ["A", "Heart Failure"],
    ["B", "Cardiac Sources of Embolism (PFO, ASA, SEC, Aortic Atheroma, etc)"],
    ["C", "Pulmonary Heart Disease"],
    ["D", "Systemic Diseases"],
    ["E", "Atrial Fibrillation"],
    ["F", "Trauma"],
    ["G", "Athlete's Heart"],
    ["H", "Aging Changes"],
    ["I", "Pregnancy"],
    ["J", "Interventional Echocardiography"],
    ["K", "Digital Lab"],
    ["L", "Quality in the Echo Lab"],
  ]),
];

export function findSection(sectionSlug: string): Section | undefined {
  return BLUEPRINT.find((sec) => sec.slug === sectionSlug);
}

export function findSubtopic(
  sectionSlug: string,
  topicSlug: string,
): { section: Section; subtopic: Subtopic } | undefined {
  const section = findSection(sectionSlug);
  if (!section) return undefined;
  const subtopic = section.subtopics.find((st) => st.slug === topicSlug);
  if (!subtopic) return undefined;
  return { section, subtopic };
}

export function findByCode(code: string): { section: Section; subtopic: Subtopic } | undefined {
  for (const section of BLUEPRINT) {
    for (const subtopic of section.subtopics) {
      if (subtopic.code === code) return { section, subtopic };
    }
  }
  return undefined;
}

export const ALL_SUBTOPICS: Subtopic[] = BLUEPRINT.flatMap((s) => s.subtopics);
