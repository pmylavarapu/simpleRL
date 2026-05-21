// Mirror of backend/app/schemas.py — keep in sync.

export type SourceRef = { pdf_id: string; page: number; snippet: string };

export type Claim = { text: string; sources: SourceRef[] };

export type LabValue = { value: string; unit: string; date: string; source: SourceRef };
export type LabPanel = { name: string; latest: LabValue; trend: LabValue[] };
export type CardiologyStudy = { study_type: string; date: string; key_findings: Claim[] };

export type AssessmentProblem = { problem: string; paragraph: Claim[] };

export type GuidelineCitation = {
  guideline: string;
  year: number;
  section: string;
  cor: string;
  loe: string;
  url?: string | null;
  doi?: string | null;
};

export type PlanItem = {
  problem: string;
  recommendation: string;
  rationale_for_patient: string;
  priority: number;
  citation: GuidelineCitation;
  precondition_evidence: SourceRef[];
};

export type PatientHeader = {
  name?: string | null;
  age?: number | null;
  sex?: string | null;
  mrn?: string | null;
  dob?: string | null;
};

export type OnePageSummary = {
  patient: PatientHeader;
  hpi: Claim[];
  fam_hx: Claim[];
  soc_hx: Claim[];
  pmh: Claim[];
  psh: Claim[];
  meds: Claim[];
  objective: { labs: LabPanel[]; cardiology: CardiologyStudy[] };
  assessment: AssessmentProblem[];
  plan: PlanItem[];
};

export type PdfFile = { pdf_id: string; filename: string; pages: number };
export type CaseManifest = { case_id: string; pdfs: PdfFile[] };

export type ActiveSource = SourceRef | null;
