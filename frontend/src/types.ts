export type Bbox = [number, number, number, number] // [x0, y0, x1, y1], normalized 0..1, top-left origin

export interface Span {
  span_id: string
  doc_id: string
  page: number // 1-based
  bbox: Bbox
  text: string
  source: 'text_layer' | 'ocr'
}

export interface Document {
  doc_id: string
  name: string
  page_count: number
  source: 'text_layer' | 'ocr' | 'mixed'
}

export interface IngestResponse {
  session_id: string
  documents: Document[]
  spans: Span[]
}

// ---------- One-pager ----------

export type SectionKey =
  | 'hpi'
  | 'family_history'
  | 'social_history'
  | 'past_medical_history'
  | 'past_surgical_history'
  | 'medications'
  | 'objective'
  | 'labs'
  | 'cardiology_imaging_procedures'

export interface SectionItem {
  text: string
  importance: number
  evidence_span_ids: string[]
}

export interface Section {
  key: SectionKey
  title: string
  items: SectionItem[]
}

export interface GuidelineCitation {
  chunk_id: string
  document: string
  page: number
  snippet: string
  score: number
}

export interface Problem {
  name: string
  summary: string
  importance: number
  evidence_span_ids: string[]
  plan: string | null
  plan_citations: GuidelineCitation[]
}

export interface OnePager {
  sections: Section[]
  problems: Problem[]
  warnings: string[]
  disclaimer: string
}
