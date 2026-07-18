// ── Analysis ──────────────────────────────────────────────────────────────────

export type AnalysisMode = 'institution' | 'individual'
export type AnalysisStatus = 'pending' | 'running' | 'done' | 'failed'
export type GapSeverity = 'high' | 'medium' | 'low'

export interface AnalysisRequest {
  mode: AnalysisMode
  text?: string
  csv_content?: string
  target_domain?: string
  target_job_title?: string
}

export interface SkillGap {
  skill_index: number
  skill_name: string
  curriculum_score: number
  job_score: number
  gap: number
  severity: GapSeverity
  recommended_certifications: string[]
  suggested_resources: string[]
}

export interface JobMatch {
  job_id: string
  title: string
  company: string | null
  domain: string | null
  alignment_score: number
  matched_skills: string[]
  missing_skills: string[]
  url: string | null
}

export interface Certification {
  id?: string
  name: string
  provider: string
  domain?: string
  level?: string
  cost_usd?: number
  prep_hours?: number
  validity_years?: number | null
  keywords?: string[]
  url?: string
}

export interface ONetMapping {
  top_domains: string[]
  relevant_occupations: Array<{
    title: string
    domain: string
    growth_rate: number | null
    median_salary: number | null
    top_skills: string[]
  }>
}

export interface AnalysisMetrics {
  overall_score: number
  industry_coverage: number
  curriculum_relevance: number
  mean_skill_gap: number
  ndcg_at_10: number
  high_gap_skills: number
  medium_gap_skills: number
  low_gap_skills: number
}

export interface AnalysisResult {
  analysis_id: string
  mode: AnalysisMode
  status: 'done'
  overall_score: number
  industry_coverage: number
  curriculum_relevance: number
  skill_matrix: number[][]
  skill_gaps: SkillGap[]
  top_matching_jobs: JobMatch[]
  curriculum_labels: string[]
  job_labels: string[]
  metrics: AnalysisMetrics
  certifications: Certification[]
  onet_mapping: ONetMapping
  // Seven-component breakdown (present when MODEL_TYPE=curriculum_gpt)
  component_scores?: Record<string, number>
  component_weights?: Record<string, number>
  uncertainty?: number
  created_at: string
}

export interface AnalysisStatusResponse {
  analysis_id: string
  status: AnalysisStatus
  progress_pct?: number
  error?: string
  result?: AnalysisResult
}

// ── Jobs ──────────────────────────────────────────────────────────────────────

export interface Job {
  id: string
  title: string
  description: string
  company: string | null
  domain: string | null
  skills: string[] | null
  location: string | null
  salary_min: number | null
  salary_max: number | null
  source: string | null
  url: string | null
  created_at: string
}

export interface JobListResponse {
  total: number
  page: number
  page_size: number
  items: Job[]
}

export interface DomainStats {
  domain: string
  job_count: number
  top_skills: string[]
  avg_salary_min: number | null
  avg_salary_max: number | null
}

// ── Courses ───────────────────────────────────────────────────────────────────

export interface Course {
  id: string
  title: string
  description: string
  category: string | null
  skills: string[] | null
  source: string | null
  url: string | null
  created_at: string
}

export interface CourseListResponse {
  total: number
  page: number
  page_size: number
  items: Course[]
}

// ── UI helpers ────────────────────────────────────────────────────────────────

export interface SelectOption {
  value: string
  label: string
}
