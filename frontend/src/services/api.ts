import axios from 'axios'
import type {
  AnalysisRequest,
  AnalysisStatusResponse,
  CourseListResponse,
  DomainStats,
  JobListResponse,
} from '@/types'

// Render's `fromService` hands us a bare hostname (no scheme). Normalize so a
// value like "curriculumgpt-backend.onrender.com" becomes a usable base URL.
const rawApiUrl = (import.meta.env.VITE_API_URL as string | undefined)?.trim()
const apiBaseUrl = !rawApiUrl
  ? 'http://localhost:7000'
  : /^https?:\/\//.test(rawApiUrl)
    ? rawApiUrl
    : `https://${rawApiUrl}`

const api = axios.create({
  baseURL: apiBaseUrl,
  headers: { 'Content-Type': 'application/json' },
  timeout: 30_000,
})

// ── Analysis ──────────────────────────────────────────────────────────────────

export const submitAnalysis = (req: AnalysisRequest): Promise<AnalysisStatusResponse> =>
  api.post<AnalysisStatusResponse>('/api/v1/analyze', req).then((r) => r.data)

export const getAnalysis = (id: string): Promise<AnalysisStatusResponse> =>
  api.get<AnalysisStatusResponse>(`/api/v1/analyze/${id}`).then((r) => r.data)

// ── Jobs ──────────────────────────────────────────────────────────────────────

export interface JobFilters {
  page?: number
  page_size?: number
  domain?: string
  keyword?: string
}

export const fetchJobs = (filters: JobFilters = {}): Promise<JobListResponse> =>
  api.get<JobListResponse>('/api/v1/jobs', { params: filters }).then((r) => r.data)

export const searchJobs = (q: string, page = 1): Promise<JobListResponse> =>
  api.get<JobListResponse>('/api/v1/jobs/search', { params: { q, page } }).then((r) => r.data)

export const fetchDomains = (): Promise<DomainStats[]> =>
  api.get<DomainStats[]>('/api/v1/domains').then((r) => r.data)

// ── Courses ───────────────────────────────────────────────────────────────────

export interface CourseFilters {
  page?: number
  page_size?: number
  category?: string
  keyword?: string
}

export const fetchCourses = (filters: CourseFilters = {}): Promise<CourseListResponse> =>
  api.get<CourseListResponse>('/api/v1/courses', { params: filters }).then((r) => r.data)

// ── Health ────────────────────────────────────────────────────────────────────

export const fetchHealth = (): Promise<{ status: string; model_type: string }> =>
  api.get('/api/v1/health').then((r) => r.data)

export default api
