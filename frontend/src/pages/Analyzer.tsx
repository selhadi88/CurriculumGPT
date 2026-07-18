import { useEffect, useRef, useState } from 'react'
import { CheckCircle, AlertCircle } from 'lucide-react'
import toast from 'react-hot-toast'
import { CurriculumUploader } from '@/components/curriculum/CurriculumUploader'
import { ScoreCard } from '@/components/analysis/ScoreCard'
import { AlignmentHeatmap } from '@/components/analysis/AlignmentHeatmap'
import { SkillGapChart } from '@/components/analysis/SkillGapChart'
import { RecommendationList } from '@/components/analysis/RecommendationList'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { FullPageSpinner } from '@/components/ui/Spinner'
import { JobCard } from '@/components/jobs/JobCard'
import { submitAnalysis, getAnalysis } from '@/services/api'
import { useAppStore } from '@/store/useAppStore'
import type { AnalysisResult } from '@/types'

const POLL_INTERVAL_MS = 3000
const MAX_POLL_ATTEMPTS = 600  // 30 minutes max (CPU analysis can be slow)

export function Analyzer() {
  const { mode, setCurrentAnalysisId, setCurrentResult, addToHistory } = useAppStore()
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const pollCountRef = useRef(0)

  const clearPoll = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
    pollCountRef.current = 0
  }

  useEffect(() => () => clearPoll(), [])

  const handleSubmit = async (payload: {
    text?: string
    csv_content?: string
    target_domain?: string
  }) => {
    setLoading(true)
    setResult(null)
    setError(null)
    clearPoll()

    try {
      const resp = await submitAnalysis({ mode, ...payload })
      const analysisId = resp.analysis_id
      setCurrentAnalysisId(analysisId)

      pollRef.current = setInterval(async () => {
        pollCountRef.current += 1

        if (pollCountRef.current > MAX_POLL_ATTEMPTS) {
          clearPoll()
          setError('Analysis timed out. The 7-component model runs on CPU and can be slow — try again, or use fewer courses.')
          setLoading(false)
          return
        }

        try {
          const status = await getAnalysis(analysisId)
          if (status.status === 'done' && status.result) {
            clearPoll()
            setResult(status.result)
            setCurrentResult(status.result)
            addToHistory({
              id: analysisId,
              mode,
              score: status.result.overall_score,
              created_at: new Date().toISOString(),
            })
            setLoading(false)
            toast.success('Analysis complete!')
          } else if (status.status === 'failed') {
            clearPoll()
            setError(status.error ?? 'Analysis failed')
            setLoading(false)
            toast.error('Analysis failed')
          }
        } catch (e) {
          clearPoll()
          setError('Polling error — check the backend is running')
          setLoading(false)
        }
      }, POLL_INTERVAL_MS)
    } catch (e) {
      setError('Failed to submit analysis. Is the backend running?')
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h1 className="text-2xl font-bold text-slate-100">Curriculum Analyzer</h1>
        <p className="text-sm text-slate-400 mt-1">
          {mode === 'institution'
            ? 'Upload your department course catalog and benchmark it against the job market.'
            : 'List your courses or skills and see how well they align with job requirements.'}
        </p>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Input panel */}
        <div className="lg:col-span-1">
          <Card>
            <CardHeader>
              <CardTitle>
                {mode === 'institution' ? 'Upload Curriculum' : 'Enter Your Skills'}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <CurriculumUploader mode={mode} onSubmit={handleSubmit} loading={loading} />
            </CardContent>
          </Card>
        </div>

        {/* Results panel */}
        <div className="lg:col-span-2">
          {loading && (
            <div className="flex flex-col items-center justify-center h-64 border border-dashed border-slate-700 rounded-xl">
              <FullPageSpinner label="Running the 7-component analysis on CPU — this takes 1–3 minutes. Please wait…" />
            </div>
          )}

          {error && (
            <div className="flex items-start gap-3 rounded-xl border border-red-900 bg-red-900/20 p-4">
              <AlertCircle className="h-5 w-5 text-red-400 shrink-0 mt-0.5" />
              <div>
                <p className="text-sm font-medium text-red-400">Analysis Error</p>
                <p className="text-xs text-red-400/70 mt-1">{error}</p>
              </div>
            </div>
          )}

          {!loading && !error && !result && (
            <div className="flex flex-col items-center justify-center h-64 border border-dashed border-slate-800 rounded-xl text-slate-600">
              <CheckCircle className="h-8 w-8 mb-3" />
              <p className="text-sm">Submit a curriculum to see alignment results</p>
            </div>
          )}

          {result && (
            <div className="space-y-5">
              <div className="rounded-lg border border-emerald-900/40 bg-emerald-900/10 p-3">
                <p className="text-xs font-medium text-emerald-400 flex items-center gap-1.5">
                  <CheckCircle className="h-3.5 w-3.5" />
                  Analysis complete
                </p>
                <p className="text-xs text-slate-400 mt-1.5 leading-relaxed">
                  We compared what you entered against <b>{result.job_labels?.length ?? 20} real job postings</b> from
                  the database. The big number on the left is the <b>overall alignment</b> (0–100%): how well your
                  skills/courses match what these jobs require. Below it are the 7 components that make up that score,
                  then your skill gaps, recommended certifications, and the best-matching jobs.
                </p>
              </div>
              <ScoreCard result={result} />
            </div>
          )}
        </div>
      </div>

      {/* Full results below */}
      {result && (
        <div className="space-y-6">
          <div>
            <h2 className="text-base font-semibold text-slate-200">Detailed breakdown</h2>
            <p className="text-xs text-slate-500 mt-0.5">
              <b>Heatmap:</b> each cell = how strongly one of your items matches one job (green = strong, red = weak).
              &nbsp;<b>Skill gaps:</b> skills the jobs want more than your curriculum provides.
            </p>
          </div>
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <AlignmentHeatmap matrix={result.skill_matrix} />
            <SkillGapChart gaps={result.skill_gaps} />
          </div>

          <RecommendationList skillGaps={result.skill_gaps} certifications={result.certifications} />

          {result.top_matching_jobs.length > 0 && (
            <div>
              <h2 className="text-base font-semibold text-slate-200 mb-4">Top Matching Jobs</h2>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {result.top_matching_jobs.slice(0, 6).map((match) => (
                  <JobCard
                    key={match.job_id}
                    job={{
                      id: match.job_id,
                      title: match.title,
                      description: '',
                      company: match.company,
                      domain: match.domain,
                      skills: match.matched_skills,
                      location: null,
                      salary_min: null,
                      salary_max: null,
                      source: null,
                      url: match.url,
                      created_at: '',
                    }}
                    alignmentScore={match.alignment_score}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
