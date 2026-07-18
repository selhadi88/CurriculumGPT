import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ChevronLeft, ChevronRight, TrendingUp, Link as LinkIcon } from 'lucide-react'
import { Link } from 'react-router-dom'
import { JobCard } from '@/components/jobs/JobCard'
import { JobFilters } from '@/components/jobs/JobFilters'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { FullPageSpinner } from '@/components/ui/Spinner'
import { fetchJobs } from '@/services/api'
import { useAppStore } from '@/store/useAppStore'

export function JobExplorer() {
  const { currentResult } = useAppStore()
  const [keyword, setKeyword] = useState('')
  const [domain, setDomain] = useState('')
  const [page, setPage] = useState(1)
  const pageSize = 12

  const { data, isLoading } = useQuery({
    queryKey: ['jobs', { keyword, domain, page, page_size: pageSize }],
    queryFn: () => fetchJobs({ keyword: keyword || undefined, domain: domain || undefined, page, page_size: pageSize }),
    placeholderData: (prev) => prev,
    staleTime: 60_000,
  })

  const totalPages = data ? Math.ceil(data.total / pageSize) : 0

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h1 className="text-2xl font-bold text-slate-100">Job Market</h1>
        <p className="text-sm text-slate-400 mt-1">
          {data?.total.toLocaleString() ?? '…'} job postings indexed from the market.
        </p>
      </div>

      {/* Top matches from last analysis — only shown when a result exists */}
      {currentResult && currentResult.top_matching_jobs.length > 0 && (
        <div>
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-base font-semibold text-slate-200 flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-emerald-400" />
              Top Matches from Your Last Analysis
            </h2>
            <Link to="/skills">
              <Button variant="ghost" size="sm">View full report</Button>
            </Link>
          </div>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {currentResult.top_matching_jobs.slice(0, 6).map((match) => (
              <Card key={match.job_id} className="border-emerald-900/40">
                <CardContent className="py-4 space-y-3">
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-semibold text-slate-100 truncate">{match.title}</p>
                      {match.company && (
                        <p className="text-xs text-slate-500 mt-0.5 truncate">{match.company}</p>
                      )}
                    </div>
                    <span className="text-emerald-400 font-bold text-sm tabular-nums shrink-0">
                      {Math.round(match.alignment_score * 100)}%
                    </span>
                  </div>

                  {match.matched_skills.length > 0 && (
                    <div>
                      <p className="text-[10px] text-slate-500 mb-1">Skills you cover</p>
                      <div className="flex flex-wrap gap-1">
                        {match.matched_skills.slice(0, 3).map((s) => (
                          <Badge key={s} variant="success" className="text-[10px]">{s}</Badge>
                        ))}
                      </div>
                    </div>
                  )}

                  {match.missing_skills.length > 0 && (
                    <div>
                      <p className="text-[10px] text-slate-500 mb-1">Skills to develop</p>
                      <div className="flex flex-wrap gap-1">
                        {match.missing_skills.slice(0, 3).map((s) => (
                          <Badge key={s} variant="danger" className="text-[10px]">{s}</Badge>
                        ))}
                      </div>
                    </div>
                  )}

                  {match.url && (
                    <a
                      href={match.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-1 text-xs text-brand-400 hover:text-brand-300"
                    >
                      <LinkIcon className="h-3 w-3" /> View posting
                    </a>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      )}

      {!currentResult && (
        <div className="rounded-xl border border-dashed border-slate-700 p-6 text-center text-sm text-slate-500">
          Run an analysis to see which jobs match your curriculum.{' '}
          <Link to="/analyze" className="text-brand-400 hover:underline">Go to Analyzer →</Link>
        </div>
      )}

      {/* Full job browser */}
      <div>
        <h2 className="text-base font-semibold text-slate-200 mb-3">Browse All Jobs</h2>
        <JobFilters
          keyword={keyword}
          domain={domain}
          onKeywordChange={(v) => { setKeyword(v); setPage(1) }}
          onDomainChange={(v) => { setDomain(v); setPage(1) }}
        />
      </div>

      {isLoading ? (
        <FullPageSpinner label="Loading jobs…" />
      ) : data && data.items.length > 0 ? (
        <>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {data.items.map((job) => (
              <JobCard key={job.id} job={job} />
            ))}
          </div>

          {totalPages > 1 && (
            <div className="flex items-center justify-center gap-3">
              <Button variant="secondary" size="sm" onClick={() => setPage((p) => Math.max(1, p - 1))} disabled={page === 1}>
                <ChevronLeft className="h-4 w-4" />
              </Button>
              <span className="text-sm text-slate-400">Page {page} of {totalPages}</span>
              <Button variant="secondary" size="sm" onClick={() => setPage((p) => Math.min(totalPages, p + 1))} disabled={page === totalPages}>
                <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
          )}
        </>
      ) : (
        <div className="flex flex-col items-center justify-center h-48 text-slate-600">
          <p className="text-sm">No jobs found. Try different filters.</p>
        </div>
      )}
    </div>
  )
}
