import { useQuery } from '@tanstack/react-query'
import { TrendingUp, Briefcase, BookOpen, Clock } from 'lucide-react'
import { Link } from 'react-router-dom'
import { Card, CardContent } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import { ScoreRing } from '@/components/ui/Progress'
import { FullPageSpinner } from '@/components/ui/Spinner'
import { fetchDomains } from '@/services/api'
import { useAppStore } from '@/store/useAppStore'

export function Dashboard() {
  const { mode, history, currentResult } = useAppStore()

  const { data: domains, isLoading } = useQuery({
    queryKey: ['domains'],
    queryFn: fetchDomains,
    staleTime: 5 * 60 * 1000,
  })

  if (isLoading) return <FullPageSpinner label="Loading dashboard…" />

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Hero */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-100">
            {mode === 'institution' ? 'Institution Dashboard' : 'Career Dashboard'}
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            {mode === 'institution'
              ? 'Monitor curriculum alignment across departments and job market trends.'
              : 'Track your skill alignment with the job market and identify growth opportunities.'}
          </p>
        </div>
        <Link to="/analyze">
          <Button size="lg">
            <TrendingUp className="h-4 w-4" />
            Run Analysis
          </Button>
        </Link>
      </div>

      {/* Last result quick-view */}
      {currentResult && (
        <Card glass>
          <CardContent className="flex items-center gap-6 py-5">
            <ScoreRing value={currentResult.overall_score} size={64} />
            <div className="flex-1">
              <p className="text-sm font-semibold text-slate-100">Latest Analysis</p>
              <p className="text-xs text-slate-400 mt-0.5">
                {currentResult.skill_gaps.filter(g => g.severity === 'high').length} high-priority gaps detected
              </p>
            </div>
            <div className="flex gap-4 text-center">
              <Metric value={`${Math.round(currentResult.industry_coverage)}%`} label="Coverage" />
              <Metric value={`${Math.round(currentResult.curriculum_relevance)}%`} label="Relevance" />
              <Metric value={currentResult.metrics.high_gap_skills?.toString()} label="High Gaps" />
            </div>
            <Link to="/skills">
              <Button variant="secondary" size="sm">View Details</Button>
            </Link>
          </CardContent>
        </Card>
      )}

      {/* Stats row */}
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <StatCard
          icon={<Briefcase className="h-4 w-4 text-brand-400" />}
          label="Job Domains"
          value={domains?.length ?? 0}
        />
        <StatCard
          icon={<BookOpen className="h-4 w-4 text-emerald-400" />}
          label="Analyses Run"
          value={history.length}
        />
        <StatCard
          icon={<TrendingUp className="h-4 w-4 text-amber-400" />}
          label="Avg Score"
          value={history.length > 0
            ? `${Math.round(history.reduce((s, h) => s + h.score, 0) / history.length)}%`
            : '—'}
        />
        <StatCard
          icon={<Clock className="h-4 w-4 text-slate-400" />}
          label="Last Analysis"
          value={history[0]
            ? new Date(history[0].created_at).toLocaleDateString()
            : '—'}
        />
      </div>

      {/* Domain overview */}
      {domains && domains.length > 0 && (
        <div>
          <h2 className="text-base font-semibold text-slate-200 mb-4">Job Market Overview</h2>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {domains.slice(0, 6).map((d) => (
              <Card key={d.domain}>
                <CardContent className="py-4">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="text-sm font-medium text-slate-200 capitalize">
                      {d.domain.replace(/_/g, ' ')}
                    </h3>
                    <Badge variant="info">{d.job_count} jobs</Badge>
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {d.top_skills.slice(0, 4).map((skill) => (
                      <Badge key={skill} variant="default" className="text-[10px]">{skill}</Badge>
                    ))}
                  </div>
                  {d.avg_salary_min && (
                    <p className="text-xs text-emerald-400 mt-2">
                      ${Math.round(d.avg_salary_min / 1000)}k – ${Math.round((d.avg_salary_max ?? 0) / 1000)}k avg
                    </p>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      )}

      {/* Recent analyses */}
      {history.length > 0 && (
        <div>
          <h2 className="text-base font-semibold text-slate-200 mb-4">Recent Analyses</h2>
          <div className="space-y-2">
            {history.slice(0, 5).map((h) => (
              <div
                key={h.id}
                className="flex items-center justify-between rounded-lg border border-slate-800 bg-slate-900 px-4 py-3"
              >
                <div className="flex items-center gap-3">
                  <ScoreRing value={h.score} size={36} />
                  <div>
                    <p className="text-sm text-slate-200 capitalize">{h.mode} analysis</p>
                    <p className="text-xs text-slate-500">{new Date(h.created_at).toLocaleString()}</p>
                  </div>
                </div>
                <Link to="/skills">
                  <Button variant="ghost" size="sm">View</Button>
                </Link>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function Metric({ value, label }: { value: string | number; label: string }) {
  return (
    <div>
      <p className="text-lg font-bold text-slate-100">{value}</p>
      <p className="text-xs text-slate-500">{label}</p>
    </div>
  )
}

function StatCard({ icon, label, value }: { icon: React.ReactNode; label: string; value: string | number }) {
  return (
    <Card>
      <CardContent className="flex items-center gap-3 py-4">
        {icon}
        <div>
          <p className="text-lg font-bold text-slate-100">{value}</p>
          <p className="text-xs text-slate-500">{label}</p>
        </div>
      </CardContent>
    </Card>
  )
}
