import { Link } from 'react-router-dom'
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer, Tooltip,
} from 'recharts'
import { AlertTriangle } from 'lucide-react'
import { AlignmentHeatmap } from '@/components/analysis/AlignmentHeatmap'
import { SkillGapChart } from '@/components/analysis/SkillGapChart'
import { RecommendationList } from '@/components/analysis/RecommendationList'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Badge, severityBadge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import { useAppStore } from '@/store/useAppStore'

export function SkillGaps() {
  const { currentResult } = useAppStore()

  if (!currentResult) {
    return (
      <div className="flex flex-col items-center justify-center h-72 gap-4 text-center animate-fade-in">
        <AlertTriangle className="h-10 w-10 text-amber-500" />
        <div>
          <p className="text-lg font-semibold text-slate-200">No analysis yet</p>
          <p className="text-sm text-slate-500 mt-1">Run an analysis first to see your skill gaps.</p>
        </div>
        <Link to="/analyze">
          <Button>Go to Analyzer</Button>
        </Link>
      </div>
    )
  }

  const { skill_gaps, skill_matrix, certifications, curriculum_labels, job_labels } = currentResult

  // Build radar data from the top skills the JOBS require most (clearest story).
  const radarData = [...skill_gaps]
    .sort((a, b) => b.job_score - a.job_score)
    .slice(0, 8)
    .map((g) => ({
      skill: g.skill_name.length > 14 ? g.skill_name.slice(0, 14) + '…' : g.skill_name,
      You: Math.round(g.curriculum_score * 100),
      Required: Math.round(g.job_score * 100),
    }))

  // Group gaps by severity
  const high = skill_gaps.filter((g) => g.severity === 'high')
  const medium = skill_gaps.filter((g) => g.severity === 'medium')
  const low = skill_gaps.filter((g) => g.severity === 'low')

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h1 className="text-2xl font-bold text-slate-100">Skill Gap Analysis</h1>
        <p className="text-sm text-slate-400 mt-1">
          {high.length} high, {medium.length} medium, {low.length} low priority gaps identified.
        </p>
      </div>

      {/* Radar + heatmap row */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Skill Radar — You vs. Required</CardTitle>
            <p className="text-xs text-slate-500 mt-1">
              Purple = your curriculum's coverage. Grey = what the jobs require. Gaps where grey extends past purple.
            </p>
          </CardHeader>
          <CardContent>
            {radarData.length < 3 ? (
              <div className="flex h-[280px] items-center justify-center text-center px-6">
                <p className="text-sm text-slate-500">
                  Not enough detected skills to draw the radar. Run an analysis with a richer
                  curriculum (more courses or skill keywords).
                </p>
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={280}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="#1e293b" />
                  <PolarAngleAxis dataKey="skill" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                  <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fill: '#475569', fontSize: 9 }} />
                  <Tooltip
                    contentStyle={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8 }}
                  />
                  <Radar name="You" dataKey="You" stroke="#6366f1" fill="#6366f1" fillOpacity={0.3} />
                  <Radar name="Required" dataKey="Required" stroke="#64748b" fill="#64748b" fillOpacity={0.15} />
                </RadarChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>

        <AlignmentHeatmap
          matrix={skill_matrix}
          rowLabels={curriculum_labels}
          colLabels={job_labels}
        />
      </div>

      {/* Gap bar chart */}
      {skill_gaps.length === 0 ? (
        <div className="rounded-xl border border-amber-800/40 bg-amber-900/10 p-4 text-sm text-amber-400">
          No skill gaps detected — the curriculum and job descriptions may not contain specific skill keywords.
          Try submitting a curriculum with richer course descriptions.
        </div>
      ) : (
        <SkillGapChart gaps={skill_gaps} maxItems={15} />
      )}

      {/* Prioritized gap list */}
      <div>
        <h2 className="text-base font-semibold text-slate-200 mb-4">Gap Priority List</h2>
        <div className="space-y-2">
          {skill_gaps.slice(0, 15).map((gap) => (
            <div
              key={gap.skill_index}
              className="flex items-center gap-4 rounded-lg border border-slate-800 bg-slate-900 px-4 py-3"
            >
              <div className="w-40 shrink-0">
                <p className="text-sm font-medium text-slate-200 truncate">{gap.skill_name}</p>
              </div>
              <div className="flex-1">
                <div className="flex h-2 overflow-hidden rounded-full bg-slate-800">
                  <div
                    className="bg-brand-500 transition-all"
                    style={{ width: `${gap.curriculum_score * 100}%` }}
                  />
                  <div
                    className="bg-red-500/60 transition-all"
                    style={{ width: `${gap.gap * 100}%` }}
                  />
                </div>
              </div>
              <div className="w-16 text-right text-xs text-slate-500 tabular-nums">
                +{Math.round(gap.gap * 100)}%
              </div>
              <Badge variant={severityBadge(gap.severity)}>{gap.severity}</Badge>
            </div>
          ))}
        </div>
      </div>

      <RecommendationList skillGaps={skill_gaps} certifications={certifications} />
    </div>
  )
}
