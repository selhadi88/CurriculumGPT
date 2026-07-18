import { TrendingUp, Target, BookOpen } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/Card'
import { ScoreRing } from '@/components/ui/Progress'
import type { AnalysisResult } from '@/types'

interface ScoreCardProps {
  result: AnalysisResult
}

export function ScoreCard({ result }: ScoreCardProps) {
  const gapSkills = (result.metrics.high_gap_skills ?? 0) + (result.metrics.medium_gap_skills ?? 0)
  const metrics = [
    {
      label: 'Industry Coverage',
      value: result.industry_coverage,
      icon: Target,
      description: '% of jobs matched by your curriculum',
      color: result.industry_coverage >= 70 ? 'text-emerald-400' : result.industry_coverage >= 40 ? 'text-amber-400' : 'text-red-400',
    },
    {
      label: 'Curriculum Relevance',
      value: result.curriculum_relevance,
      icon: BookOpen,
      description: '% of courses relevant to job market',
      color: result.curriculum_relevance >= 70 ? 'text-emerald-400' : result.curriculum_relevance >= 40 ? 'text-amber-400' : 'text-red-400',
    },
    {
      label: 'Skill Gaps Found',
      value: gapSkills,
      icon: TrendingUp,
      description: `${result.metrics.high_gap_skills ?? 0} high · ${result.metrics.medium_gap_skills ?? 0} medium priority`,
      color: gapSkills === 0 ? 'text-slate-400' : gapSkills <= 5 ? 'text-amber-400' : 'text-red-400',
      isCount: true,
    },
  ]

  const componentScores = result.component_scores
  const componentWeights = result.component_weights ?? {}
  const hasComponents = componentScores && Object.keys(componentScores).length > 0

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {/* Overall score */}
        <Card className="col-span-1 flex flex-col items-center justify-center py-6">
          <CardContent className="flex flex-col items-center gap-3">
            <ScoreRing value={result.overall_score} size={88} />
            <div className="text-center">
              <p className="text-sm font-semibold text-slate-100">Overall Alignment</p>
              <p className="text-xs text-slate-500 mt-0.5">Curriculum × Industry</p>
              {result.uncertainty != null && result.uncertainty > 0 && (
                <p className="text-[11px] text-slate-500 mt-1"
                   title="Estimated by Monte Carlo Dropout (10 passes)">
                  ± {(result.uncertainty * 100).toFixed(1)}% uncertainty
                </p>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Sub-metrics */}
        {metrics.map(({ label, value, icon: Icon, description, color, isCount }) => (
          <Card key={label}>
            <CardContent className="flex flex-col gap-4 py-6">
              <div className="flex items-center justify-between">
                <Icon className="h-4 w-4 text-slate-400" />
                <span className={`text-2xl font-bold tabular-nums ${color}`}>
                  {isCount ? Math.round(value) : `${Math.round(value)}%`}
                </span>
              </div>
              <div>
                <p className="text-sm font-medium text-slate-200">{label}</p>
                <p className="text-xs text-slate-500 mt-0.5">{description}</p>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Seven-component breakdown (only for the curriculum_gpt model) */}
      {hasComponents && (
        <Card>
          <CardContent className="py-5">
            <p className="text-sm font-semibold text-slate-100 mb-1">
              How this score was calculated — the 7 components
            </p>
            <p className="text-xs text-slate-500 mb-4">
              The alignment score blends 7 different ways of comparing your curriculum to the job market.
              Each bar shows that component's score (0–100%). Higher = stronger match on that dimension.
            </p>
            <div className="flex flex-col gap-3">
              {COMPONENT_ORDER.filter((name) => name in componentScores!).map((name) => {
                const score = componentScores![name]
                const meta = COMPONENT_INFO[name]
                const pct = Math.round(score * 100)
                const inactive = meta.inactive
                return (
                  <div key={name} className="flex items-start gap-3">
                    <div className="w-32 shrink-0">
                      <p className="text-xs font-medium text-slate-200">{meta.label}</p>
                      <p className="text-[10px] leading-tight text-slate-500">{meta.desc}</p>
                    </div>
                    <div className="flex-1 pt-0.5">
                      <div className="relative h-2.5 overflow-hidden rounded-full bg-slate-800">
                        <div
                          className={`absolute inset-y-0 left-0 rounded-full ${inactive ? 'bg-slate-600' : 'bg-indigo-500'}`}
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                      {inactive && (
                        <p className="text-[10px] text-amber-500/80 mt-0.5">
                          inactive — needs job dates / course ordering in the data
                        </p>
                      )}
                    </div>
                    <span className="w-10 shrink-0 text-right text-xs tabular-nums text-slate-200">
                      {pct}%
                    </span>
                  </div>
                )
              })}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

// Friendly names + one-line explanations for each of the 7 components.
const COMPONENT_INFO: Record<string, { label: string; desc: string; inactive?: boolean }> = {
  semantic:  { label: 'Semantic Meaning', desc: 'Do they talk about the same things?' },
  topic:     { label: 'Topic Coverage', desc: 'Same subject areas covered?' },
  skill:     { label: 'Skill Match', desc: 'Do specific skills overlap?' },
  depth:     { label: 'Cognitive Depth', desc: 'Right level (recall vs. design)?' },
  recency:   { label: 'Recency', desc: 'Up to date vs. recent job postings?' },
  trend:     { label: 'Industry Trend', desc: 'Aligned with 12-month skill trends?' },
  structure: { label: 'Course Structure', desc: 'Topics in a sensible order?' },
}
const COMPONENT_ORDER = ['semantic', 'topic', 'skill', 'depth', 'recency', 'trend', 'structure']
