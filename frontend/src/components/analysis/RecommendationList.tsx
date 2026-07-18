import { BookOpen, Award, ExternalLink } from 'lucide-react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Badge, severityBadge } from '@/components/ui/Badge'
import type { SkillGap, Certification } from '@/types'

interface RecommendationListProps {
  skillGaps: SkillGap[]
  certifications: Certification[]
}

export function RecommendationList({ skillGaps, certifications }: RecommendationListProps) {
  const topGaps = skillGaps.filter((g) => g.severity === 'high' || g.severity === 'medium').slice(0, 6)

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
      {/* Skill recommendations */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BookOpen className="h-4 w-4 text-brand-400" />
            Priority Learning Areas
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {topGaps.length === 0 && (
            <p className="text-sm text-slate-500">No significant skill gaps found.</p>
          )}
          {topGaps.map((gap) => (
            <div key={gap.skill_index} className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-slate-200">{gap.skill_name}</span>
                <Badge variant={severityBadge(gap.severity)}>{gap.severity} gap</Badge>
              </div>

              {/* Gap bar */}
              <div className="flex h-2 overflow-hidden rounded-full bg-slate-800">
                <div
                  className="bg-brand-500"
                  style={{ width: `${gap.curriculum_score * 100}%` }}
                />
                <div
                  className="bg-red-500/60"
                  style={{ width: `${gap.gap * 100}%` }}
                />
              </div>
              <div className="flex justify-between text-xs text-slate-500">
                <span>You: {Math.round(gap.curriculum_score * 100)}%</span>
                <span>Required: {Math.round(gap.job_score * 100)}%</span>
              </div>

              {gap.suggested_resources.length > 0 && (
                <div className="flex flex-wrap gap-1 mt-1">
                  {gap.suggested_resources.slice(0, 2).map((r, i) => (
                    <span key={i} className="text-xs text-brand-400 bg-brand-900/30 rounded px-2 py-0.5">
                      {r}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
        </CardContent>
      </Card>

      {/* Certifications */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Award className="h-4 w-4 text-amber-400" />
            Recommended Certifications
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {certifications.length === 0 && (
            <p className="text-sm text-slate-500">No certifications to suggest.</p>
          )}
          {certifications.slice(0, 5).map((cert, i) => (
            <div
              key={i}
              className="flex items-start justify-between rounded-lg border border-slate-800 p-3 hover:border-slate-700 transition-colors"
            >
              <div className="flex-1 min-w-0 mr-3">
                <p className="text-sm font-medium text-slate-200 truncate">{cert.name}</p>
                <p className="text-xs text-slate-500 mt-0.5">{cert.provider}</p>
                <div className="flex items-center gap-3 mt-1.5 text-xs text-slate-500">
                  {cert.cost_usd != null && (
                    <span>${cert.cost_usd}</span>
                  )}
                  {cert.prep_hours && (
                    <span>~{cert.prep_hours}h prep</span>
                  )}
                  {cert.level && (
                    <Badge variant="default" className="text-xs">{cert.level}</Badge>
                  )}
                </div>
              </div>
              {cert.url && (
                <a
                  href={cert.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-slate-500 hover:text-brand-400 transition-colors mt-0.5 shrink-0"
                >
                  <ExternalLink className="h-3.5 w-3.5" />
                </a>
              )}
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  )
}
