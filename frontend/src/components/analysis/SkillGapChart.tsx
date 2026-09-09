import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend,
} from 'recharts'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Badge, severityBadge } from '@/components/ui/Badge'
import type { SkillGap } from '@/types'

interface SkillGapChartProps {
  gaps: SkillGap[]
  maxItems?: number
}

export function SkillGapChart({ gaps, maxItems = 12 }: SkillGapChartProps) {
  const topGaps = gaps.slice(0, maxItems)

  const chartData = topGaps.map((g) => ({
    name: g.skill_name.length > 18 ? g.skill_name.slice(0, 18) + '…' : g.skill_name,
    fullName: g.skill_name,
    curriculum: Math.round(g.curriculum_score * 100),
    job: Math.round(g.job_score * 100),
    gap: Math.round(g.gap * 100),
    severity: g.severity,
  }))

  return (
    <Card>
      <CardHeader>
        <CardTitle>Top Skill Gaps</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={chartData} layout="vertical" barGap={2} margin={{ left: 8, right: 16 }}>
            <XAxis type="number" domain={[0, 100]} tick={{ fill: '#94a3b8', fontSize: 11 }} tickFormatter={(v) => `${v}%`} />
            <YAxis type="category" dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} width={130} />
            <Tooltip
              contentStyle={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8 }}
              labelStyle={{ color: '#f1f5f9' }}
              formatter={(value: number, name: string) => [`${value}%`, name === 'curriculum' ? 'Curriculum Coverage' : 'Job Requirement']}
            />
            <Legend
              wrapperStyle={{ color: '#94a3b8', fontSize: 12 }}
              formatter={(v) => v === 'curriculum' ? 'Curriculum' : 'Job Market'}
            />
            <Bar dataKey="curriculum" fill="#6366f1" radius={[0, 3, 3, 0]} maxBarSize={14} />
            <Bar dataKey="job" fill="#475569" radius={[0, 3, 3, 0]} maxBarSize={14} />
          </BarChart>
        </ResponsiveContainer>

        {/* Severity legend */}
        <div className="mt-4 space-y-2">
          {topGaps.slice(0, 5).map((gap) => (
            <div key={gap.skill_index} className="flex items-center justify-between text-sm">
              <span className="text-slate-300 truncate flex-1 mr-3">{gap.skill_name}</span>
              <Badge variant={severityBadge(gap.severity)}>{gap.severity}</Badge>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
