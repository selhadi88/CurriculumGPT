import { Download, FileJson, FileText, AlertTriangle } from 'lucide-react'
import { Link } from 'react-router-dom'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { ScoreCard } from '@/components/analysis/ScoreCard'
import { useAppStore } from '@/store/useAppStore'

function downloadJSON(data: object, filename: string) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

function downloadCSV(rows: string[][], filename: string) {
  const csv = rows.map((r) => r.map((v) => `"${String(v).replace(/"/g, '""')}"`).join(',')).join('\n')
  const blob = new Blob([csv], { type: 'text/csv' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

export function Reports() {
  const { currentResult } = useAppStore()

  const handleExportJSON = () => {
    if (!currentResult) return
    downloadJSON(currentResult, `gurriculum_report_${currentResult.analysis_id.slice(0, 8)}.json`)
  }

  const handleExportCSV = () => {
    if (!currentResult) return
    const rows: string[][] = [
      ['Skill Name', 'Curriculum Score (%)', 'Job Requirement (%)', 'Gap (%)', 'Severity'],
      ...currentResult.skill_gaps.map((g) => [
        g.skill_name,
        String(Math.round(g.curriculum_score * 100)),
        String(Math.round(g.job_score * 100)),
        String(Math.round(g.gap * 100)),
        g.severity,
      ]),
    ]
    downloadCSV(rows, `skill_gaps_${currentResult.analysis_id.slice(0, 8)}.csv`)
  }

  if (!currentResult) {
    return (
      <div className="flex flex-col items-center justify-center h-72 gap-4 text-center animate-fade-in">
        <AlertTriangle className="h-10 w-10 text-amber-500" />
        <div>
          <p className="text-lg font-semibold text-slate-200">No report available</p>
          <p className="text-sm text-slate-500 mt-1">Run an analysis to generate a report.</p>
        </div>
        <Link to="/analyze">
          <Button>Go to Analyzer</Button>
        </Link>
      </div>
    )
  }

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-100">Analysis Report</h1>
          <p className="text-sm text-slate-400 mt-1">
            {currentResult.mode === 'institution' ? 'Institution' : 'Individual'} analysis ·{' '}
            {new Date(currentResult.created_at).toLocaleString()}
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="secondary" size="sm" onClick={handleExportCSV}>
            <Download className="h-3.5 w-3.5" />
            Export CSV
          </Button>
          <Button variant="secondary" size="sm" onClick={handleExportJSON}>
            <FileJson className="h-3.5 w-3.5" />
            Export JSON
          </Button>
        </div>
      </div>

      {/* Summary scores */}
      <ScoreCard result={currentResult} />

      {/* Metrics table */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FileText className="h-4 w-4 text-brand-400" />
            Full Metrics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <table className="w-full text-sm">
            <tbody className="divide-y divide-slate-800">
              {Object.entries(currentResult.metrics).map(([key, value]) => (
                <tr key={key}>
                  <td className="py-2 text-slate-400 capitalize">{key.replace(/_/g, ' ')}</td>
                  <td className="py-2 text-right text-slate-200 tabular-nums font-medium">
                    {typeof value === 'number' ? value.toLocaleString() : String(value)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </CardContent>
      </Card>

      {/* O*NET occupations */}
      {currentResult.onet_mapping.relevant_occupations.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>O*NET Occupation Matches</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {currentResult.onet_mapping.relevant_occupations.map((occ, i) => (
                <div key={i} className="flex items-center justify-between border-b border-slate-800 pb-3 last:border-0 last:pb-0">
                  <div>
                    <p className="text-sm font-medium text-slate-200">{occ.title}</p>
                    <p className="text-xs text-slate-500 mt-0.5 capitalize">
                      {occ.domain.replace(/_/g, ' ')}
                    </p>
                  </div>
                  <div className="text-right text-xs text-slate-400">
                    {occ.growth_rate != null && (
                      <p className="text-emerald-400">+{occ.growth_rate}% growth</p>
                    )}
                    {occ.median_salary != null && (
                      <p>${Math.round(occ.median_salary / 1000)}k median</p>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
