import { useMemo } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'

interface AlignmentHeatmapProps {
  matrix: number[][]
  rowLabels?: string[]
  colLabels?: string[]
  maxRows?: number
  maxCols?: number
}

function scoreToColor(score: number): string {
  const clamped = Math.max(0, Math.min(1, score))
  if (clamped >= 0.7) return 'bg-emerald-500/80'
  if (clamped >= 0.5) return 'bg-emerald-500/50'
  if (clamped >= 0.35) return 'bg-amber-500/60'
  if (clamped >= 0.2) return 'bg-orange-500/50'
  return 'bg-red-500/40'
}

export function AlignmentHeatmap({
  matrix,
  rowLabels,
  colLabels,
  maxRows = 15,
  maxCols = 10,
}: AlignmentHeatmapProps) {
  const rows = matrix.slice(0, maxRows)
  const cols = rows[0]?.length ?? 0
  const displayCols = Math.min(cols, maxCols)

  const displayRows = useMemo(
    () => rows.map((r) => r.slice(0, displayCols)),
    [rows, displayCols]
  )

  if (!displayRows.length) {
    return (
      <Card>
        <CardContent>
          <p className="text-sm text-slate-500">No matrix data available.</p>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Alignment Heatmap</CardTitle>
        <p className="text-xs text-slate-500 mt-1">
          Curriculum items (rows) × Job postings (columns) — darker = stronger alignment
        </p>
      </CardHeader>
      <CardContent className="overflow-x-auto">
        <div className="inline-block min-w-full">
          {/* Column labels */}
          {colLabels && (
            <div
              className="mb-1 grid gap-0.5"
              style={{ gridTemplateColumns: `120px repeat(${displayCols}, minmax(32px, 1fr))` }}
            >
              <div />
              {colLabels.slice(0, displayCols).map((label, i) => (
                <div
                  key={i}
                  className="text-center text-[9px] text-slate-500 truncate px-0.5"
                  title={label}
                >
                  {label.length > 8 ? label.slice(0, 8) + '…' : label}
                </div>
              ))}
            </div>
          )}

          {/* Rows */}
          <div className="space-y-0.5">
            {displayRows.map((row, ri) => (
              <div
                key={ri}
                className="grid gap-0.5 items-center"
                style={{ gridTemplateColumns: `120px repeat(${displayCols}, minmax(32px, 1fr))` }}
              >
                <div
                  className="text-[11px] text-slate-400 truncate pr-2 text-right"
                  title={rowLabels?.[ri]}
                >
                  {rowLabels?.[ri]
                    ? rowLabels[ri].length > 14 ? rowLabels[ri].slice(0, 14) + '…' : rowLabels[ri]
                    : `Course ${ri + 1}`}
                </div>
                {row.map((score, ci) => (
                  <div
                    key={ci}
                    className={`h-6 rounded-sm ${scoreToColor(score)} transition-opacity`}
                    title={`${rowLabels?.[ri] ?? `Row ${ri}`} × ${colLabels?.[ci] ?? `Col ${ci}`}: ${(score * 100).toFixed(1)}%`}
                  />
                ))}
              </div>
            ))}
          </div>

          {/* Color scale legend */}
          <div className="mt-4 flex items-center gap-2">
            <span className="text-xs text-slate-500">Low</span>
            <div className="flex gap-0.5">
              {['bg-red-500/40', 'bg-orange-500/50', 'bg-amber-500/60', 'bg-emerald-500/50', 'bg-emerald-500/80'].map(
                (cls, i) => (
                  <div key={i} className={`h-3 w-6 rounded-sm ${cls}`} />
                )
              )}
            </div>
            <span className="text-xs text-slate-500">High</span>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
