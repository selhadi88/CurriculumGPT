import { clsx } from 'clsx'

interface ProgressProps {
  value: number  // 0–100
  className?: string
  color?: 'brand' | 'emerald' | 'amber' | 'red'
  showLabel?: boolean
  size?: 'sm' | 'md'
}

const colorMap = {
  brand: 'bg-brand-500',
  emerald: 'bg-emerald-500',
  amber: 'bg-amber-500',
  red: 'bg-red-500',
}

export function Progress({
  value,
  className,
  color = 'brand',
  showLabel = false,
  size = 'md',
}: ProgressProps) {
  const clamped = Math.min(100, Math.max(0, value))

  return (
    <div className={clsx('flex items-center gap-3', className)}>
      <div
        className={clsx(
          'flex-1 overflow-hidden rounded-full bg-slate-800',
          size === 'sm' ? 'h-1.5' : 'h-2'
        )}
      >
        <div
          className={clsx('h-full rounded-full transition-all duration-500', colorMap[color])}
          style={{ width: `${clamped}%` }}
        />
      </div>
      {showLabel && (
        <span className="w-10 text-right text-sm tabular-nums text-slate-400">
          {clamped.toFixed(0)}%
        </span>
      )}
    </div>
  )
}

export function ScoreRing({ value, size = 80 }: { value: number; size?: number }) {
  const v = Math.max(0, Math.min(100, Math.round(value || 0)))
  const c = size / 2
  const radius = (size - 12) / 2
  const circumference = 2 * Math.PI * radius
  const offset = circumference - (v / 100) * circumference
  const color = v >= 70 ? '#10b981' : v >= 40 ? '#f59e0b' : '#ef4444'

  // Only the progress arc is rotated (SVG attribute, around the centre). The
  // track and the number stay upright — rotating the whole <svg> sent the
  // <text> off-screen.
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <circle cx={c} cy={c} r={radius} stroke="#1e293b" strokeWidth={10} fill="none" />
      <circle
        cx={c}
        cy={c}
        r={radius}
        stroke={color}
        strokeWidth={10}
        fill="none"
        strokeDasharray={circumference}
        strokeDashoffset={offset}
        strokeLinecap="round"
        transform={`rotate(-90 ${c} ${c})`}
        style={{ transition: 'stroke-dashoffset 0.7s ease' }}
      />
      <text
        x={c}
        y={c}
        textAnchor="middle"
        dominantBaseline="central"
        style={{ fill: '#f1f5f9', fontSize: size / 3.2, fontWeight: 700 }}
      >
        {v}
      </text>
    </svg>
  )
}
