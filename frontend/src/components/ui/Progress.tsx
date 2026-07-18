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
  const radius = (size - 12) / 2
  const circumference = 2 * Math.PI * radius
  const offset = circumference - (value / 100) * circumference
  const color = value >= 70 ? '#10b981' : value >= 40 ? '#f59e0b' : '#ef4444'

  return (
    <svg width={size} height={size} className="-rotate-90">
      <circle cx={size / 2} cy={size / 2} r={radius} stroke="#1e293b" strokeWidth={10} fill="none" />
      <circle
        cx={size / 2}
        cy={size / 2}
        r={radius}
        stroke={color}
        strokeWidth={10}
        fill="none"
        strokeDasharray={circumference}
        strokeDashoffset={offset}
        strokeLinecap="round"
        className="transition-all duration-700"
      />
      <text
        x={size / 2}
        y={size / 2}
        textAnchor="middle"
        dominantBaseline="central"
        className="rotate-90"
        style={{ transform: `rotate(90deg) translate(0, 0)`, fill: '#f1f5f9', fontSize: size / 4, fontWeight: 700 }}
      >
        {Math.round(value)}
      </text>
    </svg>
  )
}
