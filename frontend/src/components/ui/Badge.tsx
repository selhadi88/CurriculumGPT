import { clsx } from 'clsx'

type BadgeVariant = 'default' | 'success' | 'warning' | 'danger' | 'info'

const variantStyles: Record<BadgeVariant, string> = {
  default: 'bg-slate-800 text-slate-300 border-slate-700',
  success: 'bg-emerald-900/50 text-emerald-400 border-emerald-800',
  warning: 'bg-amber-900/50 text-amber-400 border-amber-800',
  danger: 'bg-red-900/50 text-red-400 border-red-800',
  info: 'bg-brand-900/50 text-brand-400 border-brand-800',
}

interface BadgeProps {
  variant?: BadgeVariant
  children: React.ReactNode
  className?: string
}

export function Badge({ variant = 'default', children, className }: BadgeProps) {
  return (
    <span
      className={clsx(
        'inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium',
        variantStyles[variant],
        className
      )}
    >
      {children}
    </span>
  )
}

export function severityBadge(severity: 'high' | 'medium' | 'low') {
  const map: Record<string, BadgeVariant> = {
    high: 'danger',
    medium: 'warning',
    low: 'success',
  }
  return map[severity] ?? 'default'
}
