import { type HTMLAttributes } from 'react'
import { clsx } from 'clsx'

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  glass?: boolean
}

export function Card({ className, glass, children, ...props }: CardProps) {
  return (
    <div
      className={clsx(
        'rounded-xl border',
        glass
          ? 'border-white/10 bg-white/5 backdrop-blur-sm'
          : 'border-slate-800 bg-slate-900',
        className
      )}
      {...props}
    >
      {children}
    </div>
  )
}

export function CardHeader({ className, children, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={clsx('px-6 py-4 border-b border-slate-800', className)} {...props}>
      {children}
    </div>
  )
}

export function CardContent({ className, children, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={clsx('px-6 py-4', className)} {...props}>
      {children}
    </div>
  )
}

export function CardTitle({ className, children, ...props }: HTMLAttributes<HTMLHeadingElement>) {
  return (
    <h3 className={clsx('text-base font-semibold text-slate-100', className)} {...props}>
      {children}
    </h3>
  )
}
