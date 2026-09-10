import { Building2, User } from 'lucide-react'
import { clsx } from 'clsx'
import { useAppStore } from '@/store/useAppStore'
import { ThemeToggle } from '@/components/ThemeToggle'

export function Header() {
  const { mode, setMode } = useAppStore()

  return (
    <header className="flex h-16 items-center justify-between border-b border-slate-800 bg-slate-950 px-6">
      <div />

      {/* Mode switcher */}
      <div className="flex items-center gap-1 rounded-lg bg-slate-900 p-1">
        <ModeButton
          active={mode === 'individual'}
          icon={<User className="h-3.5 w-3.5" />}
          label="Individual"
          onClick={() => setMode('individual')}
        />
        <ModeButton
          active={mode === 'institution'}
          icon={<Building2 className="h-3.5 w-3.5" />}
          label="Institution"
          onClick={() => setMode('institution')}
        />
      </div>

      <div className="flex items-center gap-3">
        <div className="hidden text-xs text-slate-500 sm:block">
          {mode === 'institution' ? 'Curriculum coordinator mode' : 'Student / job seeker mode'}
        </div>
        <ThemeToggle />
      </div>
    </header>
  )
}

function ModeButton({
  active,
  icon,
  label,
  onClick,
}: {
  active: boolean
  icon: React.ReactNode
  label: string
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      className={clsx(
        'flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors',
        active
          ? 'bg-brand-600 text-white shadow-sm'
          : 'text-slate-400 hover:text-slate-200'
      )}
    >
      {icon}
      {label}
    </button>
  )
}
