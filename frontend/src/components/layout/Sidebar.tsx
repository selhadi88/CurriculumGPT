import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  Microscope,
  Briefcase,
  BarChart3,
  FileText,
  ChevronLeft,
  ChevronRight,
  GraduationCap,
  HelpCircle,
} from 'lucide-react'
import { clsx } from 'clsx'
import { useAppStore } from '@/store/useAppStore'

const NAV_ITEMS = [
  { to: '/how',       icon: HelpCircle,      label: 'How It Works' },
  { to: '/',          icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/analyze',   icon: Microscope,      label: 'Analyzer' },
  { to: '/jobs',      icon: Briefcase,       label: 'Job Explorer' },
  { to: '/skills',    icon: BarChart3,       label: 'Skill Gaps' },
  { to: '/reports',   icon: FileText,        label: 'Reports' },
]

export function Sidebar() {
  const { sidebarCollapsed, toggleSidebar } = useAppStore()

  return (
    <aside
      className={clsx(
        'flex flex-col border-r border-slate-800 bg-slate-950 transition-all duration-300',
        sidebarCollapsed ? 'w-16' : 'w-56'
      )}
    >
      {/* Logo */}
      <div className="flex h-16 items-center border-b border-slate-800 px-4">
        <GraduationCap className="h-6 w-6 shrink-0 text-brand-500" />
        {!sidebarCollapsed && (
          <span className="ml-3 text-sm font-bold tracking-tight text-slate-100">
            GurriculumGPT
          </span>
        )}
      </div>

      {/* Nav */}
      <nav className="flex-1 space-y-1 px-2 py-4">
        {NAV_ITEMS.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              clsx(
                'flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors',
                isActive
                  ? 'bg-brand-600/20 text-brand-400 font-medium'
                  : 'text-slate-400 hover:bg-slate-800 hover:text-slate-100'
              )
            }
          >
            <Icon className="h-4 w-4 shrink-0" />
            {!sidebarCollapsed && <span>{label}</span>}
          </NavLink>
        ))}
      </nav>

      {/* Collapse toggle */}
      <button
        onClick={toggleSidebar}
        className="flex h-12 items-center justify-center border-t border-slate-800 text-slate-500 hover:text-slate-300 transition-colors"
      >
        {sidebarCollapsed ? (
          <ChevronRight className="h-4 w-4" />
        ) : (
          <ChevronLeft className="h-4 w-4" />
        )}
      </button>
    </aside>
  )
}
