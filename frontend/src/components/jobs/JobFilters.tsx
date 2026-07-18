import { Search, Filter } from 'lucide-react'

const DOMAINS = [
  { value: '', label: 'All Domains' },
  { value: 'computer_science', label: 'Computer Science' },
  { value: 'data_science', label: 'Data Science & ML' },
  { value: 'cloud_devops', label: 'Cloud & DevOps' },
  { value: 'web_development', label: 'Web Development' },
  { value: 'cybersecurity', label: 'Cybersecurity' },
  { value: 'business', label: 'Business' },
  { value: 'design', label: 'Design & UX' },
]

interface JobFiltersProps {
  keyword: string
  domain: string
  onKeywordChange: (v: string) => void
  onDomainChange: (v: string) => void
}

export function JobFilters({ keyword, domain, onKeywordChange, onDomainChange }: JobFiltersProps) {
  return (
    <div className="flex flex-col gap-3 sm:flex-row">
      <div className="relative flex-1">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-slate-500" />
        <input
          value={keyword}
          onChange={(e) => onKeywordChange(e.target.value)}
          placeholder="Search jobs…"
          className="w-full rounded-lg border border-slate-700 bg-slate-900 py-2 pl-9 pr-4 text-sm text-slate-200 placeholder-slate-600 focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-500"
        />
      </div>

      <div className="relative">
        <Filter className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-slate-500" />
        <select
          value={domain}
          onChange={(e) => onDomainChange(e.target.value)}
          className="rounded-lg border border-slate-700 bg-slate-900 py-2 pl-9 pr-8 text-sm text-slate-200 focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-500 appearance-none"
        >
          {DOMAINS.map((d) => (
            <option key={d.value} value={d.value}>{d.label}</option>
          ))}
        </select>
      </div>
    </div>
  )
}
