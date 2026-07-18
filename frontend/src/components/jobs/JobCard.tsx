import { Building2, MapPin, DollarSign, ExternalLink } from 'lucide-react'
import { Badge } from '@/components/ui/Badge'
import { Progress } from '@/components/ui/Progress'
import type { Job } from '@/types'

interface JobCardProps {
  job: Job
  alignmentScore?: number
}

export function JobCard({ job, alignmentScore }: JobCardProps) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900 p-5 hover:border-slate-700 transition-colors">
      <div className="flex items-start justify-between gap-3 mb-3">
        <div className="flex-1 min-w-0">
          <h3 className="text-sm font-semibold text-slate-100 truncate">{job.title}</h3>
          {job.company && (
            <div className="flex items-center gap-1 mt-0.5 text-xs text-slate-400">
              <Building2 className="h-3 w-3 shrink-0" />
              {job.company}
            </div>
          )}
        </div>
        {job.url && (
          <a
            href={job.url}
            target="_blank"
            rel="noopener noreferrer"
            className="shrink-0 text-slate-500 hover:text-brand-400 transition-colors mt-0.5"
          >
            <ExternalLink className="h-3.5 w-3.5" />
          </a>
        )}
      </div>

      <p className="text-xs text-slate-400 line-clamp-2 mb-3">{job.description}</p>

      <div className="flex flex-wrap gap-1.5 mb-3">
        {job.domain && <Badge variant="info">{job.domain.replace(/_/g, ' ')}</Badge>}
        {job.location && (
          <Badge variant="default">
            <MapPin className="h-2.5 w-2.5 mr-1" />
            {job.location}
          </Badge>
        )}
        {(job.skills ?? []).slice(0, 3).map((skill) => (
          <Badge key={skill} variant="default">{skill}</Badge>
        ))}
      </div>

      {job.salary_min && job.salary_max && (
        <div className="flex items-center gap-1 text-xs text-emerald-400 mb-3">
          <DollarSign className="h-3 w-3" />
          {Math.round(job.salary_min / 1000)}k – {Math.round(job.salary_max / 1000)}k / yr
        </div>
      )}

      {alignmentScore != null && (
        <div>
          <div className="flex justify-between text-xs text-slate-500 mb-1">
            <span>Alignment</span>
            <span>{Math.round(alignmentScore * 100)}%</span>
          </div>
          <Progress
            value={alignmentScore * 100}
            color={alignmentScore >= 0.7 ? 'emerald' : alignmentScore >= 0.4 ? 'amber' : 'red'}
            size="sm"
          />
        </div>
      )}
    </div>
  )
}
