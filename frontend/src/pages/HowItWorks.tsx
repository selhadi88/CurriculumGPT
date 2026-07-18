import { Link } from 'react-router-dom'
import {
  FileInput, Database, Cpu, BarChart3, ArrowRight,
  Brain, Layers, Wrench, GraduationCap, Clock, TrendingUp, ListOrdered,
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'

const STEPS = [
  { icon: FileInput, title: '1. You provide a curriculum',
    body: 'Either type a list of skills/courses (Individual mode) or upload a CSV course catalog (Institution mode).' },
  { icon: Database, title: '2. We pull relevant jobs',
    body: 'The system searches 108,000 real job postings and selects the ones most related to your curriculum.' },
  { icon: Cpu, title: '3. The model compares them',
    body: 'The 7-component CurriculumGPT model scores how well your curriculum matches those jobs, on 7 dimensions.' },
  { icon: BarChart3, title: '4. You get an alignment report',
    body: 'An overall score, the 7-component breakdown, your skill gaps, certifications to take, and best-matching jobs.' },
]

const COMPONENTS = [
  { icon: Brain, name: 'Semantic Meaning', desc: 'Do the course and the jobs talk about the same things?', active: true },
  { icon: Layers, name: 'Topic Coverage', desc: 'Do they cover the same subject areas?', active: true },
  { icon: Wrench, name: 'Skill Match', desc: 'Do the specific skills overlap (Python, SQL, AWS…)?', active: true },
  { icon: GraduationCap, name: 'Cognitive Depth', desc: 'Is it taught at the right level (memorize vs. design)?', active: true },
  { icon: Clock, name: 'Recency', desc: 'Is the course up to date, vs. recent job postings?', active: true },
  { icon: TrendingUp, name: 'Industry Trend', desc: 'Aligned with where the industry is heading (12-month skill trends)?', active: true },
  { icon: ListOrdered, name: 'Course Structure', desc: 'Are topics taught in a sensible order vs. job emphasis?', active: true },
]

export function HowItWorks() {
  return (
    <div className="space-y-8 animate-fade-in max-w-4xl">
      <div>
        <h1 className="text-2xl font-bold text-slate-100">How GurriculumGPT Works</h1>
        <p className="text-sm text-slate-400 mt-1">
          A plain-language guide: what goes in, what comes out, and how the score is built.
        </p>
      </div>

      {/* What it does */}
      <Card>
        <CardContent className="py-5">
          <p className="text-slate-200 leading-relaxed">
            <b>In one sentence:</b> you give it a curriculum, and it tells you how well that prepares
            students for real jobs — with a score, the reasons behind it, the missing skills, and what to learn.
          </p>
        </CardContent>
      </Card>

      {/* The flow */}
      <div>
        <h2 className="text-base font-semibold text-slate-200 mb-4">The flow: input → output</h2>
        <div className="grid grid-cols-1 gap-3 md:grid-cols-4">
          {STEPS.map(({ icon: Icon, title, body }, i) => (
            <div key={title} className="relative">
              <Card className="h-full">
                <CardContent className="py-5">
                  <Icon className="h-6 w-6 text-brand-400 mb-3" />
                  <p className="text-sm font-semibold text-slate-100">{title}</p>
                  <p className="text-xs text-slate-400 mt-1.5 leading-relaxed">{body}</p>
                </CardContent>
              </Card>
              {i < STEPS.length - 1 && (
                <ArrowRight className="hidden md:block absolute top-1/2 -right-2.5 -translate-y-1/2 h-4 w-4 text-slate-600 z-10" />
              )}
            </div>
          ))}
        </div>
      </div>

      {/* The 7 components */}
      <div>
        <h2 className="text-base font-semibold text-slate-200 mb-1">The 7 components of the score</h2>
        <p className="text-xs text-slate-500 mb-4">
          The overall alignment score is a blend of 7 different ways of comparing curriculum to jobs.
          The model learns how much to weight each one.
        </p>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          {COMPONENTS.map(({ icon: Icon, name, desc, active }) => (
            <Card key={name}>
              <CardContent className="flex items-start gap-3 py-4">
                <Icon className={`h-5 w-5 mt-0.5 shrink-0 ${active ? 'text-indigo-400' : 'text-slate-600'}`} />
                <div>
                  <div className="flex items-center gap-2">
                    <p className="text-sm font-medium text-slate-200">{name}</p>
                    {!active && (
                      <span className="text-[10px] rounded bg-amber-900/40 text-amber-400 px-1.5 py-0.5">
                        inactive
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-slate-400 mt-0.5">{desc}</p>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
        <div className="mt-3 rounded-lg border border-slate-700 bg-slate-900/50 p-3">
          <p className="text-xs text-slate-400 leading-relaxed">
            <b>All 7 components are active.</b> Recency and Industry Trend are powered by job posting
            dates and a 12-month skill-frequency history; Course Structure compares the ordering of
            skills between curriculum and jobs. The job-date layer is a reproducible synthesis over the
            108K-job corpus (documented in the research notes) — the same code consumes real posting
            dates if a dated source (e.g. USAJOBS) is scraped in.
          </p>
        </div>
      </div>

      {/* Where data comes from */}
      <Card>
        <CardHeader>
          <CardTitle>Where the data comes from</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm text-slate-300">
          <p>• <b>Jobs:</b> 108,191 real job postings scraped from job boards (stored in the database).</p>
          <p>• <b>Courses:</b> 80,000 real course descriptions (used to train/test the model).</p>
          <p>• <b>Labels:</b> the model was trained on course–job pairs judged "aligned / not aligned" by an
            LLM (Groq llama-3.3-70b) — see the Reports page and the research files for details.</p>
          <p>• <b>Skills taxonomy:</b> a fixed list of ~49 skills with keywords used to detect what a course/job covers.</p>
        </CardContent>
      </Card>

      <div className="flex gap-3">
        <Link to="/analyze"><Button size="lg">Try an analysis →</Button></Link>
        <Link to="/jobs"><Button size="lg" variant="secondary">Browse the job data</Button></Link>
      </div>
    </div>
  )
}
