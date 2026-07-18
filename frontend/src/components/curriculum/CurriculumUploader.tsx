import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { UploadCloud, FileText, X } from 'lucide-react'
import { clsx } from 'clsx'
import { Button } from '@/components/ui/Button'
import type { AnalysisMode } from '@/types'

interface CurriculumUploaderProps {
  mode: AnalysisMode
  onSubmit: (payload: { text?: string; csv_content?: string; target_domain?: string }) => void
  loading?: boolean
}

export function CurriculumUploader({ mode, onSubmit, loading }: CurriculumUploaderProps) {
  const [text, setText] = useState('')
  const [csvContent, setCsvContent] = useState('')
  const [csvFileName, setCsvFileName] = useState('')
  const [targetDomain, setTargetDomain] = useState('')

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = (e) => {
      setCsvContent(e.target?.result as string)
      setCsvFileName(file.name)
    }
    reader.readAsText(file)
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'text/csv': ['.csv'] },
    maxFiles: 1,
    disabled: loading,
  })

  const handleSubmit = () => {
    if (mode === 'institution') {
      if (!csvContent) return
      onSubmit({ csv_content: csvContent, target_domain: targetDomain || undefined })
    } else {
      if (!text.trim()) return
      onSubmit({ text: text.trim(), target_domain: targetDomain || undefined })
    }
  }

  const canSubmit = mode === 'institution' ? !!csvContent : text.trim().length > 0

  return (
    <div className="space-y-5">
      {mode === 'institution' ? (
        <>
          {/* CSV drop zone */}
          {!csvContent ? (
            <div
              {...getRootProps()}
              className={clsx(
                'flex flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-12 cursor-pointer transition-colors',
                isDragActive
                  ? 'border-brand-500 bg-brand-500/10'
                  : 'border-slate-700 hover:border-slate-600 bg-slate-900/50'
              )}
            >
              <input {...getInputProps()} />
              <UploadCloud className="h-10 w-10 text-slate-500 mb-3" />
              <p className="text-sm font-medium text-slate-300">
                {isDragActive ? 'Drop your CSV here' : 'Drop course catalog CSV or click to browse'}
              </p>
              <p className="text-xs text-slate-500 mt-1">CSV with title, description columns</p>
            </div>
          ) : (
            <div className="flex items-center justify-between rounded-lg border border-slate-700 bg-slate-900 px-4 py-3">
              <div className="flex items-center gap-3">
                <FileText className="h-4 w-4 text-brand-400" />
                <span className="text-sm text-slate-200">{csvFileName}</span>
                <span className="text-xs text-slate-500">
                  {csvContent.split('\n').length - 1} courses
                </span>
              </div>
              <button
                onClick={() => { setCsvContent(''); setCsvFileName('') }}
                className="text-slate-500 hover:text-slate-300"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          )}
        </>
      ) : (
        /* Text input for individual mode */
        <div>
          <label className="block text-sm font-medium text-slate-300 mb-2">
            Your completed courses / skills
          </label>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder={'Python Programming\nMachine Learning Fundamentals\nSQL and Database Design\nStatistics for Data Science\n…'}
            rows={8}
            disabled={loading}
            className="w-full rounded-lg border border-slate-700 bg-slate-900 px-4 py-3 text-sm text-slate-200 placeholder-slate-600 focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-500 disabled:opacity-50 resize-none"
          />
          <p className="mt-1 text-xs text-slate-500">Enter course names or skills — one per line or comma-separated</p>
        </div>
      )}

      {/* Domain filter (optional) */}
      <div>
        <label className="block text-sm font-medium text-slate-300 mb-2">
          Target Domain <span className="text-slate-500 font-normal">(optional)</span>
        </label>
        <select
          value={targetDomain}
          onChange={(e) => setTargetDomain(e.target.value)}
          disabled={loading}
          className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-200 focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-500 disabled:opacity-50"
        >
          <option value="">All domains</option>
          <option value="computer_science">Computer Science</option>
          <option value="data_science">Data Science & ML</option>
          <option value="cloud_devops">Cloud & DevOps</option>
          <option value="web_development">Web Development</option>
          <option value="cybersecurity">Cybersecurity</option>
          <option value="business">Business</option>
          <option value="design">Design & UX</option>
        </select>
      </div>

      <Button onClick={handleSubmit} disabled={!canSubmit} loading={loading} className="w-full" size="lg">
        {loading ? 'Analyzing…' : 'Run Analysis'}
      </Button>
    </div>
  )
}
