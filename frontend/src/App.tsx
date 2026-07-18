import { Routes, Route } from 'react-router-dom'
import { Layout } from '@/components/layout/Layout'
import { HowItWorks } from '@/pages/HowItWorks'
import { Dashboard } from '@/pages/Dashboard'
import { Analyzer } from '@/pages/Analyzer'
import { JobExplorer } from '@/pages/JobExplorer'
import { SkillGaps } from '@/pages/SkillGaps'
import { Reports } from '@/pages/Reports'

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/"          element={<Dashboard />} />
        <Route path="/how"       element={<HowItWorks />} />
        <Route path="/analyze"   element={<Analyzer />} />
        <Route path="/jobs"      element={<JobExplorer />} />
        <Route path="/skills"    element={<SkillGaps />} />
        <Route path="/reports"   element={<Reports />} />
      </Route>
    </Routes>
  )
}
