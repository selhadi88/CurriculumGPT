import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { AnalysisMode, AnalysisResult } from '@/types'

interface AppState {
  // Dashboard mode toggle
  mode: AnalysisMode
  setMode: (mode: AnalysisMode) => void

  // Current analysis
  currentAnalysisId: string | null
  setCurrentAnalysisId: (id: string | null) => void

  currentResult: AnalysisResult | null
  setCurrentResult: (result: AnalysisResult | null) => void

  // Analysis history (last 10)
  history: Array<{ id: string; mode: AnalysisMode; score: number; created_at: string }>
  addToHistory: (entry: { id: string; mode: AnalysisMode; score: number; created_at: string }) => void
  clearHistory: () => void

  // UI state
  sidebarCollapsed: boolean
  toggleSidebar: () => void
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      mode: 'individual',
      setMode: (mode) => set({ mode }),

      currentAnalysisId: null,
      setCurrentAnalysisId: (id) => set({ currentAnalysisId: id }),

      currentResult: null,
      setCurrentResult: (result) => set({ currentResult: result }),

      history: [],
      addToHistory: (entry) =>
        set((state) => ({
          history: [entry, ...state.history].slice(0, 10),
        })),
      clearHistory: () => set({ history: [] }),

      sidebarCollapsed: false,
      toggleSidebar: () =>
        set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
    }),
    {
      name: 'curriculumgpt-store',
      partialize: (state) => ({
        mode: state.mode,
        history: state.history,
        sidebarCollapsed: state.sidebarCollapsed,
        currentAnalysisId: state.currentAnalysisId,
        currentResult: state.currentResult,
      }),
    }
  )
)
