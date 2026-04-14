import { create } from 'zustand'

const useAppStore = create((set) => ({
  // Theme
  theme: 'dark',
  toggleTheme: () => set((state) => {
    const newTheme = state.theme === 'dark' ? 'light' : 'dark'
    document.documentElement.setAttribute('data-theme', newTheme)
    return { theme: newTheme }
  }),

  // Workflow State
  currentStep: 1,
  setStep: (step) => set({ currentStep: step }),

  //Dataset
  datasetLoaded: false,
  datasetInfo: null,
  datasetData: null,
  setDataset: (info, data) => set({
    datasetLoaded: true,
    datasetInfo: info,
    datasetData: data,
    currentStep: 2
  }),

  // Preprocessing 
  preprocessed: false,
  preprocessInfo: null,
  targetColumn: null,
  setPreprocessed: (info, target) => set({
    preprocessed: true,
    preprocessInfo: info,
    targetColumn: target,
    currentStep: 3
  }),

  //Training
  trained: false,
  trainingResults: null,
  setTrained: (results) => set({
    trained: true,
    trainingResults: results,
    currentStep: 4
  }),

  // Dashboard
  dashboardData: null,
  setDashboardData: (data) => set({ dashboardData: data }),

  //AI Explanations
  aiDatasetText: '',
  aiTrainingText: '',
  aiPatternText: '',
  aiSummaryText: '',
  setAiDatasetText: (text) => set({ aiDatasetText: text }),
  setAiTrainingText: (text) => set({ aiTrainingText: text }),
  setAiPatternText: (text) => set({ aiPatternText: text }),
  setAiSummaryText: (text) => set({ aiSummaryText: text }),


  // Dataset Finder 
  problemStatement: '',
  foundDatasets: [],
  setProblemStatement: (text) => set({ problemStatement: text }),
  setFoundDatasets: (datasets) => set({ foundDatasets: datasets }),

  //Reset
  reset: () => set({
    currentStep: 1,
    datasetLoaded: false,
    datasetInfo: null,
    datasetData: null,
    preprocessed: false,
    preprocessInfo: null,
    targetColumn: null,
    trained: false,
    trainingResults: null,
    dashboardData: null,
    aiDatasetText: '',
    aiTrainingText: '',
    aiPatternText: '',
    aiSummaryText: '',
    problemStatement: '',
    foundDatasets: [],
  })
}))

export default useAppStore