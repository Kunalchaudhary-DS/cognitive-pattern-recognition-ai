const BASE = 'http://localhost:8000'

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, options)
  const data = await res.json()
  return data
}

//Dataset
export async function uploadDataset(file) {
  const fd = new FormData()
  fd.append('file', file)
  return request('/upload/', { method: 'POST', body: fd })
}

export async function getDemoDatasets() {
  return request('/demo-datasets/')
}

export async function loadDemoDataset(name) {
  const fd = new FormData()
  fd.append('dataset_name', name)
  return request('/load-demo-dataset/', { method: 'POST', body: fd })
}

//Preprocessing
export async function getFeatureImportance(target) {
  const fd = new FormData()
  fd.append('target_column', target)
  return request('/feature-importance/', { method: 'POST', body: fd })
}

export async function runPreprocessing(target) {
  const fd = new FormData()
  fd.append('target_column', target)
  return request('/preprocess/', { method: 'POST', body: fd })
}

//Training
export async function trainModels() {
  return request('/train/', { method: 'POST' })
}

//Dashboard
export async function getDashboardData() {
  return request('/dashboard-data/')
}

//AI Endpoints
export async function getAIDatasetExplanation() {
  return request('/ai/dataset-explanation/', { method: 'POST' })
}

export async function getAITrainingExplanation() {
  return request('/ai/training-explanation/', { method: 'POST' })
}

export async function getAIPatternExplanation() {
  return request('/ai/pattern-explanation/', { method: 'POST' })
}

export async function getAIInsightSummary() {
  return request('/ai/insight-summary/')
}
