const BASE = 'http://localhost:8000'

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, options)
  const data = await res.json()
  return data
}

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

export async function trainModels() {
  return request('/train/', { method: 'POST' })
}

export async function getDashboardData() {
  return request('/dashboard-data/')
}

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

export async function findDatasets(problemStatement) {
  const fd = new FormData()
  fd.append('problem_statement', problemStatement)
  return request('/find-datasets/', { method: 'POST', body: fd })
}

export async function getPrediction(values) {
  return request('/predict/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ values })
  })
}

export async function getSamplePredictions() {
  return request('/sample-predictions/')
}

export async function getEncodingMaps() {
  return request('/encoding-maps/')
}