import { useState, useCallback , useEffect} from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import toast from 'react-hot-toast'
import useAppStore from '../store/appStore'
import {
  uploadDataset, getDemoDatasets, loadDemoDataset,
  getFeatureImportance, runPreprocessing, getAIDatasetExplanation
} from '../api/client'
import Card from '../components/ui/Card'
import Button from '../components/ui/Button'
import Badge from '../components/ui/Badge'
import AIPanel from '../components/ui/AIPanel'
import StepFlow from '../components/ui/StepFlow'
import PlotlyChart from '../components/charts/PlotlyChart'

export default function DataPage({ setActivePage }) {
  const { setDataset, setPreprocessed, setTrained, datasetLoaded,
        datasetInfo,foundDatasets } = useAppStore()

  const [uploading,    setUploading]    = useState(false)
  const [showDemo,     setShowDemo]     = useState(false)
  const [demoList,     setDemoList]     = useState([])
  const [selectedDemo, setSelectedDemo] = useState(null)
  const [target,       setTarget]       = useState('')
  const { aiDatasetText, setAiDatasetText } = useAppStore()
  const [aiLoading, setAiLoading] = useState(false)
  const [fiData,       setFiData]       = useState(null)
  const [preprocessing,setPreprocessing]= useState(false)
  const [preprocessInfo, setPreprocessInfo] = useState(null)
  const [training,     setTraining]     = useState(false)
  const [dragOver,     setDragOver]     = useState(false)

  // Auto-load dataset selected from landing page
  useEffect(() => {
    if (foundDatasets?.length > 0 && !datasetLoaded) {
      const ds = foundDatasets[0]
      setTimeout(() => handleDemoLoad(ds), 300)
    }
  }, [])
  
  // ── After dataset loaded ──────────────────────────────────
  async function afterLoad(data) {
    if (data.error) return toast.error(data.error)
    setDataset(data, data.full_data)
    if (data.columns?.length) setTarget(data.columns[data.columns.length - 1])
    toast.success(`Dataset loaded — ${data.rows?.toLocaleString()} rows`)
    // Fetch AI explanation
    if (!aiDatasetText) {
      setAiLoading(true)
      try {
        const ai = await getAIDatasetExplanation()
        setAiDatasetText(ai.explanation || '')
      } catch { setAiDatasetText('') }
      setAiLoading(false)
      }
    }

  // ── Upload ────────────────────────────────────────────────
  async function handleUpload(file) {
    if (!file) return
    setUploading(true)
    try {
      const data = await uploadDataset(file)
      await afterLoad(data)
    } catch { toast.error('Upload failed') }
    setUploading(false)
  }

  // ── Demo datasets ─────────────────────────────────────────
  async function toggleDemo() {
    if (!showDemo && demoList.length === 0) {
      const res = await getDemoDatasets()
      setDemoList(res.datasets || [])
    }
    setShowDemo(v => !v)
  }

  async function handleDemoLoad(ds) {
    if (!ds?.file) return 
    setSelectedDemo(ds.file)
    setUploading(true)
    try {
      const data = await loadDemoDataset(ds.file)
      await afterLoad(data)
    } catch { toast.error('Failed to load demo') }
    setUploading(false)
  }

  // ── Feature importance ────────────────────────────────────
  async function handleFeatureImportance() {
    if (!target) return toast.error('Select a target column')
    try {
      const data = await getFeatureImportance(target)
      if (data.error) return toast.error(data.error)
      const items = (data.feature_importance || []).slice(0, 10)
      setFiData({
        traces: [{
          x: items.map(i => i.correlation),
          y: items.map(i => i.feature),
          type: 'bar', orientation: 'h',
          marker: {
            color: items.map(i => i.correlation > 0 ? '#00d4ff' : '#f59e0b'),
            line: { width: 0 }
          }
        }],
        problemType: data.problem_type
      })
      toast.success('Feature analysis complete')
    } catch { toast.error('Feature importance failed') }
  }

  // ── Preprocessing ─────────────────────────────────────────
  async function handlePreprocess() {
    if (!target) return toast.error('Select a target column')
    setPreprocessing(true)
    try {
      const data = await runPreprocessing(target)
      if (data.error) return toast.error(data.error)
      setPreprocessed(data, target)
      setPreprocessInfo(data)
      toast.success('Preprocessing complete')
    } catch { toast.error('Preprocessing failed') }
    setPreprocessing(false)
  }

  // ── Training ──────────────────────────────────────────────
  async function handleTrain() {
    setTraining(true)
    toast.loading('AutoML training started...', { id: 'train' })
    try {
      const { trainModels } = await import('../api/client')
      const data = await trainModels()
      if (data.error) { toast.error(data.error, { id: 'train' }); return }
      setTrained(data)
      toast.success(`Training done — Best: ${data.BestModel}`, { id: 'train' })
      setActivePage('training')
    } catch { toast.error('Training failed', { id: 'train' }) }
    setTraining(false)
  }

  const catColors = ['blue','green','amber','cyan','violet']

  return (
    <div style={{ padding: '32px', maxWidth: 1400, margin: '0 auto' }}>
      <StepFlow/>

      {/* UPLOAD CARD */}
      <Card delay={0.05} style={{ marginBottom: 20 }}>
        <div style={{ fontSize: 13, fontWeight: 600, letterSpacing: '0.08em',
          textTransform: 'uppercase', color: 'var(--text-secondary)',
          marginBottom: 20, display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ width: 3, height: 14, background: 'var(--accent)',
            borderRadius: 2, display: 'inline-block' }}/>
          Dataset Input
        </div>

        {/* Drop zone */}
        <motion.div
          onDragOver={e => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={e => {
            e.preventDefault(); setDragOver(false)
            handleUpload(e.dataTransfer.files[0])
          }}
          onClick={() => document.getElementById('fileInput').click()}
          animate={{ borderColor: dragOver ? 'var(--accent)' : 'var(--border-hover)',
            background: dragOver ? 'rgba(0,212,255,0.06)' : 'rgba(0,212,255,0.02)' }}
          style={{
            border: '2px dashed var(--border-hover)',
            borderRadius: 12, padding: '48px',
            textAlign: 'center', cursor: 'pointer',
            transition: 'all 0.3s'
          }}
        >
          <input id="fileInput" type="file" accept=".csv"
            style={{ display: 'none' }}
            onChange={e => handleUpload(e.target.files[0])}/>
          <div style={{ fontSize: 40, marginBottom: 12 }}>📂</div>
          <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 6 }}>
            {uploading ? 'Loading...' : 'Drop your CSV file here'}
          </div>
          {datasetLoaded && datasetInfo && (
            <div style={{
              marginTop: 12,
              display: 'inline-flex',
              alignItems: 'center',
              gap: 8,
              background: 'rgba(0,212,255,0.08)',
              border: '1px solid rgba(0,212,255,0.25)',
              borderRadius: 20,
              padding: '4px 14px',
              fontSize: 12,
              fontFamily: 'JetBrains Mono, monospace',
              color: 'var(--accent)',
            }}>
              <span style={{ opacity: 0.6 }}>CURRENT DATASET</span>
              <span style={{ fontWeight: 600 }}>
                {foundDatasets?.[0]?.name || 'Uploaded File'}
              </span>
            </div>
          )}
          <div style={{ fontSize: 13, color: 'var(--text-muted)',
            fontFamily: 'JetBrains Mono, monospace' }}>
            or click to browse · CSV files only
          </div>
        </motion.div>

        <div style={{ height: 1, background: 'var(--border)', margin: '24px 0' }}/>

        {/* Demo datasets */}
        <Button onClick={toggleDemo}>
          {showDemo ? '▲ Hide' : '▼ Browse'} Demo Datasets
        </Button>

        <AnimatePresence>
          {showDemo && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              style={{ marginTop: 16, overflow: 'hidden' }}
            >
              <div style={{ display: 'grid',
                gridTemplateColumns: 'repeat(auto-fill, minmax(190px, 1fr))',
                gap: 12 }}>
                {demoList.map((ds, i) => (
                  <motion.div
                    key={ds.file}
                    whileHover={{ scale: 1.03, y: -2 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => handleDemoLoad(ds)}
                    style={{
                      background: selectedDemo === ds.file
                        ? 'rgba(0,212,255,0.08)' : 'var(--bg-tertiary)',
                      border: `1px solid ${selectedDemo === ds.file
                        ? 'rgba(0,212,255,0.4)' : 'var(--border)'}`,
                      borderRadius: 10, padding: 16, cursor: 'pointer',
                      transition: 'all 0.2s'
                    }}
                  >
                    <Badge color={catColors[i % catColors.length]}
                      style={{ marginBottom: 8 }}>
                      {ds.category}
                    </Badge>
                    <div style={{ fontSize: 13, fontWeight: 600,
                      color: 'var(--text-primary)', lineHeight: 1.3,
                      marginTop: 6 }}>
                      {ds.name}
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </Card>

      {/* STATS + CONTROLS */}
      <AnimatePresence>
        {datasetLoaded && datasetInfo && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            {/* Stat cards */}
            <div style={{ display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
              gap: 16, marginBottom: 20 }}>
              {[
                { label: 'Total Rows', value: datasetInfo.rows?.toLocaleString(), sub: 'data points', color: 'var(--accent)' },
                { label: 'Columns', value: datasetInfo.total_columns, sub: `${datasetInfo.numerical_columns?.length} num · ${datasetInfo.categorical_columns?.length} cat`, color: 'var(--accent)' },
                { label: 'Quality Score', value: `${datasetInfo.profile_summary?.quality_score}%`, sub: 'data quality', color: datasetInfo.profile_summary?.quality_score >= 80 ? 'var(--neon-green)' : 'var(--neon-amber)' },
                { label: 'Suggested Task', value: datasetInfo.profile_summary?.suggested_problem, sub: 'auto-detected', color: 'var(--neon-violet)' },
                { label: 'Missing Data', value: `${datasetInfo.profile_summary?.missing_percent}%`, sub: 'of all cells', color: 'var(--neon-amber)' },
              ].map((s, i) => (
                <motion.div
                  key={s.label}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.08 }}
                  className="glass"
                  style={{ padding: 20, position: 'relative', overflow: 'hidden' }}
                >
                  <div style={{ position: 'absolute', top: 0, left: 0, right: 0,
                    height: 2, background: s.color }}/>
                  <div style={{ fontSize: 11, fontFamily: 'JetBrains Mono, monospace',
                    color: 'var(--text-muted)', letterSpacing: '0.08em',
                    textTransform: 'uppercase', marginBottom: 8 }}>
                    {s.label}
                  </div>
                  <div style={{ fontSize: 26, fontWeight: 800,
                    color: s.color, lineHeight: 1, marginBottom: 4 }}>
                    {s.value}
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>{s.sub}</div>
                </motion.div>
              ))}
            </div>

            {/* Controls */}
            <Card delay={0.1} style={{ marginBottom: 20 }}>
              <div style={{ fontSize: 13, fontWeight: 600, letterSpacing: '0.08em',
                textTransform: 'uppercase', color: 'var(--text-secondary)',
                marginBottom: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ width: 3, height: 14, background: 'var(--accent)',
                  borderRadius: 2, display: 'inline-block' }}/>
                Analysis Controls
              </div>
              <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'flex-end' }}>
                <div>
                  <div style={{ fontSize: 11, fontFamily: 'JetBrains Mono, monospace',
                    color: 'var(--text-muted)', marginBottom: 6, letterSpacing: '0.06em' }}>
                    TARGET COLUMN
                  </div>
                  <select value={target} onChange={e => setTarget(e.target.value)}
                    style={{
                      padding: '10px 16px', background: 'var(--bg-tertiary)',
                      border: '1px solid var(--border-hover)',
                      color: 'var(--text-primary)', borderRadius: 10,
                      fontFamily: 'Inter, sans-serif', fontSize: 13,
                      cursor: 'pointer', minWidth: 200
                    }}>
                    {(datasetInfo.columns || []).map(c => (
                      <option key={c} value={c}>{c}</option>
                    ))}
                  </select>
                </div>
                <Button onClick={handleFeatureImportance}>⚡ Analyze Target</Button>
                <Button variant="primary" onClick={handlePreprocess} loading={preprocessing}>
                  ⚙️ Preprocess
                </Button>
                <Button variant="success" onClick={handleTrain} loading={training}>
                  🚀 Train Models
                </Button>
              </div>
            </Card>

            {/* AI Panel */}
            <Card delay={0.15} style={{ marginBottom: 20 }}>
              <div style={{ fontSize: 13, fontWeight: 600, letterSpacing: '0.08em',
                textTransform: 'uppercase', color: 'var(--text-secondary)',
                marginBottom: 4, display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ width: 3, height: 14, background: 'var(--neon-violet)',
                  borderRadius: 2, display: 'inline-block' }}/>
                AI Dataset Analysis
              </div>
              <AIPanel text={aiDatasetText} loading={aiLoading}/>
            </Card>

            {/* Missing values */}
            {datasetInfo.missing_summary && Object.keys(datasetInfo.missing_summary).length > 0 && (
              <Card delay={0.2} style={{ marginBottom: 20 }}>
                <div style={{ fontSize: 13, fontWeight: 600, letterSpacing: '0.08em',
                  textTransform: 'uppercase', color: 'var(--text-secondary)',
                  marginBottom: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{ width: 3, height: 14, background: 'var(--neon-amber)',
                    borderRadius: 2, display: 'inline-block' }}/>
                  Missing Values
                </div>
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                    <thead>
                      <tr>{['Column','Count','Percentage'].map(h => (
                        <th key={h} style={{ background: 'var(--bg-tertiary)',
                          color: 'var(--text-secondary)', padding: '10px 16px',
                          fontFamily: 'JetBrains Mono, monospace', fontSize: 11,
                          textAlign: 'left', borderBottom: '1px solid var(--border)' }}>
                          {h}
                        </th>
                      ))}</tr>
                    </thead>
                    <tbody>
                      {Object.entries(datasetInfo.missing_summary).map(([col, v]) => (
                        <tr key={col}>
                          <td style={{ padding: '10px 16px', fontFamily: 'JetBrains Mono, monospace',
                            fontSize: 12, color: 'var(--text-secondary)',
                            borderBottom: '1px solid rgba(0,212,255,0.04)' }}>{col}</td>
                          <td style={{ padding: '10px 16px',
                            borderBottom: '1px solid rgba(0,212,255,0.04)' }}>
                            <Badge color="amber">{v.count}</Badge>
                          </td>
                          <td style={{ padding: '10px 16px',
                            borderBottom: '1px solid rgba(0,212,255,0.04)' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                              <div style={{ flex: 1, height: 4, background: 'var(--bg-tertiary)',
                                borderRadius: 2, minWidth: 80 }}>
                                <div style={{ width: `${Math.min(v.percentage, 100)}%`,
                                  height: '100%', background: 'var(--neon-amber)',
                                  borderRadius: 2 }}/>
                              </div>
                              <span style={{ fontFamily: 'JetBrains Mono, monospace',
                                fontSize: 11, color: 'var(--text-muted)' }}>
                                {v.percentage}%
                              </span>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>
            )}

            {/* Feature importance chart */}
            {fiData && (
              <Card delay={0.25} style={{ marginBottom: 20 }}>
                <div style={{ fontSize: 13, fontWeight: 600, letterSpacing: '0.08em',
                  textTransform: 'uppercase', color: 'var(--text-secondary)',
                  marginBottom: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{ width: 3, height: 14, background: 'var(--accent2)',
                    borderRadius: 2, display: 'inline-block' }}/>
                  Feature Importance
                  <Badge color={fiData.problemType === 'classification' ? 'violet' : 'blue'}
                    style={{ marginLeft: 8 }}>
                    {fiData.problemType}
                  </Badge>
                </div>
                <div style={{ height: 300 }}>
                  <PlotlyChart
                    data={fiData.traces}
                    layout={{ xaxis: { title: 'Correlation' }, yaxis: { automargin: true } }}
                  />
                </div>
              </Card>
            )}

            {/* Preprocessing report */}
            {preprocessInfo && (
              <Card delay={0.3} style={{ marginBottom: 20 }}>
                <div style={{ fontSize: 13, fontWeight: 600, letterSpacing: '0.08em',
                  textTransform: 'uppercase', color: 'var(--text-secondary)',
                  marginBottom: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{ width: 3, height: 14, background: 'var(--neon-green)',
                    borderRadius: 2, display: 'inline-block' }}/>
                  Preprocessing Report
                </div>
                <div style={{ display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
                  gap: 12, marginBottom: 16 }}>
                  {[
                    { label: 'Original Shape', value: `${preprocessInfo.original_shape?.[0]} × ${preprocessInfo.original_shape?.[1]}` },
                    { label: 'Features After', value: preprocessInfo.processed_feature_shape?.[1] },
                    { label: 'Problem Type',   value: preprocessInfo.problem_type },
                    { label: 'Dropped Rows',   value: preprocessInfo.dropped_target_rows },
                  ].map(item => (
                    <div key={item.label} style={{ background: 'var(--bg-tertiary)',
                      border: '1px solid var(--border)', borderRadius: 10, padding: 16 }}>
                      <div style={{ fontSize: 11, fontFamily: 'JetBrains Mono, monospace',
                        color: 'var(--text-muted)', marginBottom: 6 }}>{item.label}</div>
                      <div style={{ fontSize: 18, fontWeight: 700 }}>{item.value}</div>
                    </div>
                  ))}
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  {[
                    { label: 'Binary Encoded', items: preprocessInfo.binary_encoded, color: 'blue' },
                    { label: 'OneHot Encoded', items: preprocessInfo.onehot_encoded, color: 'violet' },
                    { label: 'Frequency Encoded', items: preprocessInfo.frequency_encoded, color: 'amber' },
                  ].map(row => (
                    <div key={row.label}>
                      <div style={{ fontSize: 11, fontFamily: 'JetBrains Mono, monospace',
                        color: 'var(--text-muted)', marginBottom: 6, letterSpacing: '0.06em' }}>
                        {row.label}
                      </div>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                        {row.items?.length ? row.items.map(t => (
                          <Badge key={t} color={row.color}>{t}</Badge>
                        )) : <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>None</span>}
                      </div>
                    </div>
                  ))}
                </div>
              </Card>
            )}

            {/* Data preview */}
            <Card delay={0.35}>
              <div style={{ fontSize: 13, fontWeight: 600, letterSpacing: '0.08em',
                textTransform: 'uppercase', color: 'var(--text-secondary)',
                marginBottom: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ width: 3, height: 14, background: 'var(--text-muted)',
                  borderRadius: 2, display: 'inline-block' }}/>
                Dataset Preview
              </div>
              <div style={{ overflowX: 'auto', borderRadius: 10,
                border: '1px solid var(--border)' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr>{(datasetInfo.columns || []).map(c => (
                      <th key={c} style={{ background: 'var(--bg-tertiary)',
                        color: 'var(--text-secondary)', padding: '10px 16px',
                        fontFamily: 'JetBrains Mono, monospace', fontSize: 11,
                        textAlign: 'left', borderBottom: '1px solid var(--border)',
                        whiteSpace: 'nowrap' }}>{c}</th>
                    ))}</tr>
                  </thead>
                  <tbody>
                    {(datasetInfo.preview || []).map((row, i) => (
                      <tr key={i}>
                        {(datasetInfo.columns || []).map(c => (
                          <td key={c} style={{ padding: '10px 16px',
                            fontFamily: 'JetBrains Mono, monospace', fontSize: 11,
                            color: 'var(--text-secondary)',
                            borderBottom: '1px solid rgba(0,212,255,0.04)',
                            whiteSpace: 'nowrap' }}>
                            {row[c] ?? '—'}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}