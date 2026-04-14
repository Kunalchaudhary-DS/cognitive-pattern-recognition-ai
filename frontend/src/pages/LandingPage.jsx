import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { findDatasets, searchKaggleDatasets, downloadKaggleDataset, loadDemoDataset } from '../api/client.js'
import useAppStore from '../store/appStore'
import Badge from '../components/ui/Badge'

//Usability stars 
function UsabilityStars({ rating }) {
  const filled = Math.round((rating || 0) * 5)
  return (
    <span style={{ letterSpacing: 1 }}>
      {Array.from({ length: 5 }, (_, i) => (
        <span key={i} style={{ color: i < filled ? '#f59e0b' : 'var(--border-hover)', fontSize: 14 }}>★</span>
      ))}
      <span style={{ marginLeft: 6, fontSize: 11, color: 'var(--text-muted)' }}>
        {rating ? (rating * 5).toFixed(1) : '—'}/5
      </span>
    </span>
  )
}

// INLINE info panel shown below each Kaggle dataset row
function KaggleInfoPanel({ ds }) {
  const overview = ds.description && ds.description !== 'No description available.'
    ? ds.description
    : `This dataset covers topics related to your problem. File size is ${ds.size_mb || 0} MB.`

  return (
    <motion.div
      key="kaggle-info"
      initial={{ opacity: 0, height: 0, marginTop: 0 }}
      animate={{ opacity: 1, height: 'auto', marginTop: 8 }}
      exit={{ opacity: 0, height: 0, marginTop: 0 }}
      transition={{ duration: 0.22, ease: 'easeInOut' }}
      style={{ overflow: 'hidden' }}
    >
      <div style={{
        background: 'var(--bg-tertiary)',
        border: '1px solid var(--border)',
        borderRadius: 10,
        padding: '14px 18px',
        display: 'flex',
        alignItems: 'flex-start',
        gap: 20,
        flexWrap: 'wrap',
      }}>
        {/* Quick stats */}
        <div style={{ display: 'flex', gap: 10, flexShrink: 0, flexWrap: 'wrap' }}>
          {[
            { label: 'Size', value: `${ds.size_mb || 0} MB`, sub: null },
            { label: 'Downloads', value: (ds.download_count || 0).toLocaleString(), sub: null },
            { label: 'Rows', value: '—', sub: 'after download' },
            { label: 'Columns', value: '—', sub: 'after download' },
            { label: 'Format', value: 'CSV', sub: null },
          ].map(s => (
            <div key={s.label} style={{
              textAlign: 'center',
              background: 'var(--bg-secondary)',
              borderRadius: 8, padding: '8px 12px',
              border: '1px solid var(--border)',
            }}>
              <div style={{ fontSize: 9, color: 'var(--text-muted)', fontFamily: 'JetBrains Mono, monospace', marginBottom: 3, letterSpacing: '0.08em' }}>{s.label}</div>
              <div style={{ fontSize: 13, fontWeight: 700, color: s.value === '—' ? 'var(--text-muted)' : 'var(--text-primary)' }}>{s.value}</div>
              {s.sub && <div style={{ fontSize: 8, color: 'var(--text-muted)', fontFamily: 'JetBrains Mono, monospace', marginTop: 2 }}>{s.sub}</div>}
            </div>
          ))}
        </div>

        {/* Usability */}
        <div style={{ flexShrink: 0 }}>
          <div style={{ fontSize: 9, color: 'var(--text-muted)', fontFamily: 'JetBrains Mono, monospace', marginBottom: 5, letterSpacing: '0.08em' }}>USABILITY</div>
          <UsabilityStars rating={ds.usability} />
        </div>

        {/* Overview text */}
        <div style={{ flex: 1, minWidth: 200 }}>
          <div style={{ fontSize: 9, color: 'var(--text-muted)', fontFamily: 'JetBrains Mono, monospace', marginBottom: 6, letterSpacing: '0.08em' }}>OVERVIEW</div>
          <div style={{ fontSize: 12.5, color: 'var(--text-secondary)', lineHeight: 1.75 }}>
            {overview}
          </div>
        </div>
      </div>
    </motion.div>
  )
}

//INLINE info panel shown below each Local dataset row 
function LocalInfoPanel({ ds }) {
  const rows = ds.total_rows ? ds.total_rows.toLocaleString() : '—'
  const cols = ds.total_columns || '—'
  const sizeMb = ds.size_mb ? `${ds.size_mb} MB` : '—'

  // Generate a 3-sentence overview matching the same style as Kaggle
  function buildOverview() {
    const parts = []

    // Sentence 1: what's inside the dataset
    const numCols = ds.numerical_cols?.length || 0
    const catCols = ds.categorical_cols?.length || 0
    const totalCols = ds.total_columns || (numCols + catCols)
    if (ds.total_rows && totalCols) {
      parts.push(
        `This dataset contains ${ds.total_rows.toLocaleString()} rows and ${totalCols} columns` +
        (numCols && catCols
          ? ` (${numCols} numerical, ${catCols} categorical).`
          : '.')
      )
    } else {
      parts.push(`This is a curated local dataset in the ${ds.category || 'general'} domain.`)
    }

    // Sentence 2: why it matches the problem (the relevance reason)
    if (ds.reason) {
      parts.push(ds.reason)
    }

    // Sentence 3: size / training time warning (same thresholds as Kaggle backend)
    const size = ds.size_mb || 0
    if (size <= 0) {
      parts.push('File size is unknown — it should still load fine inside the AutoML pipeline.')
    } else if (size < 1) {
      parts.push(`The file is tiny (${size} MB), so it will download and load almost instantly.`)
    } else if (size <= 2) {
      parts.push(`At ${size} MB it is compact and will process quickly inside the AutoML pipeline.`)
    } else if (size <= 10) {
      parts.push(`At ${size} MB it is a medium-sized file — training will take a couple of minutes.`)
    } else if (size <= 50) {
      parts.push(`This dataset is ${size} MB which is quite large; the AutoML pipeline will consume more time and may use its sampling strategy.`)
    } else {
      parts.push(`Warning: this dataset is ${size} MB — it is very large and the AutoML pipeline will take significant time to train.`)
    }

    return parts.join(' ')
  }

  return (
    <motion.div
      key="local-info"
      initial={{ opacity: 0, height: 0, marginTop: 0 }}
      animate={{ opacity: 1, height: 'auto', marginTop: 8 }}
      exit={{ opacity: 0, height: 0, marginTop: 0 }}
      transition={{ duration: 0.22, ease: 'easeInOut' }}
      style={{ overflow: 'hidden' }}
    >
      <div style={{
        background: 'var(--bg-tertiary)',
        border: '1px solid var(--border)',
        borderRadius: 10,
        padding: '14px 18px',
        display: 'flex',
        alignItems: 'flex-start',
        gap: 20,
        flexWrap: 'wrap',
      }}>
        {/* Quick stats */}
        <div style={{ display: 'flex', gap: 10, flexShrink: 0, flexWrap: 'wrap' }}>
          {[
            { label: 'Rows', value: rows },
            { label: 'Columns', value: cols },
            { label: 'Size', value: sizeMb },
          ].map(s => (
            <div key={s.label} style={{
              textAlign: 'center',
              background: 'var(--bg-secondary)',
              borderRadius: 8, padding: '8px 12px',
              border: '1px solid var(--border)',
            }}>
              <div style={{ fontSize: 9, color: 'var(--text-muted)', fontFamily: 'JetBrains Mono, monospace', marginBottom: 3, letterSpacing: '0.08em' }}>{s.label}</div>
              <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)' }}>{s.value}</div>
            </div>
          ))}
        </div>

        {/* Key columns */}
        {ds.key_columns && ds.key_columns.length > 0 && (
          <div style={{ flexShrink: 0 }}>
            <div style={{ fontSize: 9, color: 'var(--text-muted)', fontFamily: 'JetBrains Mono, monospace', marginBottom: 5, letterSpacing: '0.08em' }}>KEY COLUMNS</div>
            <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap' }}>
              {ds.key_columns.slice(0, 5).map(col => (
                <span key={col} style={{
                  background: 'rgba(99,102,241,0.1)',
                  border: '1px solid rgba(99,102,241,0.2)',
                  borderRadius: 5, padding: '2px 8px',
                  fontSize: 10, color: 'var(--neon-violet)',
                  fontFamily: 'JetBrains Mono, monospace',
                }}>{col}</span>
              ))}
            </div>
          </div>
        )}

        {/* Overview text */}
        <div style={{ flex: 1, minWidth: 200 }}>
          <div style={{ fontSize: 9, color: 'var(--text-muted)', fontFamily: 'JetBrains Mono, monospace', marginBottom: 6, letterSpacing: '0.08em' }}>OVERVIEW</div>
          <div style={{ fontSize: 12.5, color: 'var(--text-secondary)', lineHeight: 1.75 }}>
            {buildOverview()}
          </div>
        </div>
      </div>
    </motion.div>
  )
}


//Main component
export default function LandingPage({ onComplete }) {
  const [problem, setProblem] = useState('')
  const [loading, setLoading] = useState(false)
  const [matches, setMatches] = useState([])
  const [searched, setSearched] = useState(false)
  const [expandedLocal, setExpandedLocal] = useState(null)   // ds.file for local
  const [expandedKaggle, setExpandedKaggle] = useState(null) // ds.ref for kaggle
  const [error, setError] = useState('')
  const [kaggleResults, setKaggleResults] = useState([])
  const [kaggleLoading, setKaggleLoading] = useState(false)
  const [downloading, setDownloading] = useState(null)
  const [showKaggle, setShowKaggle] = useState(false)
  const [downloadedFile, setDownloadedFile] = useState(null)

  const { setFoundDatasets, setProblemStatement } = useAppStore()

  const hasResults = (searched && !loading && matches.length > 0) ||
    (showKaggle && !kaggleLoading && kaggleResults.length > 0)

  const examples = [
    "Predict employee attrition in a company",
    "Analyze student academic performance",
    "Detect stroke risk from health data",
    "Forecast personal finance spending patterns",
    "Classify customer churn in e-commerce",
  ]

  async function handleSearch() {
    if (!problem.trim()) return setError('Please enter a problem statement')
    setError('')
    setLoading(true)
    setMatches([])
    setShowKaggle(false)
    setKaggleResults([])
    setExpandedLocal(null)
    setExpandedKaggle(null)
    try {
      const res = await findDatasets(problem)
      if (res.error) return setError(res.error)
      setMatches(res.matches || [])
      setSearched(true)
      setProblemStatement(problem)
      setFoundDatasets(res.matches || [])
    } catch {
      setError('Search failed. Make sure backend is running.')
    }
    setLoading(false)
  }

  async function handleKaggleSearch() {
    if (!problem.trim()) return setError('Please enter a problem statement')
    setError('')
    setKaggleLoading(true)
    setKaggleResults([])
    setShowKaggle(true)
    setSearched(false)
    setMatches([])
    setExpandedLocal(null)
    setExpandedKaggle(null)
    try {
      const res = await searchKaggleDatasets(problem)
      if (res.error) return setError(res.error)
      setKaggleResults(res.datasets || [])
    } catch {
      setError('Kaggle search failed. Make sure backend is running.')
    }
    setKaggleLoading(false)
  }

  async function handleKaggleDownload(ds) {
    setDownloading(ds.ref)
    setError('')
    try {
      const res = await downloadKaggleDataset(ds.ref, ds.title)
      if (res.error) {
        setError(res.error)
        setDownloading(null)
        return
      }
      setDownloadedFile(res.files?.[0])
      if (res.files && res.files.length > 0) {
        const dataRes = await loadDemoDataset(res.files[0])
        if (!dataRes.error) {
          setFoundDatasets([{ file: res.files[0], name: ds.title }])
          setProblemStatement(problem)
          onComplete({ file: res.files[0], name: ds.title })
        } else {
          setError('Dataset downloaded but could not be loaded: ' + dataRes.error)
        }
      }
    } catch (e) {
      setError('Download failed: ' + e.message)
    }
    setDownloading(null)
  }

  function getScoreColor(score) {
    if (score >= 80) return 'var(--neon-green)'
    if (score >= 60) return 'var(--neon-amber)'
    return 'var(--neon-red)'
  }

  const catColors = {
    'Finance': 'green',
    'Healthcare': 'red',
    'Education': 'blue',
    'Human Resources': 'violet',
    'Sports': 'cyan',
    'Technology': 'cyan',
    'Retail': 'amber',
  }

  function getRelevanceBadgeStyle(score) {
    if (score >= 70) return { bg: 'rgba(52,211,153,0.12)', border: 'rgba(52,211,153,0.3)', color: 'var(--neon-green)' }
    if (score >= 45) return { bg: 'rgba(251,191,36,0.12)', border: 'rgba(251,191,36,0.3)', color: '#f59e0b' }
    return { bg: 'rgba(156,163,175,0.12)', border: 'rgba(156,163,175,0.3)', color: 'var(--text-muted)' }
  }

  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: hasResults ? 'flex-start' : 'center',
      padding: '40px 32px',
      position: 'relative',
    }}>

      {/* Background blobs*/}
      <div style={{ position: 'fixed', inset: 0, zIndex: 0, overflow: 'hidden', pointerEvents: 'none' }}>
        <motion.div
          animate={{ x: [0, 30, 0], y: [0, -20, 0] }}
          transition={{ duration: 8, repeat: Infinity }}
          style={{
            position: 'absolute', top: '20%', left: '10%',
            width: 400, height: 400, borderRadius: '50%',
            background: 'radial-gradient(circle, rgba(139,92,246,0.06), transparent)',
            filter: 'blur(40px)',
          }}
        />
        <motion.div
          animate={{ x: [0, -20, 0], y: [0, 30, 0] }}
          transition={{ duration: 10, repeat: Infinity }}
          style={{
            position: 'absolute', bottom: '20%', right: '10%',
            width: 500, height: 500, borderRadius: '50%',
            background: 'radial-gradient(circle, rgba(99,102,241,0.06), transparent)',
            filter: 'blur(40px)',
          }}
        />
      </div>

      {/* Header (hidden when results shown)*/}
      <AnimatePresence>
        {!hasResults && (
          <motion.div
            key="header"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.4 }}
            style={{ textAlign: 'center', marginBottom: 48, zIndex: 1 }}
          >
            <motion.div
              animate={{ rotate: [0, 360] }}
              transition={{ duration: 20, repeat: Infinity, ease: 'linear' }}
              style={{ fontSize: 56, marginBottom: 20 }}
            >
              🧠
            </motion.div>
            <h1 style={{
              fontSize: 36, fontWeight: 700,
              letterSpacing: '0.02em', marginBottom: 12,
              background: 'var(--gradient-accent)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}>
              AI Cognitive Pattern Recognition
            </h1>
            <p style={{
              fontSize: 14, color: 'var(--text-secondary)',
              fontFamily: 'JetBrains Mono, monospace',
              letterSpacing: '0.05em',
            }}>
              Describe your problem — AI will find the perfect dataset
            </p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* SPLIT SCREEN CONTAINER*/}
      <div style={{
        width: '100%',
        maxWidth: hasResults ? 1280 : 820,
        position: 'relative',
        zIndex: 1,
        display: hasResults ? 'flex' : 'block',
        gap: hasResults ? 28 : 0,
        alignItems: 'flex-start',
        transition: 'max-width 0.4s ease',
      }}>

        {/*LEFT PANEL — search box (40% in split mode)*/}
        <motion.div
          layout
          style={{ width: hasResults ? '40%' : '100%', flexShrink: 0 }}
          transition={{ duration: 0.4, ease: [0.4, 0, 0.2, 1] }}
        >
          {/* Mini brand badge in split mode */}
          <AnimatePresence>
            {hasResults && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                style={{ marginBottom: 16, display: 'flex', alignItems: 'center', gap: 10 }}
              >
                <motion.span
                  animate={{ rotate: [0, 360] }}
                  transition={{ duration: 20, repeat: Infinity, ease: 'linear' }}
                  style={{ fontSize: 22 }}
                >🧠</motion.span>
                <span style={{
                  fontSize: 15, fontWeight: 700,
                  background: 'var(--gradient-accent)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                }}>
                  Pattern Recognition AI
                </span>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Search box */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="glass"
            style={{ padding: 28, marginBottom: 16 }}
          >
            <div style={{
              fontSize: 11, fontFamily: 'JetBrains Mono, monospace',
              color: 'var(--text-muted)', letterSpacing: '0.1em',
              textTransform: 'uppercase', marginBottom: 10,
            }}>
              Problem Statement
            </div>

            <textarea
              value={problem}
              onChange={e => setProblem(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSearch() }
              }}
              placeholder="e.g. I want to predict whether an employee will leave the company..."
              rows={hasResults ? 5 : 3}
              style={{
                width: '100%', padding: '12px 14px',
                background: 'var(--bg-tertiary)',
                border: '1px solid var(--border)',
                borderRadius: 10, color: 'var(--text-primary)',
                fontFamily: 'Inter, sans-serif', fontSize: 13,
                resize: 'none', outline: 'none',
                lineHeight: 1.6, boxSizing: 'border-box',
              }}
              onFocus={e => e.target.style.borderColor = 'var(--accent)'}
              onBlur={e => e.target.style.borderColor = 'var(--border)'}
            />

            {error && (
              <div style={{ marginTop: 8, fontSize: 12, color: 'var(--neon-red)', fontFamily: 'JetBrains Mono, monospace' }}>
                ⚠ {error}
              </div>
            )}

            <div style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'JetBrains Mono, monospace', marginTop: 12, marginBottom: 8 }}>
              Local: demo datasets · Kaggle: 100,000+ datasets
            </div>

            <div style={{ display: 'flex', gap: 8 }}>
              <motion.button
                whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.97 }}
                onClick={handleSearch} disabled={loading}
                style={{
                  flex: 1, padding: '9px 14px',
                  background: 'transparent',
                  border: '1px solid var(--border-hover)',
                  borderRadius: 9, color: 'var(--text-primary)',
                  fontFamily: 'Inter, sans-serif', fontSize: 12, fontWeight: 500,
                  cursor: loading ? 'not-allowed' : 'pointer',
                  display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
                }}
              >
                {loading ? (
                  <>
                    <div style={{
                      width: 12, height: 12,
                      border: '2px solid rgba(255,255,255,0.2)',
                      borderTop: '2px solid currentColor',
                      borderRadius: '50%',
                      animation: 'spin 0.7s linear infinite',
                    }} />
                    Searching...
                  </>
                ) : <>🔍 Local Search</>}
              </motion.button>

              <motion.button
                whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.97 }}
                onClick={handleKaggleSearch} disabled={kaggleLoading}
                style={{
                  flex: 1, padding: '9px 14px',
                  background: kaggleLoading ? 'var(--bg-tertiary)' : 'var(--gradient-accent)',
                  border: 'none', borderRadius: 9, color: 'white',
                  fontFamily: 'Inter, sans-serif', fontSize: 12, fontWeight: 500,
                  cursor: kaggleLoading ? 'not-allowed' : 'pointer',
                  display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
                  boxShadow: '0 2px 12px rgba(139,92,246,0.3)',
                }}
              >
                {kaggleLoading ? (
                  <>
                    <div style={{
                      width: 12, height: 12,
                      border: '2px solid rgba(255,255,255,0.2)',
                      borderTop: '2px solid white',
                      borderRadius: '50%',
                      animation: 'spin 0.7s linear infinite',
                    }} />
                    Ranking...
                  </>
                ) : <>🌐 Search Kaggle</>}
              </motion.button>
            </div>
          </motion.div>

          {/* Example prompts */}
          <AnimatePresence>
            {!loading && !kaggleLoading && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <div style={{
                  fontSize: 10, fontFamily: 'JetBrains Mono, monospace',
                  color: 'var(--text-muted)', letterSpacing: '0.1em',
                  textTransform: 'uppercase', marginBottom: 10,
                  textAlign: hasResults ? 'left' : 'center',
                }}>
                  Try an example
                </div>
                <div style={{
                  display: 'flex', flexWrap: 'wrap', gap: 8,
                  justifyContent: hasResults ? 'flex-start' : 'center',
                }}>
                  {examples.map((ex, i) => (
                    <motion.button
                      key={i}
                      whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}
                      onClick={() => setProblem(ex)}
                      style={{
                        padding: '6px 14px',
                        background: 'var(--bg-tertiary)',
                        border: '1px solid var(--border)',
                        borderRadius: 20, cursor: 'pointer',
                        fontSize: 11, color: 'var(--text-secondary)',
                        fontFamily: 'Inter, sans-serif',
                      }}
                    >
                      {ex}
                    </motion.button>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>

        {/* RIGHT PANEL — results (slides in from right)*/}
        <AnimatePresence>
          {hasResults && (
            <motion.div
              key="results-panel"
              initial={{ opacity: 0, x: 60 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 40 }}
              transition={{ duration: 0.4, ease: [0.4, 0, 0.2, 1] }}
              style={{ flex: 1, minWidth: 0 }}
            >

              {/*LOCAL RESULTS*/}
              <AnimatePresence>
                {searched && !loading && matches.length > 0 && (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                    <div style={{
                      display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12,
                    }}>
                      <div style={{ fontSize: 11, fontFamily: 'JetBrains Mono, monospace', color: 'var(--text-muted)', letterSpacing: '0.1em', textTransform: 'uppercase' }}>
                        {matches.length} local datasets — ranked by relevance
                      </div>
                      <button
                        onClick={() => { setSearched(false); setMatches([]); setExpandedLocal(null) }}
                        style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: 12, fontFamily: 'JetBrains Mono, monospace' }}
                      >
                        ✕ Clear
                      </button>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                      {matches.map((ds, i) => (
                        <motion.div
                          key={ds.file}
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: i * 0.07 }}
                        >
                          {/* Row — click to expand/collapse info */}
                          <motion.div
                            whileHover={{ scale: 1.005 }}
                            className="glass"
                            style={{
                              padding: '16px 20px', cursor: 'pointer',
                              background: i === 0
                                ? 'linear-gradient(135deg, rgba(52,211,153,0.06), rgba(139,92,246,0.03))'
                                : undefined,
                            }}
                            onClick={() => setExpandedLocal(expandedLocal === ds.file ? null : ds.file)}
                          >
                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 10 }}>
                              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                                <div style={{
                                  width: 30, height: 30, borderRadius: 8,
                                  background: i === 0 ? 'rgba(52,211,153,0.15)' : 'var(--bg-tertiary)',
                                  border: `1px solid ${i === 0 ? 'rgba(52,211,153,0.3)' : 'var(--border)'}`,
                                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                                  fontSize: 12, fontWeight: 700,
                                  color: i === 0 ? 'var(--neon-green)' : 'var(--text-muted)',
                                  fontFamily: 'JetBrains Mono, monospace', flexShrink: 0,
                                }}>
                                  {i === 0 ? '★' : i + 1}
                                </div>
                                <div>
                                  <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 2, display: 'flex', alignItems: 'center', gap: 8 }}>
                                    {ds.name}
                                    {i === 0 && <Badge color="green" style={{ fontSize: 9 }}>BEST MATCH</Badge>}
                                  </div>
                                  <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                                    {ds.category} · hover for details
                                  </div>
                                </div>
                              </div>

                              <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexShrink: 0 }}>
                                <Badge color={catColors[ds.category] || 'blue'}>{ds.category}</Badge>
                                <div style={{ textAlign: 'center' }}>
                                  <div style={{ fontSize: 20, fontWeight: 800, color: getScoreColor(ds.score), fontFamily: 'JetBrains Mono, monospace', lineHeight: 1 }}>
                                    {ds.score}<span style={{ fontSize: 11, fontWeight: 400 }}>%</span>
                                  </div>
                                  <div style={{ fontSize: 8, color: 'var(--text-muted)', fontFamily: 'JetBrains Mono, monospace' }}>MATCH</div>
                                </div>
                                {/* Expand chevron */}
                                <span style={{ fontSize: 12, color: 'var(--text-muted)', transition: 'transform 0.2s', display: 'inline-block', transform: expandedLocal === ds.file ? 'rotate(180deg)' : 'rotate(0deg)' }}>▾</span>
                              </div>
                            </div>

                            {/* Match bar */}
                            <div style={{ marginTop: 10, height: 3, background: 'var(--bg-tertiary)', borderRadius: 2 }}>
                              <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${ds.score}%` }}
                                transition={{ duration: 0.8, delay: i * 0.07 }}
                                style={{ height: '100%', borderRadius: 2, background: getScoreColor(ds.score) }}
                              />
                            </div>
                          </motion.div>

                          {/* Info panel (inline, pushes content down) */}
                          <AnimatePresence>
                            {expandedLocal === ds.file && (
                              <LocalInfoPanel ds={ds} />
                            )}
                          </AnimatePresence>

                          {/* "Click to use" button (shown when expanded) */}
                          <AnimatePresence>
                            {expandedLocal === ds.file && (
                              <motion.div
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                exit={{ opacity: 0 }}
                                transition={{ delay: 0.1 }}
                                style={{ marginTop: 8, display: 'flex', justifyContent: 'flex-end' }}
                              >
                                <motion.button
                                  whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.96 }}
                                  onClick={() => onComplete(ds)}
                                  style={{
                                    padding: '9px 22px',
                                    background: 'var(--gradient-accent)',
                                    border: 'none', borderRadius: 9,
                                    color: 'white', fontFamily: 'Inter, sans-serif',
                                    fontSize: 12, fontWeight: 500, cursor: 'pointer',
                                  }}
                                >
                                  Use This Dataset →
                                </motion.button>
                              </motion.div>
                            )}
                          </AnimatePresence>
                        </motion.div>
                      ))}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* No local result */}
              <AnimatePresence>
                {searched && !loading && matches.length === 0 && (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="glass" style={{ padding: 36, textAlign: 'center' }}>
                    <div style={{ fontSize: 30, marginBottom: 12 }}>🤔</div>
                    <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 8 }}>No strong local matches found</div>
                    <div style={{ fontSize: 12, color: 'var(--text-muted)', fontFamily: 'JetBrains Mono, monospace', marginBottom: 16 }}>
                      Try searching Kaggle for 100,000+ datasets
                    </div>
                    <motion.button
                      whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.97 }}
                      onClick={handleKaggleSearch}
                      style={{
                        padding: '9px 22px', background: 'var(--gradient-accent)',
                        border: 'none', borderRadius: 9, color: 'white',
                        fontFamily: 'Inter, sans-serif', fontSize: 12, fontWeight: 500, cursor: 'pointer',
                      }}
                    >
                      🌐 Search Kaggle Instead
                    </motion.button>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Kaggle loading*/}
              <AnimatePresence>
                {kaggleLoading && (
                  <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="glass" style={{ padding: 36, textAlign: 'center' }}>
                    <div style={{ fontSize: 30, marginBottom: 14 }}>🌐</div>
                    <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 6 }}>Running 2-Stage Ranking Engine...</div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'JetBrains Mono, monospace', marginBottom: 4 }}>Stage 1 · Filtering 50 datasets by quality signals</div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'JetBrains Mono, monospace', marginBottom: 18 }}>Stage 2 · Deep relevance scoring on top candidates</div>
                    <div style={{ height: 3, background: 'var(--bg-tertiary)', borderRadius: 2, overflow: 'hidden' }}>
                      <motion.div
                        animate={{ x: ['-100%', '100%'] }}
                        transition={{ duration: 1.2, repeat: Infinity }}
                        style={{ height: '100%', width: '40%', background: 'linear-gradient(90deg, transparent, var(--neon-violet), transparent)', borderRadius: 2 }}
                      />
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {/*KAGGLE RESULTS */}
              <AnimatePresence>
                {showKaggle && !kaggleLoading && kaggleResults.length > 0 && (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                    <div style={{
                      display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12,
                    }}>
                      <div style={{ fontSize: 11, fontFamily: 'JetBrains Mono, monospace', color: 'var(--text-muted)', letterSpacing: '0.1em', textTransform: 'uppercase', display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span style={{ background: 'var(--gradient-accent)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', fontWeight: 700 }}>
                          🌐 KAGGLE
                        </span>
                        {kaggleResults.length} datasets · 2-stage ranked
                      </div>
                      <button
                        onClick={() => { setShowKaggle(false); setKaggleResults([]); setExpandedKaggle(null) }}
                        style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: 12, fontFamily: 'JetBrains Mono, monospace' }}
                      >
                        ✕ Clear
                      </button>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                      {kaggleResults.map((ds, i) => {
                        const badgeStyle = getRelevanceBadgeStyle(ds.relevance_score || 0)
                        const isExpanded = expandedKaggle === ds.ref
                        return (
                          <motion.div
                            key={ds.ref}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: i * 0.06 }}
                          >
                            {/* Row — click title area to expand */}
                            <div
                              className="glass"
                              style={{
                                padding: '14px 18px',
                                background: i === 0
                                  ? 'linear-gradient(135deg, rgba(139,92,246,0.07), rgba(99,102,241,0.04))'
                                  : undefined,
                              }}
                            >
                              <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', flexWrap: 'wrap', gap: 10 }}>
                                {/* Clickable title + meta */}
                                <div
                                  style={{ flex: 1, minWidth: 0, cursor: 'pointer' }}
                                  onClick={() => setExpandedKaggle(isExpanded ? null : ds.ref)}
                                >
                                  <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 5, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                                    {i === 0 && <Badge color="violet" style={{ fontSize: 9 }}>TOP RESULT</Badge>}
                                    <span style={{
                                      borderBottom: '1px dashed var(--border-hover)',
                                      paddingBottom: 1,
                                      color: isExpanded ? 'var(--accent)' : 'var(--text-primary)',
                                      transition: 'color 0.15s',
                                    }}>
                                      {ds.title}
                                    </span>
                                    <span style={{ fontSize: 11, color: 'var(--text-muted)', transition: 'transform 0.2s', display: 'inline-block', transform: isExpanded ? 'rotate(180deg)' : 'rotate(0deg)' }}>▾</span>
                                  </div>
                                  <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', fontSize: 11, fontFamily: 'JetBrains Mono, monospace', color: 'var(--text-muted)' }}>
                                    <span>📦 {ds.size_mb} MB</span>
                                    <span>⬇️ {(ds.download_count || 0).toLocaleString()}</span>
                                    <span>⭐ {ds.vote_count || 0} votes</span>
                                    <span>📅 {ds.last_updated}</span>
                                  </div>
                                </div>

                                {/* Score badge + download button */}
                                <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexShrink: 0 }}>
                                  {ds.relevance_score != null && (
                                    <div style={{
                                      background: badgeStyle.bg,
                                      border: `1px solid ${badgeStyle.border}`,
                                      borderRadius: 8, padding: '4px 10px', textAlign: 'center',
                                    }}>
                                      <div style={{ fontSize: 16, fontWeight: 800, color: badgeStyle.color, fontFamily: 'JetBrains Mono, monospace', lineHeight: 1 }}>{ds.relevance_score}</div>
                                      <div style={{ fontSize: 8, color: 'var(--text-muted)', fontFamily: 'JetBrains Mono, monospace' }}>SCORE</div>
                                    </div>
                                  )}
                                  <motion.button
                                    whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}
                                    onClick={() => handleKaggleDownload(ds)}
                                    disabled={downloading !== null}
                                    style={{
                                      padding: '8px 16px',
                                      background: downloading === ds.ref ? 'var(--bg-tertiary)' : 'rgba(139,92,246,0.1)',
                                      border: '1px solid rgba(139,92,246,0.3)',
                                      borderRadius: 9,
                                      cursor: downloading !== null ? 'not-allowed' : 'pointer',
                                      color: 'var(--neon-violet)',
                                      fontFamily: 'Inter, sans-serif', fontSize: 11, fontWeight: 500,
                                      display: 'flex', alignItems: 'center', gap: 6, whiteSpace: 'nowrap',
                                    }}
                                  >
                                    {downloading === ds.ref ? (
                                      <>
                                        <div style={{
                                          width: 11, height: 11,
                                          border: '2px solid rgba(139,92,246,0.2)',
                                          borderTop: '2px solid var(--neon-violet)',
                                          borderRadius: '50%',
                                          animation: 'spin 0.7s linear infinite',
                                        }} />
                                        Downloading...
                                      </>
                                    ) : <>⬇️ Download &amp; Use</>}
                                  </motion.button>
                                </div>
                              </div>
                            </div>

                            {/* ── Inline info panel (pushes next item down) ── */}
                            <AnimatePresence>
                              {isExpanded && (
                                <KaggleInfoPanel ds={ds} />
                              )}
                            </AnimatePresence>
                          </motion.div>
                        )
                      })}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Kaggle no results */}
              <AnimatePresence>
                {showKaggle && !kaggleLoading && kaggleResults.length === 0 && (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="glass" style={{ padding: 36, textAlign: 'center' }}>
                    <div style={{ fontSize: 30, marginBottom: 12 }}>🌐</div>
                    <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 8 }}>No Kaggle datasets found</div>
                    <div style={{ fontSize: 12, color: 'var(--text-muted)', fontFamily: 'JetBrains Mono, monospace' }}>
                      Try different keywords in your problem statement
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

            </motion.div>
          )}
        </AnimatePresence>

      </div>
    </div>
  )
}