import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { findDatasets } from '../api/client.js'
import useAppStore from '../store/appStore'
import Badge from '../components/ui/Badge'

export default function LandingPage({ onComplete }) {
  const [problem,   setProblem]   = useState('')
  const [loading,   setLoading]   = useState(false)
  const [matches,   setMatches]   = useState([])
  const [searched,  setSearched]  = useState(false)
  const [hoveredDs, setHoveredDs] = useState(null)
  const [error,     setError]     = useState('')

  const { setFoundDatasets, setProblemStatement } = useAppStore()

  const examples = [
    "Predict employee attrition in a company",
    "Analyze student academic performance",
    "Detect stroke risk from health data",
    "Forecast personal finance spending patterns",
  ]

  async function handleSearch() {
    if (!problem.trim()) return setError('Please enter a problem statement')
    setError('')
    setLoading(true)
    setMatches([])
    try {
      const res = await findDatasets(problem)
      if (res.error) return setError(res.error)
      setMatches(res.matches || [])
      setSearched(true)
      setProblemStatement(problem)
      setFoundDatasets(res.matches || [])
    } catch {
      setError('Search failed. Make sure Ollama is running.')
    }
    setLoading(false)
  }

  function getScoreColor(score) {
    if (score >= 80) return 'var(--neon-green)'
    if (score >= 60) return 'var(--neon-amber)'
    return 'var(--neon-red)'
  }

  function getScoreBadge(score) {
    if (score >= 80) return 'green'
    if (score >= 60) return 'amber'
    return 'red'
  }

  const catColors = {
    'Finance':         'green',
    'Healthcare':      'red',
    'Education':       'blue',
    'Human Resources': 'violet',
    'Sports':          'cyan',
    'Technology':      'cyan',
    'Retail':          'amber',
  }

  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '40px 32px',
      position: 'relative',
    }}>

      {/* Animated background blobs */}
      <div style={{
        position: 'fixed', inset: 0, zIndex: 0,
        overflow: 'hidden', pointerEvents: 'none'
      }}>
        <motion.div
          animate={{ x: [0, 30, 0], y: [0, -20, 0] }}
          transition={{ duration: 8, repeat: Infinity }}
          style={{
            position: 'absolute', top: '20%', left: '10%',
            width: 400, height: 400, borderRadius: '50%',
            background: 'radial-gradient(circle, rgba(0,212,255,0.06), transparent)',
            filter: 'blur(40px)'
          }}
        />
        <motion.div
          animate={{ x: [0, -20, 0], y: [0, 30, 0] }}
          transition={{ duration: 10, repeat: Infinity }}
          style={{
            position: 'absolute', bottom: '20%', right: '10%',
            width: 500, height: 500, borderRadius: '50%',
            background: 'radial-gradient(circle, rgba(124,58,237,0.06), transparent)',
            filter: 'blur(40px)'
          }}
        />
      </div>

      <div style={{
        width: '100%', maxWidth: 800,
        position: 'relative', zIndex: 1
      }}>

        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          style={{ textAlign: 'center', marginBottom: 48 }}
        >
          <motion.div
            animate={{ rotate: [0, 360] }}
            transition={{ duration: 20, repeat: Infinity, ease: 'linear' }}
            style={{ fontSize: 56, marginBottom: 20 }}
          >
            🧠
          </motion.div>
          <h1 style={{
            fontSize: 36, fontWeight: 900,
            letterSpacing: '0.02em', marginBottom: 12,
            background: 'linear-gradient(135deg, var(--accent), var(--neon-violet))',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}>
            AI Cognitive Pattern Recognition
          </h1>
          <p style={{
            fontSize: 15, color: 'var(--text-secondary)',
            fontFamily: 'JetBrains Mono, monospace',
            letterSpacing: '0.05em'
          }}>
            Describe your problem — AI will find the perfect dataset
          </p>
        </motion.div>

        {/* Search box */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="glass"
          style={{ padding: 32, marginBottom: 24 }}
        >
          <div style={{
            fontSize: 12, fontFamily: 'JetBrains Mono, monospace',
            color: 'var(--text-muted)', letterSpacing: '0.1em',
            textTransform: 'uppercase', marginBottom: 12
          }}>
            Problem Statement
          </div>

          <textarea
            value={problem}
            onChange={e => setProblem(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && !e.shiftKey && handleSearch()}
            placeholder="e.g. I want to predict whether an employee will leave the company based on HR data..."
            rows={3}
            style={{
              width: '100%', padding: '14px 16px',
              background: 'var(--bg-tertiary)',
              border: '1px solid var(--border-hover)',
              borderRadius: 10, color: 'var(--text-primary)',
              fontFamily: 'Inter, sans-serif', fontSize: 14,
              resize: 'none', outline: 'none',
              transition: 'border-color 0.2s',
              lineHeight: 1.6,
            }}
            onFocus={e => e.target.style.borderColor = 'var(--accent)'}
            onBlur={e => e.target.style.borderColor = 'var(--border-hover)'}
          />

          {error && (
            <div style={{
              marginTop: 8, fontSize: 12,
              color: 'var(--neon-red)',
              fontFamily: 'JetBrains Mono, monospace'
            }}>
              ⚠ {error}
            </div>
          )}

          <div style={{
            display: 'flex', alignItems: 'center',
            justifyContent: 'space-between',
            marginTop: 16, flexWrap: 'wrap', gap: 12
          }}>
            <div style={{
              fontSize: 12, color: 'var(--text-muted)',
              fontFamily: 'JetBrains Mono, monospace'
            }}>
              Press Enter to search
            </div>
            <motion.button
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
              onClick={handleSearch}
              disabled={loading}
              style={{
                padding: '12px 32px',
                background: loading
                  ? 'var(--bg-tertiary)'
                  : 'linear-gradient(135deg, var(--accent), var(--neon-violet))',
                border: 'none', borderRadius: 10,
                color: 'white', fontFamily: 'Inter, sans-serif',
                fontSize: 14, fontWeight: 600,
                cursor: loading ? 'not-allowed' : 'pointer',
                display: 'flex', alignItems: 'center', gap: 10,
                transition: 'all 0.2s',
              }}
            >
              {loading ? (
                <>
                  <div style={{
                    width: 16, height: 16,
                    border: '2px solid rgba(255,255,255,0.2)',
                    borderTop: '2px solid white',
                    borderRadius: '50%',
                    animation: 'spin 0.7s linear infinite'
                  }}/>
                  AI is searching...
                </>
              ) : (
                <>🔍 Find Best Datasets</>
              )}
            </motion.button>
          </div>
        </motion.div>

        {/* Example prompts */}
        <AnimatePresence>
          {!searched && !loading && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              style={{ marginBottom: 24 }}
            >
              <div style={{
                fontSize: 11, fontFamily: 'JetBrains Mono, monospace',
                color: 'var(--text-muted)', letterSpacing: '0.1em',
                textTransform: 'uppercase', marginBottom: 12,
                textAlign: 'center'
              }}>
                Try an example
              </div>
              <div style={{
                display: 'flex', flexWrap: 'wrap',
                gap: 10, justifyContent: 'center'
              }}>
                {examples.map((ex, i) => (
                  <motion.button
                    key={i}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => setProblem(ex)}
                    style={{
                      padding: '8px 16px',
                      background: 'var(--bg-tertiary)',
                      border: '1px solid var(--border)',
                      borderRadius: 20, cursor: 'pointer',
                      fontSize: 12, color: 'var(--text-secondary)',
                      fontFamily: 'Inter, sans-serif',
                      transition: 'all 0.2s',
                    }}
                  >
                    {ex}
                  </motion.button>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Loading state */}
        <AnimatePresence>
          {loading && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="glass"
              style={{ padding: 32, textAlign: 'center' }}
            >
              <div style={{ fontSize: 32, marginBottom: 16 }}>🔍</div>
              <div style={{
                fontSize: 15, fontWeight: 600, marginBottom: 8
              }}>
                Phi-3 is analyzing all datasets...
              </div>
              <div style={{
                fontSize: 12, color: 'var(--text-muted)',
                fontFamily: 'JetBrains Mono, monospace'
              }}>
                Reading columns, sample data and scoring relevance
              </div>
              <div style={{
                marginTop: 20, height: 4,
                background: 'var(--bg-tertiary)',
                borderRadius: 2, overflow: 'hidden'
              }}>
                <motion.div
                  animate={{ x: ['-100%', '100%'] }}
                  transition={{ duration: 1.5, repeat: Infinity }}
                  style={{
                    height: '100%', width: '40%',
                    background: 'linear-gradient(90deg, transparent, var(--accent), transparent)',
                    borderRadius: 2
                  }}
                />
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Results */}
        <AnimatePresence>
          {searched && !loading && matches.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <div style={{
                display: 'flex', alignItems: 'center',
                justifyContent: 'space-between', marginBottom: 16
              }}>
                <div style={{
                  fontSize: 12, fontFamily: 'JetBrains Mono, monospace',
                  color: 'var(--text-muted)', letterSpacing: '0.1em',
                  textTransform: 'uppercase'
                }}>
                  {matches.length} datasets found — ranked by relevance
                </div>
                <button
                  onClick={() => { setSearched(false); setMatches([]); setProblem('') }}
                  style={{
                    background: 'transparent', border: 'none',
                    color: 'var(--text-muted)', cursor: 'pointer',
                    fontSize: 12, fontFamily: 'JetBrains Mono, monospace'
                  }}
                >
                  ← Search again
                </button>
              </div>

              <div style={{
                display: 'flex', flexDirection: 'column', gap: 12
              }}>
                {matches.map((ds, i) => (
                  <motion.div
                    key={ds.file}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.08 }}
                    onMouseEnter={() => setHoveredDs(ds.file)}
                    onMouseLeave={() => setHoveredDs(null)}
                    style={{ position: 'relative' }}
                  >
                    {/* Main card */}
                    <motion.div
                      whileHover={{ scale: 1.01 }}
                      className="glass"
                      style={{
                        padding: '20px 24px',
                        cursor: 'pointer',
                        borderColor: hoveredDs === ds.file
                          ? 'var(--border-hover)' : 'var(--border)',
                        background: i === 0
                          ? 'linear-gradient(135deg, rgba(16,185,129,0.06), rgba(0,212,255,0.03))'
                          : 'var(--bg-glass)',
                      }}
                      onClick={() => onComplete(ds)}
                    >
                      <div style={{
                        display: 'flex', alignItems: 'center',
                        justifyContent: 'space-between', flexWrap: 'wrap', gap: 12
                      }}>
                        <div style={{
                          display: 'flex', alignItems: 'center', gap: 12
                        }}>
                          {/* Rank */}
                          <div style={{
                            width: 32, height: 32, borderRadius: 8,
                            background: i === 0
                              ? 'rgba(16,185,129,0.2)'
                              : 'var(--bg-tertiary)',
                            border: `1px solid ${i === 0 ? 'rgba(16,185,129,0.4)' : 'var(--border)'}`,
                            display: 'flex', alignItems: 'center',
                            justifyContent: 'center',
                            fontSize: 13, fontWeight: 700,
                            color: i === 0 ? 'var(--neon-green)' : 'var(--text-muted)',
                            fontFamily: 'JetBrains Mono, monospace',
                            flexShrink: 0
                          }}>
                            {i === 0 ? '★' : i + 1}
                          </div>

                          <div>
                            <div style={{
                              fontSize: 15, fontWeight: 700,
                              marginBottom: 4, display: 'flex',
                              alignItems: 'center', gap: 8
                            }}>
                              {ds.name}
                              {i === 0 && (
                                <Badge color="green" style={{ fontSize: 10 }}>
                                  BEST MATCH
                                </Badge>
                              )}
                            </div>
                            <div style={{
                              fontSize: 12, color: 'var(--text-muted)',
                              lineHeight: 1.5
                            }}>
                              {ds.reason}
                            </div>
                          </div>
                        </div>

                        <div style={{
                          display: 'flex', alignItems: 'center', gap: 12,
                          flexShrink: 0
                        }}>
                          <Badge color={catColors[ds.category] || 'blue'}>
                            {ds.category}
                          </Badge>
                          {/* Score ring */}
                          <div style={{
                            display: 'flex', flexDirection: 'column',
                            alignItems: 'center', gap: 2
                          }}>
                            <div style={{
                              fontSize: 22, fontWeight: 900,
                              color: getScoreColor(ds.score),
                              fontFamily: 'JetBrains Mono, monospace',
                              lineHeight: 1
                            }}>
                              {ds.score}
                              <span style={{ fontSize: 12, fontWeight: 400 }}>%</span>
                            </div>
                            <div style={{
                              fontSize: 9, color: 'var(--text-muted)',
                              fontFamily: 'JetBrains Mono, monospace',
                              letterSpacing: '0.05em'
                            }}>
                              MATCH
                            </div>
                          </div>
                        </div>
                      </div>

                      {/* Progress bar */}
                      <div style={{
                        marginTop: 12, height: 3,
                        background: 'var(--bg-tertiary)', borderRadius: 2
                      }}>
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${ds.score}%` }}
                          transition={{ duration: 0.8, delay: i * 0.08 }}
                          style={{
                            height: '100%', borderRadius: 2,
                            background: getScoreColor(ds.score),
                            boxShadow: `0 0 8px ${getScoreColor(ds.score)}`
                          }}
                        />
                      </div>
                    </motion.div>

                    {/* Hover insight card */}
                    <AnimatePresence>
                      {hoveredDs === ds.file && (
                        <motion.div
                          initial={{ opacity: 0, y: 8, scale: 0.97 }}
                          animate={{ opacity: 1, y: 0, scale: 1 }}
                          exit={{ opacity: 0, y: 8, scale: 0.97 }}
                          transition={{ duration: 0.15 }}
                          style={{
                            position: 'absolute',
                            top: '100%', left: 0, right: 0,
                            zIndex: 50, marginTop: 4,
                            background: 'var(--bg-secondary)',
                            border: '1px solid var(--border-hover)',
                            borderRadius: 12, padding: 20,
                            boxShadow: '0 20px 60px rgba(0,0,0,0.4)',
                          }}
                        >
                          <div style={{
                            display: 'grid',
                            gridTemplateColumns: '1fr 1fr 1fr',
                            gap: 16, marginBottom: 16
                          }}>
                            {[
                              { label: 'Total Rows', value: ds.total_rows?.toLocaleString() },
                              { label: 'Columns', value: ds.total_columns },
                              { label: 'Numerical', value: ds.numerical_cols?.length },
                            ].map(stat => (
                              <div key={stat.label} style={{
                                background: 'var(--bg-tertiary)',
                                borderRadius: 8, padding: '10px 14px',
                                border: '1px solid var(--border)'
                              }}>
                                <div style={{
                                  fontSize: 10,
                                  fontFamily: 'JetBrains Mono, monospace',
                                  color: 'var(--text-muted)',
                                  marginBottom: 4
                                }}>
                                  {stat.label}
                                </div>
                                <div style={{
                                  fontSize: 18, fontWeight: 700,
                                  color: 'var(--text-primary)'
                                }}>
                                  {stat.value}
                                </div>
                              </div>
                            ))}
                          </div>

                          {/* Key columns */}
                          {ds.key_columns?.length > 0 && (
                            <div style={{ marginBottom: 12 }}>
                              <div style={{
                                fontSize: 10,
                                fontFamily: 'JetBrains Mono, monospace',
                                color: 'var(--text-muted)',
                                letterSpacing: '0.08em',
                                marginBottom: 8
                              }}>
                                KEY COLUMNS FOR YOUR PROBLEM
                              </div>
                              <div style={{
                                display: 'flex', flexWrap: 'wrap', gap: 6
                              }}>
                                {ds.key_columns.map(col => (
                                  <Badge key={col} color="cyan"
                                    style={{ fontSize: 11 }}>
                                    {col}
                                  </Badge>
                                ))}
                              </div>
                            </div>
                          )}

                          <div style={{
                            padding: '10px 14px',
                            background: 'rgba(0,212,255,0.05)',
                            borderLeft: '2px solid var(--accent)',
                            borderRadius: '0 6px 6px 0',
                            fontSize: 12,
                            color: 'var(--text-secondary)',
                            lineHeight: 1.6
                          }}>
                            Click to load this dataset and start analysis →
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* No results */}
        <AnimatePresence>
          {searched && !loading && matches.length === 0 && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="glass"
              style={{ padding: 40, textAlign: 'center' }}
            >
              <div style={{ fontSize: 32, marginBottom: 12 }}>🤔</div>
              <div style={{ fontSize: 15, fontWeight: 600, marginBottom: 8 }}>
                No strong matches found
              </div>
              <div style={{
                fontSize: 13, color: 'var(--text-muted)',
                fontFamily: 'JetBrains Mono, monospace'
              }}>
                Try rephrasing your problem statement
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}