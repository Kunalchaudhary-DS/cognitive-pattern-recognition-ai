import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { getPrediction, getSamplePredictions ,getEncodingMaps} from '../../api/client.js'
import Badge from './Badge'
import Spinner from './Spinner'

export default function PredictionEngine({ featureNames = [], targetColumn, problemType }) {
  const [formValues,    setFormValues]    = useState({})
  const [predicting,    setPredicting]    = useState(false)
  const [result,        setResult]        = useState(null)
  const [samples,       setSamples]       = useState([])
  const [samplesLoading,setSamplesLoading]= useState(false)
  const [samplesError,  setSamplesError]  = useState('')
  const [error,         setError]         = useState('')
  const [encodingMaps, setEncodingMaps] = useState({})

  // Show only top 8 most important features in form
  const displayFeatures = featureNames.slice(0, 8)

  useEffect(() => {
    // Init form with empty values
    const init = {}
    displayFeatures.forEach(f => { init[f] = '' })
    setFormValues(init)
    // Load sample predictions
    loadSamples()
  }, [featureNames])


  useEffect(() => {
  const init = {}
  displayFeatures.forEach(f => { init[f] = '' })
  setFormValues(init)
  loadSamples()
  loadEncodingMaps()
}, [featureNames])

async function loadEncodingMaps() {
  try {
    const res = await getEncodingMaps()
    setEncodingMaps(res.encoding_maps || {})
  } catch {}
}

  async function loadSamples() {
    setSamplesLoading(true)
    setSamplesError('')
    try {
      const res = await getSamplePredictions()
      if (res.error) {
        setSamplesError(res.error)
        setSamples([])
      } else {
        setSamples(res.samples || [])
      }
    } catch (e) {
      setSamplesError('Failed to load sample predictions')
    }
    setSamplesLoading(false)
  }

  async function handlePredict() {
    setError('')
    const filled = Object.values(formValues).some(v => v !== '')
    if (!filled) return setError('Please enter at least one feature value')

    setPredicting(true)
    setResult(null)
    try {
      const res = await getPrediction(formValues)
      if (res.error) return setError(res.error)
      setResult(res)
    } catch { setError('Prediction failed') }
    setPredicting(false)
  }

  const getAccuracyColor = (match) => match ? 'var(--neon-green)' : 'var(--neon-amber)'

  return (
    <div>
      {/* LIVE PREDICTION FORM */}
      <div style={{
        background: 'var(--bg-tertiary)',
        border: '1px solid var(--border)',
        borderRadius: 12, padding: 24, marginBottom: 20
      }}>
        <div style={{
          fontSize: 12, fontFamily: 'JetBrains Mono, monospace',
          color: 'var(--text-muted)', letterSpacing: '0.08em',
          textTransform: 'uppercase', marginBottom: 16
        }}>
          Enter Feature Values
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))',
          gap: 12, marginBottom: 20
        }}>
            {displayFeatures.map(feature => {
            const isCategorical = !!encodingMaps[feature]
            const validValues   = isCategorical
                ? Object.keys(encodingMaps[feature]).slice(0, 5)
                : null

            return (
                <div key={feature}>
                <div style={{
                    fontSize: 10, fontFamily: 'JetBrains Mono, monospace',
                    color: 'var(--text-muted)', marginBottom: 5,
                    letterSpacing: '0.05em',
                    whiteSpace: 'nowrap', overflow: 'hidden',
                    textOverflow: 'ellipsis'
                }}>
                    {feature}
                    {isCategorical && (
                    <span style={{
                        marginLeft: 6, color: 'var(--neon-violet)',
                        fontSize: 9
                    }}>
                        TEXT
                    </span>
                    )}
                </div>
                <input
                    type={isCategorical ? 'text' : 'number'}
                    placeholder={isCategorical
                    ? validValues?.join(' / ')
                    : '0'
                    }
                    value={formValues[feature] || ''}
                    onChange={e => setFormValues(prev => ({
                    ...prev, [feature]: e.target.value
                    }))}
                    style={{
                    width: '100%', padding: '9px 12px',
                    background: 'var(--bg-secondary)',
                    border: `1px solid ${isCategorical
                        ? 'rgba(139,92,246,0.25)'
                        : 'var(--border)'}`,
                    borderRadius: 8, color: 'var(--text-primary)',
                    fontFamily: 'JetBrains Mono, monospace',
                    fontSize: 12, outline: 'none',
                    transition: 'border-color 0.2s'
                    }}
                    onFocus={e => e.target.style.borderColor = 'var(--accent)'}
                    onBlur={e => e.target.style.borderColor = isCategorical
                    ? 'rgba(139,92,246,0.25)'
                    : 'var(--border)'}
                />
                {isCategorical && validValues && (
                    <div style={{
                    fontSize: 9, color: 'var(--text-muted)',
                    fontFamily: 'JetBrains Mono, monospace',
                    marginTop: 4, lineHeight: 1.4
                    }}>
                    Options: {validValues.join(', ')}
                    {Object.keys(encodingMaps[feature]).length > 5 && '...'}
                    </div>
                )}
                </div>
            )
            })}
        </div>

        {error && (
          <div style={{
            fontSize: 12, color: 'var(--neon-red)',
            fontFamily: 'JetBrains Mono, monospace',
            marginBottom: 12
          }}>
            ⚠ {error}
          </div>
        )}

        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.97 }}
          onClick={handlePredict}
          disabled={predicting}
          style={{
            padding: '11px 28px',
            background: 'var(--gradient-accent)',
            border: 'none', borderRadius: 10,
            color: 'white', fontFamily: 'Inter, sans-serif',
            fontSize: 13, fontWeight: 500,
            cursor: predicting ? 'not-allowed' : 'pointer',
            display: 'flex', alignItems: 'center', gap: 10,
            boxShadow: '0 2px 12px rgba(139,92,246,0.3)'
          }}
        >
          {predicting ? (
            <><Spinner size={14} color="white"/> Predicting...</>
          ) : (
            <>🎯 Run Prediction</>
          )}
        </motion.button>
      </div>

      {/* PREDICTION RESULT */}
      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            style={{
              background: 'linear-gradient(135deg, rgba(139,92,246,0.08), rgba(99,102,241,0.05))',
              border: '1px solid rgba(139,92,246,0.25)',
              borderRadius: 12, padding: 24, marginBottom: 20
            }}
          >
            <div style={{
              display: 'flex', alignItems: 'center',
              justifyContent: 'space-between', marginBottom: 16,
              flexWrap: 'wrap', gap: 12
            }}>
              <div>
                <div style={{
                  fontSize: 11, fontFamily: 'JetBrains Mono, monospace',
                  color: 'var(--text-muted)', marginBottom: 6,
                  letterSpacing: '0.08em'
                }}>
                  PREDICTED {targetColumn?.toUpperCase()}
                </div>
                <div style={{
                  fontSize: 36, fontWeight: 700,
                  color: 'var(--neon-violet)', lineHeight: 1
                }}>
                  {result.prediction}
                </div>

                {/* Raw model output — shown only when correction was applied */}
                {result.was_corrected && result.raw_prediction != null && (
                  <div style={{
                    marginTop: 6,
                    fontSize: 12, fontFamily: 'JetBrains Mono, monospace',
                    color: 'rgba(251,191,36,0.8)',
                    display: 'flex', alignItems: 'center', gap: 6
                  }}>
                    <span style={{ fontSize: 10, opacity: 0.7 }}>RAW MODEL OUTPUT:</span>
                    <span style={{ textDecoration: 'line-through', opacity: 0.6 }}>
                      {result.raw_prediction}
                    </span>
                    <span style={{ fontSize: 10 }}>→ corrected</span>
                  </div>
                )}
              </div>
              <div style={{ textAlign: 'right' }}>
                <Badge color="violet" style={{ marginBottom: 8, display: 'block' }}>
                  {result.model_used}
                </Badge>
                <Badge color={result.problem_type === 'classification' ? 'blue' : 'cyan'}>
                  {result.problem_type}
                </Badge>
              </div>
            </div>

            {/* Constraint Correction Banner */}
            {result.was_corrected && result.constraints_applied?.length > 0 && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                style={{
                  marginBottom: 12,
                  padding: '12px 16px',
                  background: 'rgba(251,191,36,0.06)',
                  border: '1px solid rgba(251,191,36,0.25)',
                  borderRadius: 8,
                  borderLeft: '3px solid rgba(251,191,36,0.7)',
                }}
              >
                <div style={{
                  fontSize: 10, fontFamily: 'JetBrains Mono, monospace',
                  color: 'rgba(251,191,36,0.9)',
                  letterSpacing: '0.08em', marginBottom: 8,
                  display: 'flex', alignItems: 'center', gap: 6
                }}>
                  <span>⚡</span>
                  <span>SEMANTIC INTERCEPTOR ACTIVE — PREDICTION CORRECTED</span>
                </div>
                {result.constraints_applied.map((msg, i) => (
                  <div key={i} style={{
                    fontSize: 11, color: 'rgba(251,191,36,0.75)',
                    fontFamily: 'JetBrains Mono, monospace',
                    lineHeight: 1.6,
                    paddingLeft: 8,
                    borderLeft: '1px solid rgba(251,191,36,0.2)',
                    marginBottom: 4
                  }}>
                    › {msg}
                  </div>
                ))}
              </motion.div>
            )}

            {/* Soft Domain Warnings */}
            {result.soft_warnings?.length > 0 && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                style={{
                  marginBottom: 12,
                  padding: '10px 14px',
                  background: 'rgba(99,102,241,0.05)',
                  border: '1px solid rgba(99,102,241,0.15)',
                  borderRadius: 8,
                }}
              >
                <div style={{
                  fontSize: 10, fontFamily: 'JetBrains Mono, monospace',
                  color: 'rgba(99,102,241,0.7)', letterSpacing: '0.08em',
                  marginBottom: 6
                }}>
                  DOMAIN NOTICES
                </div>
                {result.soft_warnings.map((msg, i) => (
                  <div key={i} style={{
                    fontSize: 11, color: 'var(--text-muted)',
                    fontFamily: 'JetBrains Mono, monospace',
                    lineHeight: 1.6
                  }}>
                    {msg}
                  </div>
                ))}
              </motion.div>
            )}

            {result.explanation && (
              <div style={{
                padding: '12px 16px',
                background: 'rgba(0,0,0,0.2)',
                borderRadius: 8, borderLeft: '2px solid var(--accent)',
                fontSize: 13, color: 'var(--text-secondary)',
                lineHeight: 1.7
              }}>
                <span style={{
                  fontSize: 10, fontFamily: 'JetBrains Mono, monospace',
                  color: 'var(--accent)', letterSpacing: '0.08em',
                  display: 'block', marginBottom: 6
                }}>
                  CPR AI INTERPRETATION
                </span>
                {result.explanation}
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* SAMPLE PREDICTIONS */}
      <div style={{
        background: 'var(--bg-tertiary)',
        border: '1px solid var(--border)',
        borderRadius: 12, padding: 24
      }}>
        <div style={{
          fontSize: 12, fontFamily: 'JetBrains Mono, monospace',
          color: 'var(--text-muted)', letterSpacing: '0.08em',
          textTransform: 'uppercase', marginBottom: 16,
          display: 'flex', alignItems: 'center',
          justifyContent: 'space-between'
        }}>
          <span>Sample Row Predictions</span>
          <motion.button
            whileHover={{ scale: 1.05 }}
            onClick={loadSamples}
            style={{
              background: 'transparent',
              border: '1px solid var(--border)',
              borderRadius: 6, padding: '4px 10px',
              color: 'var(--text-muted)', cursor: 'pointer',
              fontSize: 10, fontFamily: 'JetBrains Mono, monospace'
            }}
          >
            ↻ Refresh
          </motion.button>
        </div>

        {samplesLoading ? (
          <div style={{
            display: 'flex', alignItems: 'center', gap: 10,
            color: 'var(--text-muted)', fontSize: 12,
            fontFamily: 'JetBrains Mono, monospace'
          }}>
            <Spinner size={14}/> Loading samples...
          </div>
        ) : samplesError ? (
          <div style={{
            color: 'var(--neon-amber)', fontSize: 12,
            fontFamily: 'JetBrains Mono, monospace',
            padding: '8px 12px',
            background: 'rgba(251,191,36,0.06)',
            borderRadius: 6,
            border: '1px solid rgba(251,191,36,0.2)'
          }}>
            ⚠ {samplesError}
          </div>
        ) : samples.length > 0 ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {samples.map((row, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.06 }}
                style={{
                  display: 'flex', alignItems: 'center',
                  gap: 12, padding: '12px 16px',
                  background: 'var(--bg-secondary)',
                  borderRadius: 8,
                  border: `1px solid ${row.match ? 'rgba(52,211,153,0.15)' : 'rgba(251,191,36,0.15)'}`,
                }}
              >
                <div style={{
                  width: 28, height: 28, borderRadius: 7,
                  background: row.match
                    ? 'rgba(52,211,153,0.1)' : 'rgba(251,191,36,0.1)',
                  display: 'flex', alignItems: 'center',
                  justifyContent: 'center', fontSize: 12,
                  flexShrink: 0
                }}>
                  {row.match ? '✓' : '≈'}
                </div>

                <div style={{ flex: 1 }}>
                  <div style={{
                    fontSize: 11, fontFamily: 'JetBrains Mono, monospace',
                    color: 'var(--text-muted)', marginBottom: 3
                  }}>
                    Row {row.row}
                  </div>
                  <div style={{
                    display: 'flex', gap: 16, flexWrap: 'wrap'
                  }}>
                    <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                      Actual: <strong style={{ color: 'var(--text-primary)' }}>
                        {row.actual}
                      </strong>
                    </span>
                    <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                      Predicted: <strong style={{
                        color: getAccuracyColor(row.match)
                      }}>
                        {row.predicted}
                      </strong>
                    </span>
                    {row.error > 0 && (
                      <span style={{
                        fontSize: 11,
                        fontFamily: 'JetBrains Mono, monospace',
                        color: 'var(--text-muted)'
                      }}>
                        Error: {row.error}
                      </span>
                    )}
                  </div>
                </div>

                <div style={{
                  fontSize: 10,
                  fontFamily: 'JetBrains Mono, monospace',
                  color: getAccuracyColor(row.match),
                  fontWeight: 600
                }}>
                  {row.match ? 'ACCURATE' : 'APPROX'}
                </div>
              </motion.div>
            ))}
          </div>
        ) : (
          <div style={{
            color: 'var(--text-muted)', fontSize: 12,
            fontFamily: 'JetBrains Mono, monospace'
          }}>
            No sample predictions available
          </div>
        )}
      </div>
    </div>
  )
}