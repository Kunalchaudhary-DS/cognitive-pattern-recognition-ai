import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import toast from 'react-hot-toast'
import useAppStore from '../store/appStore'
import { getAITrainingExplanation } from '../api/client'
import Card from '../components/ui/Card'
import Badge from '../components/ui/Badge'
import AIPanel from '../components/ui/AIPanel'
import PlotlyChart from '../components/charts/PlotlyChart'
import StepFlow from '../components/ui/StepFlow'

export default function TrainingPage() {
  const { trainingResults, trained } = useAppStore()
  const { aiTrainingText, setAiTrainingText } = useAppStore()
  const [aiLoading, setAiLoading] = useState(false)

  useEffect(() => {
    if (trained && trainingResults) fetchAI()
  }, [trained, trainingResults])

  async function fetchAI() {
    if (aiTrainingText) return
    setAiLoading(true)
    try {
      const res = await getAITrainingExplanation()
      setAiTrainingText(res.explanation || '')
    } catch { setAiTrainingText('') }
    setAiLoading(false)
  }

  if (!trained || !trainingResults) {
    return (
      <div style={{ padding: 32 }}>
        <StepFlow/>
        <div style={{
          textAlign: 'center', padding: '80px 32px',
          color: 'var(--text-muted)',
          fontFamily: 'JetBrains Mono, monospace', fontSize: 13
        }}>
          Complete data processing first, then train your models
        </div>
      </div>
    )
  }

  const ignore  = ['BestModel', 'ProblemType', 'ConfusionMatrix', 'PrimaryMetric', 'Imbalanced']
  const best    = trainingResults.BestModel
  const pType   = trainingResults.ProblemType
  const metric  = trainingResults.PrimaryMetric
              || (pType === 'regression' ? 'CV_R2_Mean' : 'CV_Accuracy_Mean')
  const imbalanced = trainingResults.Imbalanced || false

  // Human-readable label for the primary metric
  const mLabel = pType === 'regression'
    ? 'R² Score'
    : (imbalanced ? 'F1-Macro (imbalanced)' : 'Accuracy')

  const models  = Object.entries(trainingResults)
    .filter(([k]) => !ignore.includes(k))
    .sort((a, b) => (b[1][metric] || 0) - (a[1][metric] || 0))

  const maxScore  = Math.max(...models.map(([, v]) => v[metric] || 0))
  const metricKeys = models[0] ? Object.keys(models[0][1]).filter(k => !ignore.includes(k)) : []

  const bestScore = trainingResults[best]?.[metric] || 0

  return (
    <div style={{ padding: 32, maxWidth: 1400, margin: '0 auto' }}>
      <StepFlow/>

      {/* Header */}
      <div style={{
        display: 'flex', alignItems: 'center',
        justifyContent: 'space-between', marginBottom: 28, flexWrap: 'wrap', gap: 16
      }}>
        <div>
          <h1 style={{ fontSize: 26, fontWeight: 800, marginBottom: 6 }}>
            Model Training Results
          </h1>
          <p style={{ fontSize: 13, color: 'var(--text-muted)',
            fontFamily: 'JetBrains Mono, monospace' }}>
            AutoML evaluated {models.length} models · Metric: {mLabel}
          </p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <Badge color="green" style={{ fontSize: 13, padding: '8px 16px' }}>
            ⭐ Best: {best}
          </Badge>
          <Badge color={pType === 'regression' ? 'blue' : 'violet'}
            style={{ fontSize: 13, padding: '8px 16px' }}>
            {pType}
          </Badge>
        </div>
      </div>

      {/* Best model highlight */}
      <motion.div
        initial={{ opacity: 0, scale: 0.98 }}
        animate={{ opacity: 1, scale: 1 }}
        style={{
          background: 'linear-gradient(135deg, rgba(16,185,129,0.08), rgba(0,212,255,0.05))',
          border: '1px solid rgba(16,185,129,0.3)',
          borderRadius: 16, padding: 24, marginBottom: 20,
          display: 'flex', alignItems: 'center',
          justifyContent: 'space-between', flexWrap: 'wrap', gap: 16
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <div style={{
            width: 56, height: 56, borderRadius: 14,
            background: 'rgba(16,185,129,0.15)',
            border: '1px solid rgba(16,185,129,0.3)',
            display: 'flex', alignItems: 'center',
            justifyContent: 'center', fontSize: 24
          }}>🏆</div>
          <div>
            <div style={{ fontSize: 11, fontFamily: 'JetBrains Mono, monospace',
              color: 'var(--neon-green)', letterSpacing: '0.1em',
              marginBottom: 4 }}>
              BEST PERFORMING MODEL
            </div>
            <div style={{ fontSize: 22, fontWeight: 800 }}>{best}</div>
          </div>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div style={{ fontSize: 11, fontFamily: 'JetBrains Mono, monospace',
            color: 'var(--text-muted)', marginBottom: 4 }}>
            {mLabel.toUpperCase()}
          </div>
          <div style={{ fontSize: 36, fontWeight: 900,
            color: 'var(--neon-green)' }}>
            {(bestScore * 100).toFixed(1)}
            <span style={{ fontSize: 18, fontWeight: 400 }}>%</span>
          </div>
        </div>
      </motion.div>

      {/* Charts row */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr',
        gap: 20, marginBottom: 20 }}>

        {/* Bar chart */}
        <Card delay={0.1}>
          <div style={{ fontSize: 12, fontWeight: 600,
            letterSpacing: '0.06em', textTransform: 'uppercase',
            color: 'var(--text-secondary)', marginBottom: 16,
            fontFamily: 'JetBrains Mono, monospace' }}>
            Model Comparison — {mLabel}
          </div>
          <div style={{ height: 300 }}>
            <PlotlyChart
              data={[{
                x: models.map(([n]) => n),
                y: models.map(([, v]) => v[metric] || 0),
                type: 'bar',
                marker: {
                  color: models.map(([n]) =>
                    n === best ? '#10b981' : '#3b82f6'),
                  line: { width: 0 }
                }
              }]}
              layout={{
                xaxis: { title: 'Model' },
                yaxis: { title: mLabel }
              }}
            />
          </div>
        </Card>

        {/* Ranked list */}
        <Card delay={0.15}>
          <div style={{ fontSize: 12, fontWeight: 600,
            letterSpacing: '0.06em', textTransform: 'uppercase',
            color: 'var(--text-secondary)', marginBottom: 16,
            fontFamily: 'JetBrains Mono, monospace' }}>
            Ranked Performance
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
            {models.map(([name, scores], i) => {
              const score = scores[metric] || 0
              const pct   = maxScore > 0 ? (score / maxScore * 100) : 0
              const isBest = name === best
              return (
                <motion.div
                  key={name}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.06 }}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 12,
                    padding: '10px 12px',
                    borderRadius: 8,
                    background: isBest ? 'rgba(16,185,129,0.05)' : 'transparent',
                    marginBottom: 4
                  }}
                >
                  <div style={{ fontSize: 11, fontFamily: 'JetBrains Mono, monospace',
                    color: 'var(--text-muted)', minWidth: 20 }}>
                    {i + 1}
                  </div>
                  <div style={{ fontSize: 13, fontWeight: 600, minWidth: 140,
                    color: isBest ? 'var(--neon-green)' : 'var(--text-primary)' }}>
                    {name}
                    {isBest && <span style={{ marginLeft: 6 }}>⭐</span>}
                  </div>
                  <div style={{ flex: 1 }}>
                    <div style={{ height: 6, background: 'var(--bg-tertiary)',
                      borderRadius: 3, overflow: 'hidden' }}>
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${pct}%` }}
                        transition={{ duration: 0.8, delay: i * 0.06 }}
                        style={{
                          height: '100%', borderRadius: 3,
                          background: isBest ? 'var(--neon-green)' : 'var(--accent)'
                        }}
                      />
                    </div>
                  </div>
                  <div style={{ fontFamily: 'JetBrains Mono, monospace',
                    fontSize: 13, minWidth: 55, textAlign: 'right',
                    color: isBest ? 'var(--neon-green)' : 'var(--text-secondary)' }}>
                    {(score * 100).toFixed(1)}%
                  </div>
                </motion.div>
              )
            })}
          </div>
        </Card>
      </div>

      {/* Full metrics table */}
      <Card delay={0.2} style={{ marginBottom: 20 }}>
        <div style={{ fontSize: 12, fontWeight: 600,
          letterSpacing: '0.06em', textTransform: 'uppercase',
          color: 'var(--text-secondary)', marginBottom: 16,
          fontFamily: 'JetBrains Mono, monospace' }}>
          Full Metrics Table
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr>
                {['Model', ...metricKeys].map(h => (
                  <th key={h} style={{
                    background: 'var(--bg-tertiary)',
                    color: 'var(--text-secondary)',
                    padding: '10px 16px',
                    fontFamily: 'JetBrains Mono, monospace',
                    fontSize: 11, textAlign: 'left',
                    borderBottom: '1px solid var(--border)'
                  }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {models.map(([name, scores]) => (
                <tr key={name} style={{
                  color: name === best ? 'var(--neon-green)' : 'inherit'
                }}>
                  <td style={{ padding: '10px 16px', fontWeight: 600,
                    borderBottom: '1px solid rgba(0,212,255,0.04)',
                    fontFamily: 'JetBrains Mono, monospace', fontSize: 12 }}>
                    {name} {name === best ? '⭐' : ''}
                  </td>
                  {metricKeys.map(k => (
                    <td key={k} style={{
                      padding: '10px 16px',
                      fontFamily: 'JetBrains Mono, monospace', fontSize: 12,
                      color: name === best ? 'var(--neon-green)' : 'var(--text-secondary)',
                      borderBottom: '1px solid rgba(0,212,255,0.04)'
                    }}>
                      {scores[k]}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* AI Panel */}
      <Card delay={0.25}>
        <div style={{ fontSize: 13, fontWeight: 600,
          letterSpacing: '0.08em', textTransform: 'uppercase',
          color: 'var(--text-secondary)', marginBottom: 4,
          display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ width: 3, height: 14,
            background: 'var(--neon-violet)',
            borderRadius: 2, display: 'inline-block' }}/>
          AI Training Analysis
        </div>
        <AIPanel text={aiTrainingText} loading={aiLoading}/>
      </Card>
    </div>
  )
}