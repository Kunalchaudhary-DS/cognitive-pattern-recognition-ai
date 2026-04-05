import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import useAppStore from '../store/appStore'
import {
  getDashboardData,
  getAIPatternExplanation,
  getAIInsightSummary,
  getAIPanelInsights
} from '../api/client.js'
import Card from '../components/ui/Card'
import Badge from '../components/ui/Badge'
import AIPanel from '../components/ui/AIPanel'
import PlotlyChart from '../components/charts/PlotlyChart'
import StepFlow from '../components/ui/StepFlow'
import PredictionEngine from '../components/ui/PredictionEngine'

export default function DashboardPage() {
  const { trained, setDashboardData, dashboardData, datasetData } = useAppStore()

  const [loading, setLoading] = useState(false)
  const [aiPattern, setAiPattern] = useState('')
  const { aiPatternText, setAiPatternText, aiSummaryText, setAiSummaryText } = useAppStore()
  const [aiPatternLoad, setAiPatternLoad] = useState(false)
  const [aiSummaryLoad, setAiSummaryLoad] = useState(false)
  const [panelInsights, setPanelInsights] = useState({ patterns_insight: '', clusters_insight: '' })
  const [panelInsightsLoad, setPanelInsightsLoad] = useState(false)

  useEffect(() => {
    if (trained) loadDashboard()
  }, [trained])

  async function loadDashboard() {
    setLoading(true)
    try {
      const data = await getDashboardData()
      if (data.error) return
      setDashboardData(data)
      // Fetch AI panels in parallel
      fetchAIPattern()
      fetchAISummary()
      fetchPanelInsights()
    } catch { }
    setLoading(false)
  }

  async function fetchAIPattern() {
    if (aiPatternText) return  // already generated
    setAiPatternLoad(true)
    try {
      const res = await getAIPatternExplanation()
      setAiPatternText(res.explanation || '')
    } catch { }
    setAiPatternLoad(false)
  }

  async function fetchAISummary() {
    if (aiSummaryText) return  // already generated
    setAiSummaryLoad(true)
    try {
      const res = await getAIInsightSummary()
      setAiSummaryText(res.summary || '')
    } catch { }
    setAiSummaryLoad(false)
  }

  async function fetchPanelInsights() {
    setPanelInsightsLoad(true)
    try {
      const res = await getAIPanelInsights()
      if (!res.error) {
        setPanelInsights({
          patterns_insight: res.patterns_insight || '',
          clusters_insight: res.clusters_insight || '',
        })
      }
    } catch { }
    setPanelInsightsLoad(false)
  }

  if (!trained) {
    return (
      <div style={{ padding: 32 }}>
        <StepFlow />
        <div style={{
          textAlign: 'center', padding: '80px 32px',
          color: 'var(--text-muted)',
          fontFamily: 'JetBrains Mono, monospace', fontSize: 13
        }}>
          Complete model training to unlock the full dashboard
        </div>
      </div>
    )
  }

  if (loading || !dashboardData) {
    return (
      <div style={{ padding: 32 }}>
        <StepFlow />
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          gap: 12, padding: '80px 32px',
          fontFamily: 'JetBrains Mono, monospace',
          fontSize: 13, color: 'var(--text-muted)'
        }}>
          <div style={{
            width: 20, height: 20,
            border: '2px solid var(--bg-tertiary)',
            borderTop: '2px solid var(--accent)',
            borderRadius: '50%',
            animation: 'spin 0.7s linear infinite'
          }} />
          Loading dashboard data...
        </div>
      </div>
    )
  }

  const d = dashboardData
  const fi = d.feature_importance || {}
  const mc = d.model_comparison || {}
  const ps = d.pattern_score || {}
  const scoreColor = ps.score >= 80
    ? 'var(--neon-green)'
    : ps.score >= 60
      ? 'var(--neon-amber)'
      : 'var(--neon-red)'

  // Score ring
  const circumference = 2 * Math.PI * 40
  const offset = circumference - (ps.score || 0) / 100 * circumference

  // Chart data
  const targetChartData = [{
    x: Object.keys(d.target_distribution || {}),
    y: Object.values(d.target_distribution || {}),
    type: 'bar',
    marker: { color: '#4f46e5', line: { width: 0 } }
  }]

  const fiChartData = Object.keys(fi).length > 0 ? [{
    x: Object.values(fi),
    y: Object.keys(fi),
    type: 'bar', orientation: 'h',
    marker: { color: '#06b6d4', line: { width: 0 } }
  }] : null

  const bestModelName = dashboardData?.training_results?.BestModel ||
    Object.keys(mc).reduce((a, b) => mc[a] > mc[b] ? a : b, Object.keys(mc)[0])

  const mcChartData = Object.keys(mc).length > 0 ? [{
    x: Object.keys(mc),
    y: Object.values(mc),
    type: 'bar',
    marker: {
      color: Object.keys(mc).map(k =>
        k === bestModelName ? '#10b981' : '#8b5cf6'),
      line: { width: 0 }
    }
  }] : null

  const corrData = d.correlation_matrix?.length > 0 ? [{
    z: d.correlation_matrix,
    x: d.correlation_labels,
    y: d.correlation_labels,
    type: 'heatmap',
    colorscale: 'RdBu',
    reversescale: true,
    zmid: 0
  }] : null

  return (
    <div style={{ padding: 32, maxWidth: 1400, margin: '0 auto' }}>
      <StepFlow />

      {/* Header */}
      <div style={{ marginBottom: 28 }}>
        <h1 style={{ fontSize: 26, fontWeight: 800, marginBottom: 6 }}>
          Pattern Intelligence Dashboard
        </h1>
        <p style={{
          fontSize: 13, color: 'var(--text-muted)',
          fontFamily: 'JetBrains Mono, monospace'
        }}>
          {d.dataset_summary}
        </p>
      </div>

      {/* TOP ROW — Score + Distribution + Insights */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '220px 1fr 1fr',
        gap: 20, marginBottom: 20
      }}>

        {/* Pattern score ring */}
        <Card delay={0.05} style={{ textAlign: 'center' }}>
          <div style={{
            fontSize: 12, fontWeight: 600,
            letterSpacing: '0.06em', textTransform: 'uppercase',
            color: 'var(--text-secondary)', marginBottom: 16,
            fontFamily: 'JetBrains Mono, monospace'
          }}>
            Cognitive Score
          </div>
          <div style={{
            position: 'relative', width: 120,
            height: 120, margin: '0 auto 16px'
          }}>
            <svg viewBox="0 0 100 100" width="120" height="120"
              style={{ transform: 'rotate(-90deg)' }}>
              <circle cx="50" cy="50" r="40"
                fill="none" stroke="var(--bg-tertiary)" strokeWidth="8" />
              <motion.circle
                cx="50" cy="50" r="40"
                fill="none" stroke={scoreColor} strokeWidth="8"
                strokeLinecap="round"
                initial={{ strokeDashoffset: circumference }}
                animate={{ strokeDashoffset: offset }}
                transition={{ duration: 1.5, ease: 'easeOut' }}
                style={{
                  strokeDasharray: circumference,
                  filter: `drop-shadow(0 0 6px ${scoreColor})`
                }}
              />
            </svg>
            <div style={{
              position: 'absolute', inset: 0,
              display: 'flex', flexDirection: 'column',
              alignItems: 'center', justifyContent: 'center'
            }}>
              <div style={{
                fontSize: 28, fontWeight: 900,
                color: scoreColor, lineHeight: 1
              }}>
                {ps.score || 0}
              </div>
              <div style={{
                fontSize: 10,
                fontFamily: 'JetBrains Mono, monospace',
                color: 'var(--text-muted)'
              }}>
                / 100
              </div>
            </div>
          </div>
          <div style={{
            display: 'flex', justifyContent: 'center',
            gap: 8, flexWrap: 'wrap'
          }}>
            <Badge color={
              ps.pattern_strength === 'Strong' ? 'green'
                : ps.pattern_strength === 'Moderate' ? 'amber' : 'red'
            }>
              {ps.pattern_strength}
            </Badge>
            <Badge color="blue">
              Quality: {ps.data_quality}/40
            </Badge>
          </div>
        </Card>

        {/* Target distribution */}
        <Card delay={0.1}>
          <div style={{
            fontSize: 12, fontWeight: 600,
            letterSpacing: '0.06em', textTransform: 'uppercase',
            color: 'var(--text-secondary)', marginBottom: 16,
            fontFamily: 'JetBrains Mono, monospace'
          }}>
            Target Distribution
          </div>
          <div style={{ height: 200 }}>
            <PlotlyChart
              data={targetChartData}
              layout={{
                margin: { l: 40, r: 10, t: 10, b: 50 },
                xaxis: { title: 'Class' },
                yaxis: { title: 'Count' }
              }}
            />
          </div>
        </Card>

        {/* Key insights */}
        <Card delay={0.15}>
          <div style={{
            fontSize: 12, fontWeight: 600,
            letterSpacing: '0.06em', textTransform: 'uppercase',
            color: 'var(--text-secondary)', marginBottom: 16,
            fontFamily: 'JetBrains Mono, monospace'
          }}>
            Key Insights
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {(d.insights || []).slice(0, 4).map((ins, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: 10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.2 + i * 0.08 }}
                style={{
                  display: 'flex', alignItems: 'flex-start',
                  gap: 10, padding: 12,
                  background: 'var(--bg-tertiary)',
                  borderRadius: 8,
                  border: '1px solid var(--border)',
                  fontSize: 13, color: 'var(--text-secondary)',
                  lineHeight: 1.5
                }}
              >
                <div style={{
                  width: 8, height: 8, borderRadius: '50%',
                  background: 'var(--accent)', marginTop: 4,
                  flexShrink: 0,
                  boxShadow: '0 0 6px var(--accent)'
                }} />
                {ins}
              </motion.div>
            ))}
          </div>
        </Card>
      </div>

      {/* CHARTS ROW */}
      <div style={{
        display: 'grid', gridTemplateColumns: '1fr 1fr',
        gap: 20, marginBottom: 20
      }}>
        {fiChartData && (
          <Card delay={0.2}>
            <div style={{
              fontSize: 12, fontWeight: 600,
              letterSpacing: '0.06em', textTransform: 'uppercase',
              color: 'var(--text-secondary)', marginBottom: 16,
              fontFamily: 'JetBrains Mono, monospace'
            }}>
              Feature Importance
            </div>
            <div style={{ height: 280 }}>
              <PlotlyChart
                data={fiChartData}
                layout={{
                  xaxis: { title: 'Importance' },
                  yaxis: { automargin: true }
                }}
              />
            </div>
          </Card>
        )}

        {mcChartData && (
          <Card delay={0.25}>
            <div style={{
              fontSize: 12, fontWeight: 600,
              letterSpacing: '0.06em', textTransform: 'uppercase',
              color: 'var(--text-secondary)', marginBottom: 16,
              fontFamily: 'JetBrains Mono, monospace'
            }}>
              Model Comparison
            </div>
            <div style={{ height: 280 }}>
              <PlotlyChart
                data={mcChartData}
                layout={{
                  xaxis: { title: 'Models' },
                  yaxis: { title: 'R2 Score' }
                }}
              />
            </div>
          </Card>
        )}
      </div>

      {/* CORRELATION HEATMAP */}
      {corrData && (
        <Card delay={0.3} style={{ marginBottom: 20 }}>
          <div style={{
            fontSize: 12, fontWeight: 600,
            letterSpacing: '0.06em', textTransform: 'uppercase',
            color: 'var(--text-secondary)', marginBottom: 16,
            fontFamily: 'JetBrains Mono, monospace'
          }}>
            Correlation Heatmap
          </div>
          <div style={{ height: 360 }}>
            <PlotlyChart
              data={corrData}
              layout={{ margin: { l: 80, r: 20, t: 20, b: 80 } }}
            />
          </div>
        </Card>
      )}

      {/* PATTERNS + CLUSTERS */}
      <div style={{
        display: 'grid', gridTemplateColumns: '1fr 1fr',
        gap: 20, marginBottom: 20
      }}>
        <Card delay={0.35}>
          <div style={{
            fontSize: 12, fontWeight: 600,
            letterSpacing: '0.06em', textTransform: 'uppercase',
            color: 'var(--text-secondary)', marginBottom: 16,
            fontFamily: 'JetBrains Mono, monospace'
          }}>
            Discovered Patterns
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {(d.patterns || []).slice(0, 6).map((p, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 + i * 0.06 }}
                style={{
                  display: 'flex', alignItems: 'flex-start', gap: 10,
                  padding: 12, background: 'var(--bg-tertiary)',
                  borderRadius: 8, border: '1px solid var(--border)',
                  fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.5
                }}
              >
                <div style={{
                  width: 8, height: 8, borderRadius: '50%',
                  background: 'var(--accent2)', marginTop: 4,
                  flexShrink: 0
                }} />
                {p}
              </motion.div>
            ))}
            {(!d.patterns || d.patterns.length === 0) && (
              <div style={{
                color: 'var(--text-muted)', fontSize: 13,
                fontFamily: 'JetBrains Mono, monospace'
              }}>
                No patterns detected
              </div>
            )}
          </div>

          {/* AI Insight strip for Patterns panel */}
          {(panelInsightsLoad || panelInsights.patterns_insight) && (
            <div style={{
              marginTop: 14,
              padding: '10px 14px',
              background: 'rgba(139,92,246,0.05)',
              border: '1px solid rgba(139,92,246,0.15)',
              borderLeft: '3px solid var(--neon-violet)',
              borderRadius: '0 8px 8px 0',
            }}>
              <div style={{
                fontSize: 9, fontFamily: 'JetBrains Mono, monospace',
                color: 'var(--neon-violet)', letterSpacing: '0.1em',
                marginBottom: 5, textTransform: 'uppercase'
              }}>⚡ CPR AI INSIGHT</div>
              {panelInsightsLoad && !panelInsights.patterns_insight
                ? <div style={{ fontSize: 12, color: 'var(--text-muted)', fontFamily: 'JetBrains Mono, monospace' }}>Analysing patterns...</div>
                : <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6 }}>{panelInsights.patterns_insight}</div>
              }
            </div>
          )}
        </Card>

        <Card delay={0.4}>
          <div style={{
            fontSize: 12, fontWeight: 600,
            letterSpacing: '0.06em', textTransform: 'uppercase',
            color: 'var(--text-secondary)', marginBottom: 16,
            fontFamily: 'JetBrains Mono, monospace'
          }}>
            Cluster Analysis
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {(d.clusters || []).map((c, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.45 + i * 0.06 }}
                style={{
                  display: 'flex', alignItems: 'flex-start', gap: 10,
                  padding: 12, background: 'var(--bg-tertiary)',
                  borderRadius: 8, border: '1px solid var(--border)',
                  fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.5
                }}
              >
                <div style={{
                  width: 8, height: 8, borderRadius: '50%',
                  background: 'var(--neon-violet)', marginTop: 4,
                  flexShrink: 0
                }} />
                {c}
              </motion.div>
            ))}
            {(!d.clusters || d.clusters.length === 0) && (
              <div style={{
                color: 'var(--text-muted)', fontSize: 13,
                fontFamily: 'JetBrains Mono, monospace'
              }}>
                No clusters identified
              </div>
            )}
          </div>

          {/* AI Insight strip for Clusters panel */}
          {(panelInsightsLoad || panelInsights.clusters_insight) && (
            <div style={{
              marginTop: 14,
              padding: '10px 14px',
              background: 'rgba(6,182,212,0.05)',
              border: '1px solid rgba(6,182,212,0.15)',
              borderLeft: '3px solid var(--accent)',
              borderRadius: '0 8px 8px 0',
            }}>
              <div style={{
                fontSize: 9, fontFamily: 'JetBrains Mono, monospace',
                color: 'var(--accent)', letterSpacing: '0.1em',
                marginBottom: 5, textTransform: 'uppercase'
              }}>⚡ CPR AI INSIGHT</div>
              {panelInsightsLoad && !panelInsights.clusters_insight
                ? <div style={{ fontSize: 12, color: 'var(--text-muted)', fontFamily: 'JetBrains Mono, monospace' }}>Analysing clusters...</div>
                : <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6 }}>{panelInsights.clusters_insight}</div>
              }
            </div>
          )}
        </Card>
      </div>

      {/* LIVE PREDICTION ENGINE */}
      <Card delay={0.45} style={{ marginBottom: 20 }}>
        <div style={{
          fontSize: 13, fontWeight: 600,
          letterSpacing: '0.08em', textTransform: 'uppercase',
          color: 'var(--text-secondary)', marginBottom: 20,
          display: 'flex', alignItems: 'center', gap: 8
        }}>
          <span style={{
            width: 3, height: 14,
            background: 'var(--gradient-accent)',
            borderRadius: 2, display: 'inline-block'
          }} />
          🎯 Live Prediction Engine
        </div>
        <PredictionEngine
          featureNames={
            Object.keys(dashboardData?.feature_importance || {}).slice(0, 3)
          }
          targetColumn={useAppStore.getState().targetColumn}
          problemType={useAppStore.getState().trainingResults?.ProblemType || 'regression'}
        />
      </Card>

      {/* AI PANELS */}
      <div style={{
        display: 'grid', gridTemplateColumns: '1fr 1fr',
        gap: 20, marginBottom: 20
      }}>
        <Card delay={0.5}>
          <div style={{
            fontSize: 13, fontWeight: 600,
            letterSpacing: '0.08em', textTransform: 'uppercase',
            color: 'var(--text-secondary)', marginBottom: 4,
            display: 'flex', alignItems: 'center', gap: 8
          }}>
            <span style={{
              width: 3, height: 14,
              background: 'var(--neon-violet)',
              borderRadius: 2, display: 'inline-block'
            }} />
            AI Pattern Interpretation
          </div>
          <AIPanel text={aiPatternText} loading={aiPatternLoad} />
        </Card>

        <Card delay={0.55}>
          <div style={{
            fontSize: 13, fontWeight: 600,
            letterSpacing: '0.08em', textTransform: 'uppercase',
            color: 'var(--text-secondary)', marginBottom: 4,
            display: 'flex', alignItems: 'center', gap: 8
          }}>
            <span style={{
              width: 3, height: 14,
              background: 'var(--accent)',
              borderRadius: 2, display: 'inline-block'
            }} />
            AI Research Conclusion
          </div>
          <AIPanel text={aiSummaryText} loading={aiSummaryLoad} label="CPR AI CONCLUSION" />
        </Card>
      </div>

      {/* AUTO GRAPHS */}
      {d.auto_graphs && d.auto_graphs.length > 0 && (
        <Card delay={0.6}>
          <div style={{
            fontSize: 12, fontWeight: 600,
            letterSpacing: '0.06em', textTransform: 'uppercase',
            color: 'var(--text-secondary)', marginBottom: 20,
            fontFamily: 'JetBrains Mono, monospace',
            display: 'flex', alignItems: 'center', justifyContent: 'space-between'
          }}>
            <span>Smart Data Visualizations</span>
            <span style={{
              fontSize: 10, color: 'var(--text-muted)',
              fontWeight: 400, textTransform: 'none', letterSpacing: 0
            }}>
              Top {(d.auto_graphs || []).length} most informative graphs · feature-priority ranked
            </span>
          </div>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(400px, 1fr))',
            gap: 20
          }}>
            {d.auto_graphs.map((graph, i) => {
              const store = useAppStore.getState()
              const raw = store.datasetData || datasetData
              if (!raw) return null

              let traces = []
              let chartLayout = {
                margin: { l: 50, r: 10, t: 10, b: 50 },
                xaxis: { title: graph.x },
                yaxis: { title: graph.y || '' }
              }

              if (graph.type === 'histogram') {
                traces = [{
                  x: raw.map(r => r[graph.x]), type: 'histogram',
                  marker: { color: '#3b82f6', line: { width: 0 } }, opacity: 0.85
                }]

              } else if (graph.type === 'bar') {
                const counts = {}
                raw.forEach(r => { counts[r[graph.x]] = (counts[r[graph.x]] || 0) + 1 })
                traces = [{
                  x: Object.keys(counts), y: Object.values(counts),
                  type: 'bar', marker: { color: '#6366f1', line: { width: 0 } }
                }]

              } else if (graph.type === 'scatter') {
                traces = [{
                  x: raw.map(r => r[graph.x]), y: raw.map(r => r[graph.y]),
                  mode: 'markers', type: 'scatter',
                  marker: { size: 4, color: '#2563eb', opacity: 0.6 }
                }]

              } else if (graph.type === 'box') {
                traces = [{
                  x: raw.map(r => r[graph.x]), y: raw.map(r => r[graph.y]),
                  type: 'box', marker: { color: '#14b8a6' }
                }]

              } else if (graph.type === 'pie') {
                const counts = {}
                raw.forEach(r => {
                  const val = String(r[graph.x] ?? 'Unknown')
                  counts[val] = (counts[val] || 0) + 1
                })
                const PIE_PALETTE = [
                  '#8b5cf6', '#06b6d4', '#10b981', '#f59e0b',
                  '#ef4444', '#3b82f6', '#ec4899', '#84cc16'
                ]
                traces = [{
                  labels: Object.keys(counts),
                  values: Object.values(counts),
                  type: 'pie',
                  hole: 0.38,
                  marker: { colors: PIE_PALETTE },
                  textinfo: 'label+percent',
                  textfont: { size: 11 },
                  insidetextorientation: 'radial'
                }]
                chartLayout = {
                  margin: { l: 10, r: 10, t: 10, b: 10 },
                  showlegend: true,
                  legend: {
                    font: { size: 11, color: 'var(--text-secondary)' },
                    x: 1, y: 0.5
                  }
                }
              }

              if (!traces.length) return null

              return (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 16 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.6 + i * 0.04 }}
                  style={{
                    background: 'var(--bg-tertiary)',
                    border: '1px solid var(--border)',
                    borderRadius: 12, padding: 20
                  }}
                >
                  <div style={{
                    fontSize: 12,
                    fontFamily: 'JetBrains Mono, monospace',
                    color: 'var(--text-secondary)',
                    marginBottom: 12, fontWeight: 600,
                    textTransform: 'uppercase', letterSpacing: '0.05em'
                  }}>
                    {graph.title}
                  </div>
                  <div style={{ height: 260 }}>
                    <PlotlyChart
                      data={traces}
                      layout={chartLayout}
                      delay={i * 0.04}
                    />
                  </div>
                  {graph.insight && (
                    <div style={{
                      marginTop: 12, padding: '10px 14px',
                      background: 'rgba(59,130,246,0.05)',
                      borderLeft: '2px solid var(--accent)',
                      borderRadius: '0 6px 6px 0',
                      fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.6
                    }}>
                      {graph.insight}
                    </div>
                  )}
                </motion.div>
              )
            })}
          </div>
        </Card>
      )}
    </div>
  )
}