import { useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import useAppStore from '../../store/appStore'

export default function PlotlyChart({ data, layout = {}, delay = 0 }) {
  const ref   = useRef(null)
  const theme = useAppStore(s => s.theme)

  useEffect(() => {
    if (!ref.current || !data || !window.Plotly) return

    const baseLayout = {
      paper_bgcolor: 'transparent',
      plot_bgcolor:  'transparent',
      font: {
        family: 'JetBrains Mono, monospace',
        size: 11,
        color: theme === 'dark' ? '#94a3b8' : '#475569'
      },
      margin: { l: 60, r: 20, t: 30, b: 60 },
      xaxis: {
        gridcolor: theme === 'dark'
          ? 'rgba(0,212,255,0.06)'
          : 'rgba(37,99,235,0.08)',
        zeroline: false,
        color: theme === 'dark' ? '#475569' : '#94a3b8'
      },
      yaxis: {
        gridcolor: theme === 'dark'
          ? 'rgba(0,212,255,0.06)'
          : 'rgba(37,99,235,0.08)',
        zeroline: false,
        color: theme === 'dark' ? '#475569' : '#94a3b8'
      },
      legend: {
        bgcolor: 'transparent',
        font: { color: theme === 'dark' ? '#94a3b8' : '#475569' }
      },
      ...layout
    }

    window.Plotly.newPlot(ref.current, data, baseLayout, {
      displayModeBar: false,
      responsive: true
    })

    return () => {
      if (ref.current && window.Plotly) {
        window.Plotly.purge(ref.current)
      }
    }
  }, [data, theme])

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5, delay }}
      ref={ref}
      style={{ width: '100%', height: '100%' }}
    />
  )
}