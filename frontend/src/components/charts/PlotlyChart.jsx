import { useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import useAppStore from '../../store/appStore'

// ── Color palettes ────────────────────────────────────────────────────────────
const PALETTE = {
  dark:  ['#818cf8','#34d399','#f472b6','#fb923c','#60a5fa','#a78bfa','#4ade80','#facc15'],
  light: ['#4f46e5','#059669','#db2777','#ea580c','#2563eb','#7c3aed','#16a34a','#ca8a04'],
}

// ── Detect chart type from trace array ───────────────────────────────────────
function detectType(data) {
  if (!data || !data.length) return 'unknown'
  const t = data[0]
  if (t.type === 'heatmap')   return 'heatmap'
  if (t.type === 'pie')       return 'pie'
  if (t.type === 'histogram') return 'histogram'
  if (t.type === 'violin')    return 'violin'
  if (t.type === 'box')       return 'box'
  if (t.type === 'bar' && t.orientation === 'h') return 'bar_h'
  if (t.type === 'bar')       return 'bar'
  if (t.type === 'scatter' || t.mode?.includes('markers')) return 'scatter'
  return 'unknown'
}

// ── Enhance raw traces with professional defaults ─────────────────────────────
function enhanceTraces(data, chartType, palette) {
  return data.map((trace, idx) => {
    const color = palette[idx % palette.length]
    const base  = { ...trace }

    if (chartType === 'bar' || chartType === 'bar_h') {
      return {
        ...base,
        opacity: base.opacity ?? 0.92,
        textposition: 'outside',
        textfont: { size: 10, color: 'rgba(148,163,184,0.85)' },
        marker: {
          line: { width: 0 },
          ...base.marker,
          // honour explicit colours (e.g. green best-model highlighting)
          color: base.marker?.color ?? color,
        },
        hovertemplate: chartType === 'bar_h'
          ? '<b>%{y}</b><br>Score: %{x:.4f}<extra></extra>'
          : '<b>%{x}</b><br>Count: %{y}<extra></extra>',
      }
    }

    if (chartType === 'scatter') {
      return {
        ...base,
        mode: base.mode ?? 'markers',
        marker: {
          size: 6,
          opacity: 0.72,
          line: { width: 0.5, color: 'rgba(255,255,255,0.15)' },
          ...base.marker,
          color: base.marker?.color ?? color,
        },
        hovertemplate: '<b>%{x:.3f}</b> → <b>%{y:.3f}</b><extra></extra>',
      }
    }

    if (chartType === 'histogram') {
      return {
        ...base,
        opacity: base.opacity ?? 0.85,
        autobinx: true,
        marker: {
          line: { width: 0 },
          ...base.marker,
          color: base.marker?.color ?? color,
        },
        hovertemplate: 'Range: %{x}<br>Count: %{y}<extra></extra>',
      }
    }

    if (chartType === 'violin') {
      return {
        ...base,
        type: 'violin',
        box:      { visible: true, width: 0.3 },
        meanline: { visible: true, color: '#f472b6', width: 1.5 },
        points:   'outliers',
        jitter:   0.3,
        marker:   { size: 3, opacity: 0.45, ...base.marker, color: base.marker?.color ?? color },
        line:     { width: 1.5 },
        fillcolor: `${color}28`,
        hovertemplate: '<b>%{x}</b><br>Value: %{y:.3f}<extra></extra>',
      }
    }

    if (chartType === 'box') {
      return {
        ...base,
        boxpoints: 'outliers',
        jitter: 0.35,
        whiskerwidth: 0.8,
        marker: {
          size: 4,
          opacity: 0.55,
          ...base.marker,
          color: base.marker?.color ?? color,
        },
        line: { width: 1.5 },
      }
    }

    if (chartType === 'pie') {
      return {
        ...base,
        hole: base.hole ?? 0.38,
        marker: {
          colors: base.marker?.colors ?? palette,
          line: { color: 'rgba(0,0,0,0.35)', width: 1.5 },
        },
        textinfo: 'label+percent',
        textfont: { size: 11 },
        insidetextorientation: 'radial',
        hovertemplate: '<b>%{label}</b><br>%{percent:.1%} (%{value})<extra></extra>',
      }
    }

    return base
  })
}

// ── Build full layout from base + type-specific overrides ─────────────────────
function getChartLayout(chartType, customLayout = {}, theme = 'dark') {
  const isDark   = theme === 'dark'
  const textColor  = isDark ? '#94a3b8' : '#64748b'
  const gridColor  = isDark ? 'rgba(148,163,184,0.07)' : 'rgba(71,85,105,0.10)'
  const palette    = PALETTE[isDark ? 'dark' : 'light']

  // ── Shared axis defaults ──────────────────────────────────────────────────
  const axisBase = {
    showline:  false,
    zeroline:  false,
    showgrid:  true,
    gridcolor: gridColor,
    gridwidth: 0.5,
    tickfont:  { size: 11, color: textColor, family: 'Inter, sans-serif' },
    titlefont: { size: 12, color: textColor, family: 'Inter, sans-serif' },
    automargin: true,
  }

  // ── Base layout all charts share ─────────────────────────────────────────
  const base = {
    paper_bgcolor: 'transparent',
    plot_bgcolor:  'transparent',
    font: {
      family: 'Inter, sans-serif',
      size:   11,
      color:  textColor,
    },
    colorway:   palette,
    margin:     { l: 55, r: 16, t: 20, b: 55 },
    xaxis:      { ...axisBase },
    yaxis:      { ...axisBase },
    hoverlabel: {
      bgcolor:     isDark ? 'rgba(17,19,24,0.95)' : 'rgba(248,250,252,0.97)',
      bordercolor: isDark ? 'rgba(139,92,246,0.45)' : 'rgba(79,70,229,0.35)',
      font: {
        color:  isDark ? '#f1f0ff' : '#1e1b4b',
        size:   12,
        family: 'Inter, sans-serif',
      },
    },
    legend: {
      bgcolor:     'transparent',
      borderwidth: 0,
      font: { size: 11, color: textColor, family: 'Inter, sans-serif' },
      x: 1, xanchor: 'right',
      y: 0, yanchor: 'bottom',
    },
    transition: { duration: 420, easing: 'cubic-in-out' },
  }

  // ── Type-specific overrides ───────────────────────────────────────────────
  const overrides = {}

  if (chartType === 'bar' || chartType === 'bar_h') {
    overrides.bargap     = 0.28
    overrides.bargroupgap = 0.06
    if (chartType === 'bar') {
      overrides.yaxis = { ...axisBase, showgrid: true }
      overrides.xaxis = { ...axisBase, showgrid: false }
    } else {
      // horizontal bar — flip grid
      overrides.xaxis = { ...axisBase, showgrid: true }
      overrides.yaxis = { ...axisBase, showgrid: false }
    }
  }

  if (chartType === 'scatter') {
    overrides.xaxis = { ...axisBase }
    overrides.yaxis = { ...axisBase }
  }

  if (chartType === 'heatmap') {
    overrides.margin = { l: 80, r: 20, t: 20, b: 80 }
    overrides.xaxis  = { ...axisBase, showgrid: false, side: 'bottom' }
    overrides.yaxis  = { ...axisBase, showgrid: false, autorange: 'reversed' }
  }

  if (chartType === 'pie') {
    overrides.margin     = { l: 10, r: 10, t: 10, b: 10 }
    overrides.showlegend = true
    overrides.legend     = {
      ...base.legend,
      x: 1, xanchor: 'right',
      y: 0.5, yanchor: 'middle',
    }
    // Pie has no axes
    delete overrides.xaxis
    delete overrides.yaxis
    delete base.xaxis
    delete base.yaxis
  }

  if (chartType === 'histogram') {
    overrides.bargap = 0.04
    overrides.yaxis  = { ...axisBase }
    overrides.xaxis  = { ...axisBase, showgrid: false }
  }

  if (chartType === 'box') {
    overrides.xaxis = { ...axisBase, showgrid: false }
    overrides.yaxis = { ...axisBase }
    overrides.boxmode = 'group'
  }

  // Deep-merge: base → overrides → caller customLayout (caller always wins)
  return deepMerge(deepMerge(base, overrides), customLayout)
}

// ── Minimal deep-merge (plain objects only) ───────────────────────────────────
function deepMerge(target, source) {
  const out = { ...target }
  for (const key of Object.keys(source ?? {})) {
    if (
      source[key] !== null &&
      typeof source[key] === 'object' &&
      !Array.isArray(source[key]) &&
      typeof target[key] === 'object' &&
      !Array.isArray(target[key])
    ) {
      out[key] = deepMerge(target[key] ?? {}, source[key])
    } else {
      out[key] = source[key]
    }
  }
  return out
}

// ── Component ─────────────────────────────────────────────────────────────────
export default function PlotlyChart({ data, layout = {}, delay = 0 }) {
  const ref   = useRef(null)
  const theme = useAppStore(s => s.theme)

  useEffect(() => {
    if (!ref.current || !data || !window.Plotly) return

    const chartType    = detectType(data)
    const palette      = PALETTE[theme === 'dark' ? 'dark' : 'light']
    const enhancedData = enhanceTraces(data, chartType, palette)
    const finalLayout  = getChartLayout(chartType, layout, theme)

    window.Plotly.newPlot(ref.current, enhancedData, finalLayout, {
      displayModeBar: false,
      responsive:     true,
    })

    return () => {
      if (ref.current && window.Plotly) window.Plotly.purge(ref.current)
    }
  }, [data, layout, theme])

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, delay, ease: 'easeOut' }}
      ref={ref}
      style={{ width: '100%', height: '100%' }}
    />
  )
}