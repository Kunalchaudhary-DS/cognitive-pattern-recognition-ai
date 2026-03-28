const colors = {
  blue:   { bg: 'rgba(59,130,246,0.15)',  color: '#93c5fd', border: 'rgba(59,130,246,0.3)'  },
  green:  { bg: 'rgba(16,185,129,0.15)',  color: '#6ee7b7', border: 'rgba(16,185,129,0.3)'  },
  amber:  { bg: 'rgba(245,158,11,0.15)',  color: '#fcd34d', border: 'rgba(245,158,11,0.3)'  },
  red:    { bg: 'rgba(239,68,68,0.15)',   color: '#fca5a5', border: 'rgba(239,68,68,0.3)'   },
  violet: { bg: 'rgba(124,58,237,0.15)',  color: '#c4b5fd', border: 'rgba(124,58,237,0.3)'  },
  cyan:   { bg: 'rgba(0,212,255,0.15)',   color: '#67e8f9', border: 'rgba(0,212,255,0.3)'   },
}

export default function Badge({ children, color = 'blue', style = {} }) {
  const c = colors[color] || colors.blue
  return (
    <span style={{
      display: 'inline-flex',
      alignItems: 'center',
      padding: '3px 10px',
      borderRadius: '20px',
      fontSize: '11px',
      fontFamily: 'JetBrains Mono, monospace',
      fontWeight: 600,
      letterSpacing: '0.05em',
      background: c.bg,
      color: c.color,
      border: `1px solid ${c.border}`,
      ...style
    }}>
      {children}
    </span>
  )
}