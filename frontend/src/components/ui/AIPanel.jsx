import { motion } from 'framer-motion'
import Spinner from './Spinner'
import { useTypewriter } from '../../hooks/useTypewriter'

function TypedText({ text }) {
  const { displayed } = useTypewriter(text, 12)
  return (
    <p style={{
      fontSize: '14px',
      lineHeight: 1.8,
      color: 'var(--text-secondary)',
      margin: 0
    }}>
      {displayed}
      <span style={{
        display: 'inline-block',
        width: 2, height: 14,
        background: 'var(--neon-violet)',
        marginLeft: 2,
        animation: 'pulse 1s infinite'
      }}/>
    </p>
  )
}

export default function AIPanel({ text, loading, label = 'CPR AI ANALYSIS' }) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      style={{
        background: 'linear-gradient(135deg, rgba(124,58,237,0.08), rgba(0,212,255,0.05))',
        border: '1px solid rgba(124,58,237,0.25)',
        borderRadius: '12px',
        padding: '20px',
        marginTop: '16px'
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
        <div style={{
          display: 'flex', alignItems: 'center', gap: 6,
          background: 'rgba(124,58,237,0.2)',
          border: '1px solid rgba(124,58,237,0.4)',
          borderRadius: '20px',
          padding: '4px 12px',
          fontSize: 11,
          fontFamily: 'JetBrains Mono, monospace',
          color: '#c4b5fd',
          letterSpacing: '0.05em'
        }}>
          <div style={{
            width: 6, height: 6, borderRadius: '50%',
            background: '#8b5cf6',
            animation: 'pulse 1.5s infinite'
          }}/>
          {label}
        </div>
      </div>

      {loading ? (
        <div style={{ display: 'flex', alignItems: 'center', gap: 10,
          fontFamily: 'JetBrains Mono, monospace', fontSize: 12,
          color: 'var(--text-muted)' }}>
          <Spinner size={16} color="#8b5cf6"/>
          CPR AI is thinking...
        </div>
      ) : text ? (
        <TypedText text={text}/>
      ) : null}
    </motion.div>
  )
}