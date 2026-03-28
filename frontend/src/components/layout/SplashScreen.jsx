import { motion, AnimatePresence } from 'framer-motion'
import { useEffect, useState } from 'react'

export default function SplashScreen({ onDone }) {
  const [progress, setProgress] = useState(0)
  const [statusText, setStatusText] = useState('Initializing system...')

  const steps = [
    'Initializing system...',
    'Loading AI models...',
    'Connecting to backend...',
    'Calibrating pattern engine...',
    'Ready.',
  ]

  useEffect(() => {
    let step = 0
    const interval = setInterval(() => {
      step++
      setProgress(step * 20)
      setStatusText(steps[step] || 'Ready.')
      if (step >= 5) {
        clearInterval(interval)
        setTimeout(onDone, 600)
      }
    }, 500)
    return () => clearInterval(interval)
  }, [])

  return (
    <motion.div
      exit={{ opacity: 0, scale: 1.05 }}
      transition={{ duration: 0.5 }}
      style={{
        position: 'fixed', inset: 0, zIndex: 9999,
        background: 'var(--bg-primary)',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 32,
      }}
    >
      {/* Animated brain */}
      <motion.div
        animate={{
          scale: [1, 1.1, 1],
          filter: [
            'drop-shadow(0 0 20px rgba(0,212,255,0.3))',
            'drop-shadow(0 0 40px rgba(0,212,255,0.6))',
            'drop-shadow(0 0 20px rgba(0,212,255,0.3))',
          ]
        }}
        transition={{ duration: 2, repeat: Infinity }}
        style={{ fontSize: 80 }}
      >
        🧠
      </motion.div>

      {/* Title */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        style={{ textAlign: 'center' }}
      >
        <div style={{
          fontSize: 26, fontWeight: 600,
          letterSpacing: '0.08em',
          background: 'var(--gradient-accent)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          marginBottom: 6, fontFamily: 'Inter, sans-serif'
        }}>
          AI COGNITIVE PATTERN
        </div>
        <div style={{
          fontSize: 26, fontWeight: 300,
          letterSpacing: '0.12em',
          color: 'var(--text-secondary)',
          fontFamily: 'Inter, sans-serif'
        }}>
          RECOGNITION SYSTEM
        </div>
      </motion.div>

      {/* Progress bar */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        style={{ width: 320 }}
      >
        <div style={{
          height: 4,
          background: 'var(--bg-tertiary)',
          borderRadius: 2,
          overflow: 'hidden',
          marginBottom: 12
        }}>
          <motion.div
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.4, ease: 'easeOut' }}
            style={{
              height: '100%',
              background: 'linear-gradient(90deg, var(--accent), var(--neon-violet))',
              borderRadius: 2,
              boxShadow: '0 0 10px var(--accent)',
            }}
          />
        </div>
        <div style={{
          fontFamily: 'JetBrains Mono, monospace',
          fontSize: 12,
          color: 'var(--text-muted)',
          textAlign: 'center',
          letterSpacing: '0.05em'
        }}>
          {statusText}
        </div>
      </motion.div>
    </motion.div>
  )
}
