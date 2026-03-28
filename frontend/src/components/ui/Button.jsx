import { motion } from 'framer-motion'

export default function Button({
  children, onClick, variant = 'default',
  disabled = false, loading = false, style = {}
}) {
  const styles = {
    default: {
      background: 'transparent',
      border: '1px solid var(--border-hover)',
      color: 'var(--text-secondary)',
    },
    primary: {
      background: 'var(--gradient-accent)',
      border: 'none',
      color: 'white',
      boxShadow: '0 2px 12px rgba(139,92,246,0.3)',
    },
    success: {
      background: 'rgba(52,211,153,0.08)',
      border: '1px solid rgba(52,211,153,0.25)',
      color: 'var(--neon-green)',
    },
    ghost: {
      background: 'transparent',
      border: '1px solid transparent',
      color: 'var(--text-muted)',
    }
  }

  return (
    <motion.button
      whileHover={{ scale: 1.02, opacity: 0.95 }}
      whileTap={{ scale: 0.97 }}
      onClick={onClick}
      disabled={disabled || loading}
      style={{
        ...styles[variant],
        padding: '10px 20px',
        borderRadius: '10px',
        fontFamily: 'Inter, sans-serif',
        fontSize: '13px',
        fontWeight: 500,
        cursor: disabled || loading ? 'not-allowed' : 'pointer',
        display: 'inline-flex',
        alignItems: 'center',
        gap: '8px',
        opacity: disabled ? 0.4 : 1,
        transition: 'all 0.2s',
        whiteSpace: 'nowrap',
        letterSpacing: '0.01em',
        ...style
      }}
    >
      {loading && (
        <span style={{
          width: 13, height: 13,
          border: '1.5px solid rgba(255,255,255,0.2)',
          borderTop: '1.5px solid currentColor',
          borderRadius: '50%',
          display: 'inline-block',
          animation: 'spin 0.7s linear infinite'
        }}/>
      )}
      {children}
    </motion.button>
  )
}