import { motion } from 'framer-motion'
import useAppStore from '../../store/appStore'

export default function Navbar({ activePage, setActivePage }) {
  const { theme, toggleTheme, targetColumn } = useAppStore()

  const tabs = [
    { id: 'data',      label: 'Data Processing',   num: '01' },
    { id: 'training',  label: 'Model Training',     num: '02' },
    { id: 'dashboard', label: 'Pattern Dashboard',  num: '03' },
  ]

  return (
    <motion.nav
      initial={{ y: -60, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5, ease: [0.25, 0.1, 0.25, 1] }}
      style={{
        position: 'sticky', top: 0, zIndex: 100,
        background: 'rgba(17, 19, 24, 0.85)',
        backdropFilter: 'blur(24px)',
        WebkitBackdropFilter: 'blur(24px)',
        borderBottom: '1px solid var(--border-subtle)',
        padding: '0 40px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        height: 60,
      }}
    >
      {/* Brand */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <div style={{
          width: 32, height: 32,
          background: 'var(--gradient-accent)',
          borderRadius: 9,
          display: 'flex', alignItems: 'center',
          justifyContent: 'center', fontSize: 15,
          boxShadow: '0 2px 12px rgba(139,92,246,0.3)'
        }}>
          🧠
        </div>
        <div>
          <div style={{
            fontSize: 13, fontWeight: 600,
            letterSpacing: '0.06em', color: 'var(--text-primary)'
          }}>
            CPRS
          </div>
          <div style={{
            fontSize: 9,
            fontFamily: 'JetBrains Mono, monospace',
            color: 'var(--accent)',
            letterSpacing: '0.12em',
            opacity: 0.8
          }}>
            COGNITIVE PATTERN RECOGNITION
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 2 }}>
        {tabs.map(tab => (
          <motion.button
            key={tab.id}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => setActivePage(tab.id)}
            style={{
              padding: '7px 16px',
              borderRadius: 8,
              border: activePage === tab.id
                ? '1px solid var(--border-hover)'
                : '1px solid transparent',
              background: activePage === tab.id
                ? 'var(--accent-soft)'
                : 'transparent',
              color: activePage === tab.id
                ? 'var(--accent)'
                : 'var(--text-muted)',
              fontFamily: 'Inter, sans-serif',
              fontSize: 13,
              fontWeight: activePage === tab.id ? 500 : 400,
              cursor: 'pointer',
              display: 'flex', alignItems: 'center', gap: 7,
              transition: 'all 0.2s',
              letterSpacing: '0.01em',
            }}
          >
            <span style={{
              fontFamily: 'JetBrains Mono, monospace',
              fontSize: 9, opacity: 0.5
            }}>
              {tab.num}
            </span>
            {tab.label}
          </motion.button>
        ))}
      </div>

      {/* Right */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 12
      }}>
        {targetColumn && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            style={{
              display: 'flex', alignItems: 'center', gap: 7,
              background: 'rgba(139,92,246,0.08)',
              border: '1px solid rgba(139,92,246,0.2)',
              borderRadius: 20, padding: '4px 12px',
              fontSize: 11,
              fontFamily: 'JetBrains Mono, monospace',
            }}
          >
            <span style={{ color: 'var(--text-muted)' }}>TARGET</span>
            <span style={{ color: 'var(--neon-violet)', fontWeight: 500 }}>
              {targetColumn}
            </span>
          </motion.div>
        )}

        <div style={{
          display: 'flex', alignItems: 'center', gap: 7,
          fontSize: 11,
          fontFamily: 'JetBrains Mono, monospace',
          color: 'var(--text-muted)'
        }}>
          <motion.div
            animate={{ opacity: [1, 0.3, 1] }}
            transition={{ duration: 2.5, repeat: Infinity }}
            style={{
              width: 6, height: 6, borderRadius: '50%',
              background: 'var(--neon-green)',
              boxShadow: '0 0 6px var(--neon-green)'
            }}
          />
          System Ready
        </div>

        <motion.button
          whileHover={{ scale: 1.08 }}
          whileTap={{ scale: 0.92 }}
          onClick={toggleTheme}
          style={{
            width: 34, height: 34, borderRadius: 9,
            border: '1px solid var(--border)',
            background: 'var(--bg-tertiary)',
            cursor: 'pointer', fontSize: 14,
            display: 'flex', alignItems: 'center',
            justifyContent: 'center',
            transition: 'all 0.2s',
            color: 'var(--text-secondary)'
          }}
        >
          {theme === 'dark' ? '☀️' : '🌙'}
        </motion.button>
      </div>
    </motion.nav>
  )
}