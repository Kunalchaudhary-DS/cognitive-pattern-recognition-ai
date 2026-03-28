import { motion } from 'framer-motion'
import useAppStore from '../../store/appStore'

const steps = [
  { n: 1, label: 'Upload' },
  { n: 2, label: 'Analyze' },
  { n: 3, label: 'Preprocess' },
  { n: 4, label: 'Train' },
]

export default function StepFlow() {
  const currentStep = useAppStore(s => s.currentStep)

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8,
      marginBottom: 28, flexWrap: 'wrap' }}>
      {steps.map((step, i) => {
        const done   = currentStep > step.n
        const active = currentStep === step.n
        return (
          <div key={step.n} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <motion.div
                animate={{
                  background: done ? 'var(--neon-green)' : active ? 'var(--accent)' : 'transparent',
                  borderColor: done ? 'var(--neon-green)' : active ? 'var(--accent)' : 'var(--text-muted)',
                  scale: active ? 1.1 : 1
                }}
                style={{
                  width: 26, height: 26, borderRadius: '50%',
                  border: '1px solid',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: 11,
                  fontFamily: 'JetBrains Mono, monospace',
                  color: done || active ? 'white' : 'var(--text-muted)',
                }}
              >
                {done ? '✓' : step.n}
              </motion.div>
              <span style={{
                fontSize: 12,
                fontFamily: 'JetBrains Mono, monospace',
                color: done || active ? 'var(--text-primary)' : 'var(--text-muted)'
              }}>
                {step.label}
              </span>
            </div>
            {i < steps.length - 1 && (
              <span style={{ color: 'var(--text-muted)', fontSize: 10 }}>→</span>
            )}
          </div>
        )
      })}
    </div>
  )
}