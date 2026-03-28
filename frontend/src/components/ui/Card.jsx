import { motion } from 'framer-motion'
import VanillaTilt from 'vanilla-tilt'
import { useEffect, useRef } from 'react'

export default function Card({
  children, tilt = false, style = {}, className = '', delay = 0, glow = false
}) {
  const ref = useRef(null)

  useEffect(() => {
    if (tilt && ref.current) {
      VanillaTilt.init(ref.current, {
        max: 4,
        speed: 600,
        glare: false,
      })
      return () => ref.current?.vanillaTilt?.destroy()
    }
  }, [tilt])

  return (
    <motion.div
      ref={ref}
      className={`glass ${className}`}
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay, ease: [0.25, 0.1, 0.25, 1] }}
      style={{
        padding: '28px 32px',
        ...(glow && { boxShadow: 'var(--shadow-glow)' }),
        ...style
      }}
    >
      {children}
    </motion.div>
  )
}