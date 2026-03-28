import { useState, useEffect } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Toaster } from 'react-hot-toast'
import useAppStore from './store/appStore'
import Navbar from './components/layout/Navbar'
import SplashScreen from './components/layout/SplashScreen'
import LandingPage from './pages/LandingPage'
import DataPage from './pages/DataPage'
import TrainingPage from './pages/TrainingPage'
import DashboardPage from './pages/DashboardPage'

export default function App() {
  const [activePage,  setActivePage]  = useState('landing')
  const [showSplash,  setShowSplash]  = useState(true)
  const { theme, setFoundDatasets } = useAppStore()

  // Set default theme on mount
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
  }, [])

  // Global keyframe styles
  useEffect(() => {
    const style = document.createElement('style')
    style.textContent = `
      @keyframes spin {
        to { transform: rotate(360deg); }
      }
      @keyframes pulse {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0.4; }
      }
    `
    document.head.appendChild(style)
    return () => document.head.removeChild(style)
  }, [])

  // Called when user picks a dataset from landing page
  function handleDatasetSelected(ds) {
    setFoundDatasets([ds])
    setTimeout(() => setActivePage('data'), 50)
  }
  const showNavbar = activePage !== 'landing'

  return (
    <>
      {/* Splash screen */}
      <AnimatePresence>
        {showSplash && (
          <SplashScreen onDone={() => setShowSplash(false)}/>
        )}
      </AnimatePresence>

      {/* Main app */}
      {!showSplash && (
        <>
          {/* Navbar — hidden on landing page */}
          {showNavbar && (
            <Navbar activePage={activePage} setActivePage={setActivePage}/>
          )}

          {/* Page transitions */}
          <AnimatePresence mode="wait">
            {activePage === 'landing' && (
              <motion.div
                key="landing"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.3 }}
              >
                <LandingPage onComplete={handleDatasetSelected}/>
              </motion.div>
            )}

            {activePage === 'data' && (
              <motion.div
                key="data"
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -12 }}
                transition={{ duration: 0.25 }}
              >
                <DataPage setActivePage={setActivePage}/>
              </motion.div>
            )}

            {activePage === 'training' && (
              <motion.div
                key="training"
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -12 }}
                transition={{ duration: 0.25 }}
              >
                <TrainingPage/>
              </motion.div>
            )}

            {activePage === 'dashboard' && (
              <motion.div
                key="dashboard"
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -12 }}
                transition={{ duration: 0.25 }}
              >
                <DashboardPage/>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Toast notifications */}
          <Toaster
            position="bottom-right"
            toastOptions={{
              style: {
                background: 'var(--bg-tertiary)',
                border: '1px solid var(--border-hover)',
                color: 'var(--text-primary)',
                fontFamily: 'Inter, sans-serif',
                fontSize: 13,
                borderRadius: 10,
              },
              success: {
                iconTheme: {
                  primary: '#10b981',
                  secondary: 'white',
                }
              },
              error: {
                iconTheme: {
                  primary: '#ef4444',
                  secondary: 'white',
                }
              }
            }}
          />
        </>
      )}
    </>
  )
}