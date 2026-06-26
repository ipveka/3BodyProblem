import { Suspense, lazy, useEffect } from 'react'
import { useStore } from './store'
import Controls from './components/Controls'
import BodyEditor from './components/BodyEditor'

// Lazy-load heavy, deferrable pieces so their vendor chunks (three.js, recharts)
// are fetched on demand instead of blocking initial load.
const Viewer3D = lazy(() => import('./components/Viewer3D'))
const EnergyChart = lazy(() => import('./components/EnergyChart'))
const ComparisonPanel = lazy(() => import('./components/ComparisonPanel'))

export default function App() {
  const loadPresets = useStore((s) => s.loadPresets)

  useEffect(() => {
    loadPresets()
  }, [loadPresets])

  return (
    <div className="app">
      <aside className="sidebar">
        <Controls />
        <BodyEditor />
        <Suspense fallback={null}>
          <EnergyChart />
          <ComparisonPanel />
        </Suspense>
        <footer>Gravitational N-body simulation · FastAPI + React + three.js</footer>
      </aside>
      <main className="viewer">
        <Suspense fallback={<div className="viewer-loading">Loading 3D viewer…</div>}>
          <Viewer3D />
        </Suspense>
      </main>
    </div>
  )
}
