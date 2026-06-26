import { useEffect } from 'react'
import { useStore } from './store'
import Controls from './components/Controls'
import EnergyChart from './components/EnergyChart'
import Viewer3D from './components/Viewer3D'

export default function App() {
  const loadPresets = useStore((s) => s.loadPresets)

  useEffect(() => {
    loadPresets()
  }, [loadPresets])

  return (
    <div className="app">
      <aside className="sidebar">
        <Controls />
        <EnergyChart />
        <footer>Gravitational N-body simulation · FastAPI + React + three.js</footer>
      </aside>
      <main className="viewer">
        <Viewer3D />
      </main>
    </div>
  )
}
