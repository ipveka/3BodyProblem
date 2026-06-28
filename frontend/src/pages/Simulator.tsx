import { Suspense, lazy, useCallback, useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { useStore } from '../store'
import Controls from '../components/Controls'
import BodyEditor from '../components/BodyEditor'

// Lazy-load heavy, deferrable pieces so their vendor chunks (three.js, recharts)
// are fetched on demand instead of blocking initial load.
const Viewer3D = lazy(() => import('../components/Viewer3D'))
const EnergyChart = lazy(() => import('../components/EnergyChart'))
const ComparisonPanel = lazy(() => import('../components/ComparisonPanel'))

const MIN_WIDTH = 360
const MAX_WIDTH = 760
const DEFAULT_WIDTH = 480
const MOBILE_QUERY = '(max-width: 768px)'

export default function Simulator() {
  const loadPresets = useStore((s) => s.loadPresets)

  const [isMobile, setIsMobile] = useState(
    () => typeof window !== 'undefined' && window.matchMedia(MOBILE_QUERY).matches,
  )
  const [width, setWidth] = useState(() => {
    const saved = Number(localStorage.getItem('sidebarWidth'))
    return saved >= MIN_WIDTH && saved <= MAX_WIDTH ? saved : DEFAULT_WIDTH
  })
  const dragging = useRef(false)

  useEffect(() => {
    loadPresets()
  }, [loadPresets])

  useEffect(() => {
    const mq = window.matchMedia(MOBILE_QUERY)
    const onChange = (e: MediaQueryListEvent) => setIsMobile(e.matches)
    mq.addEventListener('change', onChange)
    return () => mq.removeEventListener('change', onChange)
  }, [])

  useEffect(() => {
    localStorage.setItem('sidebarWidth', String(width))
  }, [width])

  const startDrag = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    dragging.current = true
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
  }, [])

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!dragging.current) return
      setWidth(Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, e.clientX)))
    }
    const onUp = () => {
      if (!dragging.current) return
      dragging.current = false
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    return () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
  }, [])

  const sidebar = (
    <aside className="sidebar">
      <Controls />
      <BodyEditor />
      <Suspense fallback={null}>
        <EnergyChart />
        <ComparisonPanel />
      </Suspense>
      <footer>
        <Link to="/" className="home-link">← Home</Link>
        <span>FastAPI · React · three.js</span>
      </footer>
    </aside>
  )

  const viewer = (
    <main className="viewer">
      <Suspense fallback={<div className="viewer-loading">Loading 3D viewer…</div>}>
        <Viewer3D />
      </Suspense>
    </main>
  )

  if (isMobile) {
    // Stacked layout: viewer on top, controls scroll below.
    return (
      <div className="app app-mobile">
        {viewer}
        {sidebar}
      </div>
    )
  }

  return (
    <div className="app" style={{ gridTemplateColumns: `${width}px 6px 1fr` }}>
      {sidebar}
      <div
        className="resizer"
        onMouseDown={startDrag}
        onDoubleClick={() => setWidth(DEFAULT_WIDTH)}
        title="Drag to resize · double-click to reset"
        role="separator"
        aria-orientation="vertical"
      />
      {viewer}
    </div>
  )
}
