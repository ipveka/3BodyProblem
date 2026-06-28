import { useEffect, useState } from 'react'
import { useStore } from '../store'
import type { SimRequest } from '../api/client'

const METHODS: SimRequest['method'][] = ['rk4', 'verlet', 'euler', 'scipy']

type TimeUnit = 'hours' | 'days' | 'years'
const TIME_FACTORS: Record<TimeUnit, number> = {
  hours: 3600,
  days: 86400,
  years: 86400 * 365.25,
}

function trim(n: number): string {
  // Show a clean number (no long float tails) for the duration field.
  return String(Number(n.toFixed(6)))
}

export default function Controls() {
  const {
    presets, selectedPresetId, request, result, loading, error,
    frame, playing, viewMode, selectPreset, updateRequest, run,
    setFrame, setPlaying, setViewMode,
  } = useStore()

  // The figure-eight (and any G=1 system) runs in normalized time units; real
  // systems run in seconds, which we present in friendlier units (default days).
  const normalized = (request?.length_unit ?? 1000) === 1
  const [unit, setUnit] = useState<TimeUnit>('days')
  const [durationStr, setDurationStr] = useState('')

  // Re-derive the displayed duration when the preset or the unit changes (not on
  // every keystroke, so typing isn't disrupted).
  useEffect(() => {
    if (!request) return
    const factor = normalized ? 1 : TIME_FACTORS[unit]
    setDurationStr(trim(request.duration / factor))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedPresetId, unit, normalized])

  const onDurationChange = (value: string) => {
    setDurationStr(value)
    const v = Number(value)
    if (!Number.isNaN(v) && v > 0) {
      updateRequest({ duration: normalized ? v : v * TIME_FACTORS[unit] })
    }
  }

  return (
    <div className="panel">
      <div className="title-row">
        <h1>🌌 N-Body Simulator</h1>
        <div className="dim-toggle view-toggle">
          {(['2D', '3D'] as const).map((m) => (
            <button
              key={m}
              className={viewMode === m ? 'active' : ''}
              onClick={() => setViewMode(m)}
              title={`${m} camera`}
            >
              {m}
            </button>
          ))}
        </div>
      </div>

      <label>
        Preset
        <select
          value={selectedPresetId ?? ''}
          onChange={(e) => selectPreset(e.target.value)}
        >
          {presets.map((p) => (
            <option key={p.id} value={p.id}>{p.name}</option>
          ))}
        </select>
      </label>
      {selectedPresetId && (
        <p className="hint">
          {presets.find((p) => p.id === selectedPresetId)?.description}
        </p>
      )}

      <label>
        Integration method
        <select
          value={request?.method ?? 'rk4'}
          onChange={(e) => updateRequest({ method: e.target.value as SimRequest['method'] })}
        >
          {METHODS.map((m) => <option key={m} value={m}>{m.toUpperCase()}</option>)}
        </select>
      </label>

      <div className="field">
        <span className="field-label">Simulated time</span>
        <div className="duration-row">
          <input
            type="number"
            min={0}
            value={durationStr}
            onChange={(e) => onDurationChange(e.target.value)}
          />
          {normalized ? (
            <span className="unit-static">time units</span>
          ) : (
            <select value={unit} onChange={(e) => setUnit(e.target.value as TimeUnit)}>
              <option value="hours">hours</option>
              <option value="days">days</option>
              <option value="years">years</option>
            </select>
          )}
        </div>
        <p className="hint">
          {normalized
            ? 'Normalized units (G = 1) — this system has no physical timescale.'
            : 'How much physical time to simulate (converted to seconds internally).'}
        </p>
      </div>

      <label>
        Steps
        <input
          type="number"
          value={request?.n_points ?? 1000}
          min={2}
          max={100000}
          onChange={(e) => updateRequest({ n_points: Number(e.target.value) })}
        />
      </label>
      <p className="hint">More steps → smoother animation and higher accuracy.</p>

      <button className="run" onClick={() => run()} disabled={loading || !request}>
        {loading ? 'Simulating…' : '🚀 Run Simulation'}
      </button>

      {error && <p className="error">⚠️ {error}</p>}

      {result && (
        <div className="playback">
          <div className="metrics">
            <span>Bodies: <b>{result.bodies.length}</b></span>
            <span>Dim: <b>{result.dim}D</b></span>
            <span>Drift: <b>{(result.energy_drift * 100).toExponential(2)}%</b></span>
          </div>
          <div className="transport">
            <button onClick={() => setPlaying(!playing)}>
              {playing ? '⏸' : '▶'}
            </button>
            <input
              type="range"
              min={0}
              max={result.positions.length - 1}
              value={frame}
              onChange={(e) => setFrame(Number(e.target.value))}
            />
            <span className="frame-label">
              {frame + 1}/{result.positions.length}
            </span>
          </div>
        </div>
      )}
    </div>
  )
}
