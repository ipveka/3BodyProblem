import { useStore } from '../store'
import type { SimRequest } from '../api/client'

const METHODS: SimRequest['method'][] = ['rk4', 'verlet', 'euler', 'scipy']

export default function Controls() {
  const {
    presets, selectedPresetId, request, result, loading, error,
    frame, playing, selectPreset, updateRequest, run, setFrame, setPlaying,
  } = useStore()

  return (
    <div className="panel">
      <h1>🌌 N-Body Simulator</h1>

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
        Method
        <select
          value={request?.method ?? 'rk4'}
          onChange={(e) => updateRequest({ method: e.target.value as SimRequest['method'] })}
        >
          {METHODS.map((m) => <option key={m} value={m}>{m.toUpperCase()}</option>)}
        </select>
      </label>

      <label>
        Duration
        <input
          type="number"
          value={request?.duration ?? 0}
          min={0}
          onChange={(e) => updateRequest({ duration: Number(e.target.value) })}
        />
      </label>

      <label>
        Steps (n_points)
        <input
          type="number"
          value={request?.n_points ?? 1000}
          min={2}
          max={100000}
          onChange={(e) => updateRequest({ n_points: Number(e.target.value) })}
        />
      </label>

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
