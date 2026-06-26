import { create } from 'zustand'
import {
  getPresets,
  simulate,
  type BodyIn,
  type PresetOut,
  type SimRequest,
  type SimResponse,
} from './api/client'

export type ViewMode = '2D' | '3D'

const PALETTE = ['#e74c3c', '#2ecc71', '#3498db', '#f1c40f', '#9b59b6',
  '#1abc9c', '#e67e22', '#ff6fb5']

/** Methods compared side by side in the comparison panel. */
const COMPARE_METHODS: SimRequest['method'][] = ['rk4', 'verlet', 'euler']

export interface Comparison {
  times: number[]
  series: { method: string; drift: number[] }[]
  finalDrift: { method: string; value: number }[]
}

/** Center + half-range used to map simulation coordinates into the 3D scene. */
export interface SceneTransform {
  center: [number, number, number]
  halfRange: number
}

function computeTransform(result: SimResponse): SceneTransform {
  const dim = result.dim
  const min = [Infinity, Infinity, Infinity]
  const max = [-Infinity, -Infinity, -Infinity]
  for (const frame of result.positions) {
    for (const body of frame) {
      for (let k = 0; k < dim; k++) {
        if (body[k] < min[k]) min[k] = body[k]
        if (body[k] > max[k]) max[k] = body[k]
      }
    }
  }
  const center: [number, number, number] = [0, 0, 0]
  let halfRange = 0
  for (let k = 0; k < dim; k++) {
    center[k] = (max[k] + min[k]) / 2
    halfRange = Math.max(halfRange, (max[k] - min[k]) / 2)
  }
  if (!isFinite(halfRange) || halfRange === 0) halfRange = 1
  return { center, halfRange }
}

interface State {
  presets: PresetOut[]
  request: SimRequest | null
  result: SimResponse | null
  transform: SceneTransform | null
  selectedPresetId: string | null
  loading: boolean
  error: string | null

  // Playback
  frame: number
  playing: boolean

  // View
  viewMode: ViewMode

  // Method comparison
  comparison: Comparison | null
  comparing: boolean

  loadPresets: () => Promise<void>
  selectPreset: (id: string) => void
  updateRequest: (patch: Partial<SimRequest>) => void
  run: () => Promise<void>
  setFrame: (frame: number) => void
  setPlaying: (playing: boolean) => void

  // Body editing
  addBody: () => void
  updateBody: (index: number, patch: Partial<BodyIn>) => void
  removeBody: (index: number) => void
  setDimension: (dim: 2 | 3) => void

  setViewMode: (mode: ViewMode) => void
  runComparison: () => Promise<void>
  clearComparison: () => void
}

/** Spatial dimension of the current request (2 or 3). */
function requestDim(req: SimRequest | null): 2 | 3 {
  const n = req?.bodies?.[0]?.position?.length
  return n === 3 ? 3 : 2
}

/** Resize a vector to `dim` components (pad with 0 / truncate). */
function resize(vec: number[], dim: number): number[] {
  const out = vec.slice(0, dim)
  while (out.length < dim) out.push(0)
  return out
}

export const useStore = create<State>((set, get) => ({
  presets: [],
  request: null,
  result: null,
  transform: null,
  selectedPresetId: null,
  loading: false,
  error: null,
  frame: 0,
  playing: false,
  viewMode: '3D',
  comparison: null,
  comparing: false,

  loadPresets: async () => {
    try {
      const presets = await getPresets()
      set({ presets })
      if (presets.length > 0 && !get().selectedPresetId) {
        get().selectPreset(presets[0].id)
      }
    } catch (e) {
      set({ error: (e as Error).message })
    }
  },

  selectPreset: (id) => {
    const preset = get().presets.find((p) => p.id === id)
    if (preset) {
      const request = { ...preset.request, bodies: preset.request.bodies.map((b) => ({ ...b })) }
      set({
        selectedPresetId: id,
        request,
        viewMode: requestDim(request) === 3 ? '3D' : '2D',
        comparison: null,
      })
    }
  },

  updateRequest: (patch) => {
    const current = get().request
    if (current) set({ request: { ...current, ...patch } })
  },

  run: async () => {
    const request = get().request
    if (!request) return
    set({ loading: true, error: null, playing: false })
    try {
      const result = await simulate(request)
      set({
        result,
        transform: computeTransform(result),
        frame: 0,
        playing: true,
        loading: false,
      })
    } catch (e) {
      set({ error: (e as Error).message, loading: false })
    }
  },

  setFrame: (frame) => set({ frame, playing: false }),
  setPlaying: (playing) => set({ playing }),

  addBody: () => {
    const req = get().request
    if (!req) return
    const dim = requestDim(req)
    const i = req.bodies.length
    const body: BodyIn = {
      mass: 1e24,
      position: dim === 3 ? [1e8, 0, 0] : [1e8, 0],
      velocity: dim === 3 ? [0, 10, 0] : [0, 10],
      name: `Body ${i + 1}`,
      color: PALETTE[i % PALETTE.length],
    }
    set({ request: { ...req, bodies: [...req.bodies, body] }, comparison: null })
  },

  updateBody: (index, patch) => {
    const req = get().request
    if (!req) return
    const bodies = req.bodies.map((b, i) => (i === index ? { ...b, ...patch } : b))
    set({ request: { ...req, bodies }, comparison: null })
  },

  removeBody: (index) => {
    const req = get().request
    if (!req || req.bodies.length <= 2) return // engine requires >= 2
    set({
      request: { ...req, bodies: req.bodies.filter((_, i) => i !== index) },
      comparison: null,
    })
  },

  setDimension: (dim) => {
    const req = get().request
    if (!req) return
    const bodies = req.bodies.map((b) => ({
      ...b,
      position: resize(b.position, dim),
      velocity: resize(b.velocity, dim),
    }))
    set({
      request: { ...req, bodies },
      viewMode: dim === 3 ? '3D' : '2D',
      comparison: null,
    })
  },

  setViewMode: (mode) => set({ viewMode: mode }),

  runComparison: async () => {
    const request = get().request
    if (!request) return
    set({ comparing: true, error: null })
    try {
      const results = await Promise.all(
        COMPARE_METHODS.map((method) => simulate({ ...request, method })),
      )
      const times = results[0].energies.total.map((_, i) => i)
      const series = results.map((res, i) => {
        const total = res.energies.total
        const e0 = total[0] || 1
        return {
          method: COMPARE_METHODS[i].toUpperCase(),
          drift: total.map((e) => Math.abs((e - e0) / e0) * 100),
        }
      })
      const finalDrift = series.map((s) => ({
        method: s.method,
        value: s.drift[s.drift.length - 1],
      }))
      set({ comparison: { times, series, finalDrift }, comparing: false })
    } catch (e) {
      set({ error: (e as Error).message, comparing: false })
    }
  },

  clearComparison: () => set({ comparison: null }),
}))
