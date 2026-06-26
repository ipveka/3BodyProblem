import { create } from 'zustand'
import {
  getPresets,
  simulate,
  type PresetOut,
  type SimRequest,
  type SimResponse,
} from './api/client'

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

  loadPresets: () => Promise<void>
  selectPreset: (id: string) => void
  updateRequest: (patch: Partial<SimRequest>) => void
  run: () => Promise<void>
  setFrame: (frame: number) => void
  setPlaying: (playing: boolean) => void
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
      set({ selectedPresetId: id, request: { ...preset.request } })
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
}))
