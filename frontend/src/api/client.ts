import type { components } from './schema'

export type SimRequest = components['schemas']['SimRequest']
export type SimResponse = components['schemas']['SimResponse']
export type PresetOut = components['schemas']['PresetOut']
export type BodyIn = components['schemas']['BodyIn']

const BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (body?.detail) {
        detail = typeof body.detail === 'string'
          ? body.detail
          : JSON.stringify(body.detail)
      }
    } catch {
      /* ignore non-JSON error bodies */
    }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

export function getPresets(): Promise<PresetOut[]> {
  return request<PresetOut[]>('/api/presets')
}

export function simulate(req: SimRequest): Promise<SimResponse> {
  return request<SimResponse>('/api/simulate', {
    method: 'POST',
    body: JSON.stringify(req),
  })
}
