import { useMemo } from 'react'
import {
  Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, Legend,
} from 'recharts'
import { useStore } from '../store'

export default function EnergyChart() {
  const result = useStore((s) => s.result)

  const data = useMemo(() => {
    if (!result) return []
    const { times, energies } = result
    // Normalize by the initial |total| so the three curves share a scale and
    // energy drift is visible regardless of the system's absolute magnitude.
    const ref = Math.abs(energies.total[0]) || 1
    // Downsample to keep the chart light for long runs.
    const step = Math.max(1, Math.floor(times.length / 400))
    const rows = []
    for (let i = 0; i < times.length; i += step) {
      rows.push({
        t: times[i],
        kinetic: energies.kinetic[i] / ref,
        potential: energies.potential[i] / ref,
        total: energies.total[i] / ref,
      })
    }
    return rows
  }, [result])

  if (!result) return null

  return (
    <div className="chart">
      <h2>Energy (normalized)</h2>
      <ResponsiveContainer width="100%" height={180}>
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
          <XAxis dataKey="t" hide />
          <YAxis tick={{ fontSize: 10, fill: '#9aa3c0' }} stroke="#2a3050" width={40} />
          <Tooltip
            contentStyle={{ background: '#11142a', border: '1px solid #2a3050', fontSize: 12 }}
            labelStyle={{ color: '#9aa3c0' }}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Line type="monotone" dataKey="kinetic" stroke="#4dd0e1" dot={false} strokeWidth={1.5} />
          <Line type="monotone" dataKey="potential" stroke="#ff8a65" dot={false} strokeWidth={1.5} />
          <Line type="monotone" dataKey="total" stroke="#aed581" dot={false} strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
