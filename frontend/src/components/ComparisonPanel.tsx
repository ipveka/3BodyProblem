import {
  Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, Legend,
} from 'recharts'
import { useStore } from '../store'

const COLORS: Record<string, string> = {
  RK4: '#2ecc71',
  VERLET: '#3498db',
  EULER: '#e74c3c',
}

export default function ComparisonPanel() {
  const { comparison, comparing, runComparison, request } = useStore()

  // Reshape series into recharts rows: [{ i, RK4, VERLET, EULER }, ...]
  const data: Record<string, number>[] = []
  if (comparison) {
    const step = Math.max(1, Math.floor(comparison.times.length / 300))
    for (let i = 0; i < comparison.times.length; i += step) {
      const row: Record<string, number> = { i }
      for (const s of comparison.series) row[s.method] = s.drift[i]
      data.push(row)
    }
  }

  return (
    <div className="comparison">
      <div className="comparison-head">
        <h2>Method Comparison</h2>
        <button onClick={() => runComparison()} disabled={comparing || !request}>
          {comparing ? 'Running…' : 'Compare RK4 / Verlet / Euler'}
        </button>
      </div>

      {comparison && (
        <>
          <p className="hint">Relative energy drift over the run (lower is better).</p>
          <ResponsiveContainer width="100%" height={170}>
            <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
              <XAxis dataKey="i" hide />
              <YAxis
                width={48}
                tick={{ fontSize: 10, fill: '#9aa3c0' }}
                stroke="#2a3050"
                tickFormatter={(v) => `${v.toFixed(2)}%`}
              />
              <Tooltip
                contentStyle={{ background: '#11142a', border: '1px solid #2a3050', fontSize: 12 }}
                formatter={(v: number) => `${v.toExponential(2)}%`}
              />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              {comparison.series.map((s) => (
                <Line key={s.method} type="monotone" dataKey={s.method}
                  stroke={COLORS[s.method] ?? '#aaa'} dot={false} strokeWidth={1.5} />
              ))}
            </LineChart>
          </ResponsiveContainer>
          <table className="drift-table">
            <thead><tr><th>Method</th><th>Final drift</th></tr></thead>
            <tbody>
              {comparison.finalDrift.map((d) => (
                <tr key={d.method}>
                  <td><span className="dot" style={{ background: COLORS[d.method] }} />{d.method}</td>
                  <td>{d.value.toExponential(3)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  )
}
