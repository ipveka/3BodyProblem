import { useStore } from '../store'

const AXES = ['x', 'y', 'z']

export default function BodyEditor() {
  const { request, addBody, updateBody, removeBody, setDimension } = useStore()
  if (!request) return null

  const dim = request.bodies[0]?.position.length === 3 ? 3 : 2
  const normalized = (request.length_unit ?? 1000) === 1
  const massUnit = normalized ? '' : ' (kg)'
  const posUnit = normalized ? '' : ' (km)'
  const velUnit = normalized ? '' : ' (km/s)'

  return (
    <div className="editor">
      <div className="editor-head">
        <h2>Bodies ({request.bodies.length})</h2>
        <div className="dim-toggle">
          {([2, 3] as const).map((d) => (
            <button
              key={d}
              className={dim === d ? 'active' : ''}
              onClick={() => setDimension(d)}
            >
              {d}D
            </button>
          ))}
        </div>
      </div>

      {request.bodies.map((body, i) => (
        <div className="body-card" key={i} style={{ borderLeftColor: body.color }}>
          <div className="body-row">
            <input
              type="color"
              className="body-color"
              value={body.color}
              title="Color"
              onChange={(e) => updateBody(i, { color: e.target.value })}
            />
            <input
              className="body-name"
              value={body.name}
              onChange={(e) => updateBody(i, { name: e.target.value })}
            />
            <button
              className="remove"
              title="Remove body"
              disabled={request.bodies.length <= 2}
              onClick={() => removeBody(i)}
            >
              ✕
            </button>
          </div>

          <label className="mass-field">
            <span>mass{massUnit}</span>
            <input
              type="number"
              value={body.mass}
              onChange={(e) => updateBody(i, { mass: Number(e.target.value) })}
            />
          </label>

          <div
            className="vectors"
            style={{ gridTemplateColumns: `4.2rem repeat(${dim}, minmax(0, 1fr))` }}
          >
            <span className="axis-corner" />
            {AXES.slice(0, dim).map((a) => (
              <span key={a} className="axis-head">{a}</span>
            ))}

            <span className="vec-name">pos{posUnit}</span>
            {body.position.map((v, k) => (
              <input
                key={`p${k}`}
                type="number"
                value={v}
                onChange={(e) => {
                  const next = body.position.slice()
                  next[k] = Number(e.target.value)
                  updateBody(i, { position: next })
                }}
              />
            ))}

            <span className="vec-name">vel{velUnit}</span>
            {body.velocity.map((v, k) => (
              <input
                key={`v${k}`}
                type="number"
                value={v}
                onChange={(e) => {
                  const next = body.velocity.slice()
                  next[k] = Number(e.target.value)
                  updateBody(i, { velocity: next })
                }}
              />
            ))}
          </div>
        </div>
      ))}

      <button className="add-body" onClick={() => addBody()}>＋ Add body</button>
    </div>
  )
}
