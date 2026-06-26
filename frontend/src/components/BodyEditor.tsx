import { useStore } from '../store'

const AXES = ['x', 'y', 'z']

export default function BodyEditor() {
  const { request, addBody, updateBody, removeBody, setDimension } = useStore()
  if (!request) return null

  const dim = request.bodies[0]?.position.length === 3 ? 3 : 2

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
              className="body-name"
              value={body.name}
              onChange={(e) => updateBody(i, { name: e.target.value })}
            />
            <input
              type="color"
              value={body.color}
              onChange={(e) => updateBody(i, { color: e.target.value })}
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

          <label className="mass">
            mass (kg)
            <input
              type="number"
              value={body.mass}
              onChange={(e) => updateBody(i, { mass: Number(e.target.value) })}
            />
          </label>

          <div className="vec-grid">
            <span className="vec-label">pos</span>
            {body.position.map((v, k) => (
              <input
                key={`p${k}`}
                type="number"
                title={`position ${AXES[k]}`}
                value={v}
                onChange={(e) => {
                  const next = body.position.slice()
                  next[k] = Number(e.target.value)
                  updateBody(i, { position: next })
                }}
              />
            ))}
            <span className="vec-label">vel</span>
            {body.velocity.map((v, k) => (
              <input
                key={`v${k}`}
                type="number"
                title={`velocity ${AXES[k]}`}
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
