import { Link } from 'react-router-dom'

export default function LandingPage() {
  return (
    <div className="landing">
      <header className="landing-hero">
        <div className="hero-inner">
          <div className="hero-badge">🌌 Gravitational Dynamics</div>
          <h1>The N-Body Simulator</h1>
          <p className="hero-sub">
            Watch gravity choreograph planets, moons, and stars. Configure your own
            celestial systems and integrate Newton's laws of motion in real time —
            in 2D or 3D.
          </p>
          <div className="hero-cta">
            <Link to="/simulator" className="btn-primary">🚀 Launch Simulator</Link>
            <a href="#learn" className="btn-ghost">Learn the physics ↓</a>
          </div>
        </div>
      </header>

      <main id="learn" className="landing-body">
        <section className="section">
          <h2>What is the N-body problem?</h2>
          <p>
            The <strong>N-body problem</strong> asks a deceptively simple question:
            given a set of masses, their starting positions, and their starting
            velocities, how do they move under their mutual gravity over time?
          </p>
          <p>
            Every body pulls on every other body according to Newton's law of
            universal gravitation:
          </p>
          <div className="formula">F = G · (m₁ · m₂) / r²</div>
          <p>
            The force grows with the masses and falls off with the square of the
            distance between them. Each body feels the combined pull of all the
            others, and that total force continuously changes its motion.
          </p>
        </section>

        <section className="section">
          <h2>Why three bodies are famously hard</h2>
          <p>
            For <strong>two</strong> bodies, the problem is solved exactly — orbits
            are tidy ellipses (Kepler's laws). But add a <strong>third</strong> body
            and there is no general closed-form solution. The motion becomes
            <em> chaotic</em>: tiny changes in the starting conditions snowball into
            wildly different futures. This is the celebrated{' '}
            <strong>three-body problem</strong>, and the only way to explore it is to
            simulate it step by step.
          </p>
          <div className="callout">
            A handful of special three-body solutions <em>are</em> stable and
            beautiful — like the <strong>figure-eight orbit</strong>, where three
            equal masses chase each other along a single looping path. You can run it
            in the simulator.
          </div>
        </section>

        <section className="section">
          <h2>How this simulator works</h2>
          <div className="cards">
            <div className="card">
              <div className="card-icon">∑</div>
              <h3>Computes forces</h3>
              <p>
                At every instant it sums the gravitational pull between all pairs of
                bodies to find each one's acceleration.
              </p>
            </div>
            <div className="card">
              <div className="card-icon">∫</div>
              <h3>Integrates motion</h3>
              <p>
                It steps the system forward in time using numerical integrators —
                <strong> RK4</strong> (accurate), <strong>Verlet</strong> (energy
                conserving), or <strong>Euler</strong> (simple).
              </p>
            </div>
            <div className="card">
              <div className="card-icon">⚡</div>
              <h3>Checks energy</h3>
              <p>
                Total energy should stay constant. The app tracks{' '}
                <strong>energy drift</strong> so you can see how faithful each method
                is — compare them side by side.
              </p>
            </div>
          </div>
        </section>

        <section className="section">
          <h2>Things to try</h2>
          <ul className="try-list">
            <li><strong>Earth–Moon</strong> — a clean circular two-body orbit.</li>
            <li><strong>Sun–Earth–Moon</strong> — a hierarchical three-body system.</li>
            <li><strong>Figure-Eight</strong> — the famous periodic choreography.</li>
            <li><strong>Inclined Orbit (3D)</strong> — tilt the orbit out of plane.</li>
            <li><strong>Build your own</strong> — add bodies, set masses and
              velocities, and switch between 2D and 3D.</li>
          </ul>
        </section>

        <section className="section section-cta">
          <h2>Ready to explore?</h2>
          <p>Pick a preset or design your own gravitational system.</p>
          <Link to="/simulator" className="btn-primary">🚀 Launch Simulator</Link>
        </section>
      </main>

      <footer className="landing-footer">
        N-Body Simulator · Newtonian gravity, integrated numerically · Built with
        FastAPI, React &amp; three.js
      </footer>
    </div>
  )
}
