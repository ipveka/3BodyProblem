import { Link } from 'react-router-dom'

export default function LandingPage() {
  return (
    <div className="landing">
      <header className="landing-hero">
        <div className="hero-inner">
          <div className="hero-badge">🌌 A classic problem in physics &amp; mathematics</div>
          <h1>The Three-Body Problem</h1>
          <p className="hero-sub">
            Given three objects that attract each other through gravity, can we
            predict exactly how they will move over time? Explore it yourself —
            configure gravitational systems and watch them evolve in 2D or 3D.
          </p>
          <div className="hero-cta">
            <Link to="/simulator" className="btn-primary">🚀 Launch Simulator</Link>
            <a href="#learn" className="btn-ghost">Learn the physics ↓</a>
          </div>
        </div>
      </header>

      <main id="learn" className="landing-body">
        <section className="section">
          <h2>The question</h2>
          <p>
            The <strong>three-body problem</strong> is a classic problem in physics
            and mathematics. It asks something that sounds simple: given three
            objects that attract each other through gravity, can we{' '}
            <strong>predict exactly</strong> how they will move over time?
          </p>
          <p>
            Take the <strong>Sun, the Earth, and the Moon</strong> — they all pull on
            one another. If we know their starting positions, masses, and
            velocities, can we write a formula that tells us exactly where each one
            will be at any moment in the future?
          </p>
        </section>

        <section className="section">
          <h2>Two bodies vs. three</h2>
          <div className="compare">
            <div className="compare-card good">
              <div className="compare-tag">Two bodies</div>
              <h3>Predictable</h3>
              <p>
                For two bodies — like the Earth orbiting the Sun in a simplified
                model — the problem has a clean, exact solution: <strong>ellipses</strong>,
                stable orbits, motion you can predict far into the future
                (Kepler's laws).
              </p>
            </div>
            <div className="compare-card bad">
              <div className="compare-tag">Three bodies</div>
              <h3>Often chaotic</h3>
              <p>
                Add a third body and it becomes far harder. In most cases there is{' '}
                <strong>no simple general formula</strong>. The system can become{' '}
                <strong>chaotic</strong>: tiny differences in the starting conditions
                lead to massively different outcomes later.
              </p>
            </div>
          </div>
          <div className="callout">
            The key idea: <strong>two bodies are predictable; three bodies are often
            chaotic.</strong> Because there's usually no formula, the way to explore
            the three-body problem is to <em>simulate</em> it — step by step, which is
            exactly what this app does.
          </div>
        </section>

        <section className="section">
          <h2>The math underneath</h2>
          <p>
            Every body pulls on every other body according to Newton's law of
            universal gravitation:
          </p>
          <div className="formula">F = G · (m₁ · m₂) / r²</div>
          <p>
            The force grows with the masses and falls off with the square of the
            distance. Each body feels the combined pull of all the others, and that
            total force continuously changes its motion. With three or more bodies
            those pulls interact in ways that (almost always) have no closed-form
            answer — so we solve the equations numerically instead.
          </p>
          <div className="callout">
            A few special three-body solutions <em>are</em> perfectly stable and
            beautiful — like the <strong>figure-eight orbit</strong>, where three
            equal masses chase each other along one looping path. You can run it in
            the simulator.
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
                It steps the system forward in time with numerical integrators —
                <strong> RK4</strong> (accurate), <strong>Verlet</strong> (energy
                conserving), or <strong>Euler</strong> (simple).
              </p>
            </div>
            <div className="card">
              <div className="card-icon">⚡</div>
              <h3>Checks energy</h3>
              <p>
                Total energy should stay constant. The app tracks{' '}
                <strong>energy drift</strong> so you can judge how faithful each
                method is — and compare them side by side.
              </p>
            </div>
          </div>
        </section>

        <section className="section">
          <h2>Things to try</h2>
          <ul className="try-list">
            <li><strong>Earth–Moon</strong> — a clean, predictable two-body orbit.</li>
            <li><strong>Sun–Earth–Moon</strong> — the real three-body system to scale.</li>
            <li><strong>Figure-Eight</strong> — a rare, perfectly periodic three-body dance.</li>
            <li><strong>Inclined Orbit (3D)</strong> — tilt an orbit out of plane.</li>
            <li><strong>Build your own</strong> — add bodies, set masses and
              velocities, and nudge the start to see chaos appear.</li>
          </ul>
        </section>

        <section className="section section-cta">
          <h2>See it for yourself</h2>
          <p>Pick a preset or design your own gravitational system.</p>
          <Link to="/simulator" className="btn-primary">🚀 Launch Simulator</Link>
        </section>
      </main>

      <footer className="landing-footer">
        The Three-Body Problem · Newtonian gravity, integrated numerically · Built
        with FastAPI, React &amp; three.js
      </footer>
    </div>
  )
}
