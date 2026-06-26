import { useMemo, useRef } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { Line, OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import { useStore, type SceneTransform } from '../store'
import type { SimResponse } from '../api/client'

type Vec3 = [number, number, number]

/** Map a simulation position (2D or 3D) into normalized scene coordinates. */
function toScene(p: number[], t: SceneTransform, dim: number): Vec3 {
  const x = (p[0] - t.center[0]) / t.halfRange
  const y = (p[1] - t.center[1]) / t.halfRange
  const z = dim === 3 ? (p[2] - t.center[2]) / t.halfRange : 0
  return [x, y, z]
}

function Trajectories({ result, transform }: { result: SimResponse; transform: SceneTransform }) {
  const paths = useMemo(
    () =>
      result.bodies.map((body, i) => ({
        color: body.color,
        points: result.positions.map((frame) => toScene(frame[i], transform, result.dim)),
      })),
    [result, transform],
  )

  return (
    <>
      {paths.map((p, i) => (
        <Line key={i} points={p.points} color={p.color} lineWidth={1.5} transparent opacity={0.6} />
      ))}
    </>
  )
}

function Bodies({ result, transform }: { result: SimResponse; transform: SceneTransform }) {
  const groupRef = useRef<THREE.Group>(null)
  const meshes = useRef<(THREE.Mesh | null)[]>([])

  // Advance playback and position the body meshes every animation frame.
  useFrame(() => {
    const { result: r, playing, frame, setFrameSilently } = readStore()
    if (!r) return
    let f = frame
    if (playing) {
      f = (frame + 1) % r.positions.length
      setFrameSilently(f)
    }
    const positions = r.positions[f]
    for (let i = 0; i < positions.length; i++) {
      const m = meshes.current[i]
      if (m) {
        const [x, y, z] = toScene(positions[i], transform, r.dim)
        m.position.set(x, y, z)
      }
    }
  })

  return (
    <group ref={groupRef}>
      {result.bodies.map((b, i) => (
        <mesh key={i} ref={(el) => { meshes.current[i] = el }}>
          <sphereGeometry args={[0.035, 24, 24]} />
          <meshStandardMaterial color={b.color} emissive={b.color} emissiveIntensity={0.4} />
        </mesh>
      ))}
    </group>
  )
}

// Read store imperatively inside the render loop without re-subscribing.
function readStore() {
  const s = useStore.getState()
  return {
    result: s.result,
    playing: s.playing,
    frame: s.frame,
    // Update the frame without flipping `playing` off (setFrame pauses).
    setFrameSilently: (f: number) => useStore.setState({ frame: f }),
  }
}

export default function Viewer3D() {
  const result = useStore((s) => s.result)
  const transform = useStore((s) => s.transform)

  return (
    <Canvas camera={{ position: [2.2, 1.6, 2.2], fov: 50 }} dpr={[1, 2]}>
      <color attach="background" args={['#0b0d17']} />
      <ambientLight intensity={0.6} />
      <pointLight position={[5, 5, 5]} intensity={1.2} />
      <pointLight position={[-5, -3, -5]} intensity={0.4} />
      {result && transform && (
        <>
          <Trajectories result={result} transform={transform} />
          <Bodies result={result} transform={transform} />
        </>
      )}
      <gridHelper args={[4, 16, '#1e2440', '#151a2e']} />
      <OrbitControls enablePan enableZoom enableRotate />
    </Canvas>
  )
}
