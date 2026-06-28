import { Routes, Route } from 'react-router-dom'
import LandingPage from './pages/LandingPage'
import Simulator from './pages/Simulator'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/simulator" element={<Simulator />} />
    </Routes>
  )
}
