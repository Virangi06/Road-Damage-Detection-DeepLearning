import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Analyze from './pages/Analyze'
import Analytics from './pages/Analytics'
import History from './pages/History'
import ModelInfo from './pages/ModelInfo'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Navigate to="/dashboard" replace />} />
          <Route path="dashboard"  element={<Dashboard />} />
          <Route path="analyze"    element={<Analyze />} />
          <Route path="analytics"  element={<Analytics />} />
          <Route path="history"    element={<History />} />
          <Route path="model-info" element={<ModelInfo />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
