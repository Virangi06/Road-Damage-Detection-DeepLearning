import { useState, useEffect } from 'react'
import { Outlet, NavLink, useLocation } from 'react-router-dom'
import {
  LayoutDashboard, Search, BarChart2, Clock, Info,
  Menu, X, AlertTriangle, ChevronRight,
} from 'lucide-react'

const NAV = [
  { to: '/dashboard',  label: 'Dashboard',    icon: LayoutDashboard },
  { to: '/analyze',    label: 'Analyze Image', icon: Search },
  { to: '/analytics',  label: 'Analytics',     icon: BarChart2 },
  { to: '/history',    label: 'History',       icon: Clock },
  { to: '/model-info', label: 'Model Info',    icon: Info },
]

function NavItem({ to, label, icon: Icon, collapsed, onClick }) {
  return (
    <NavLink
      to={to}
      onClick={onClick}
      className={({ isActive }) =>
        `flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-150 group
         ${isActive
           ? 'bg-sky-100 text-sky-700 shadow-sm'
           : 'text-brand-muted hover:bg-sky-50 hover:text-sky-700'}`
      }
    >
      <Icon size={18} className="flex-shrink-0" />
      {!collapsed && <span className="truncate">{label}</span>}
      {!collapsed && (
        <ChevronRight
          size={14}
          className="ml-auto opacity-0 group-hover:opacity-60 transition-opacity"
        />
      )}
    </NavLink>
  )
}

export default function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [collapsed,   setCollapsed]   = useState(false)
  const location = useLocation()

  // Close mobile drawer on route change
  useEffect(() => { setSidebarOpen(false) }, [location.pathname])

  // Close on resize to lg
  useEffect(() => {
    const handler = () => { if (window.innerWidth >= 1024) setSidebarOpen(false) }
    window.addEventListener('resize', handler)
    return () => window.removeEventListener('resize', handler)
  }, [])

  const sidebarW = collapsed ? 'w-16' : 'w-60'

  return (
    <div className="flex h-screen overflow-hidden bg-brand-bg font-sans">

      {/* ── Mobile overlay ── */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-20 bg-black/20 backdrop-blur-sm lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* ── Sidebar ── */}
      <aside
        className={`
          fixed inset-y-0 left-0 z-30 flex flex-col bg-white border-r border-brand-border
          transition-all duration-300 ease-in-out
          ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}
          lg:relative lg:translate-x-0 lg:flex-shrink-0
          ${sidebarW}
        `}
      >
        {/* Brand */}
        <div className={`flex items-center gap-3 px-4 py-5 border-b border-brand-border ${collapsed ? 'justify-center' : ''}`}>
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-sky-400 to-sky-600 flex items-center justify-center flex-shrink-0 shadow-sm">
            <AlertTriangle size={16} className="text-white" />
          </div>
          {!collapsed && (
            <div className="overflow-hidden">
              <p className="text-sm font-bold text-brand-dark leading-tight truncate">Road Damage AI</p>
              <p className="text-xs text-brand-muted truncate">RDD2022 System</p>
            </div>
          )}
          {/* Collapse toggle — desktop only */}
          <button
            onClick={() => setCollapsed(c => !c)}
            className="hidden lg:flex ml-auto text-brand-muted hover:text-sky-600 p-1 rounded-lg hover:bg-sky-50 transition-colors flex-shrink-0"
            title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            <Menu size={16} />
          </button>
        </div>

        {/* Nav links */}
        <nav className="flex-1 overflow-y-auto p-3 space-y-1">
          {NAV.map(item => (
            <NavItem
              key={item.to}
              {...item}
              collapsed={collapsed}
              onClick={() => setSidebarOpen(false)}
            />
          ))}
        </nav>

        {/* Footer */}
        {!collapsed && (
          <div className="px-4 py-4 border-t border-brand-border">
            <div className="bg-sky-50 rounded-xl p-3">
              <p className="text-xs font-semibold text-sky-700">Tech Stack</p>
              <p className="text-xs text-sky-600 mt-1 leading-relaxed">
                PyTorch · EfficientNet-B0<br />
                ViT-Tiny · Hybrid CNN+ViT<br />
                RDD2022 · 38,385 images
              </p>
            </div>
          </div>
        )}
      </aside>

      {/* ── Main content area ── */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">

        {/* Top bar */}
        <header className="flex-shrink-0 flex items-center gap-3 px-4 py-3.5 bg-white border-b border-brand-border">
          {/* Mobile menu button */}
          <button
            onClick={() => setSidebarOpen(s => !s)}
            className="lg:hidden p-2 rounded-xl text-brand-muted hover:bg-sky-50 hover:text-sky-600 transition-colors"
          >
            {sidebarOpen ? <X size={20} /> : <Menu size={20} />}
          </button>

          {/* Page title — matches active nav */}
          <div className="flex-1 min-w-0">
            <Breadcrumb />
          </div>

          {/* API status dot */}
          <StatusDot />
        </header>

        {/* Page content */}
        <main className="flex-1 overflow-y-auto p-4 md:p-6">
          <div className="page-enter max-w-7xl mx-auto">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  )
}

function Breadcrumb() {
  const location = useLocation()
  const match = NAV.find(n => location.pathname.startsWith(n.to))
  if (!match) return null
  const Icon = match.icon
  return (
    <div className="flex items-center gap-2 text-sm font-semibold text-brand-dark">
      <Icon size={16} className="text-sky-500" />
      <span>{match.label}</span>
    </div>
  )
}

function StatusDot() {
  const [ok, setOk] = useState(null)
  useEffect(() => {
    fetch('/api/health')
      .then(r => r.json())
      .then(d => setOk(d.status === 'ok'))
      .catch(() => setOk(false))
  }, [])
  if (ok === null) return null
  return (
    <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${ok ? 'bg-emerald-50 text-emerald-700' : 'bg-red-50 text-red-600'}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${ok ? 'bg-emerald-500 animate-pulse' : 'bg-red-400'}`} />
      {ok ? 'API Online' : 'API Offline'}
    </div>
  )
}
