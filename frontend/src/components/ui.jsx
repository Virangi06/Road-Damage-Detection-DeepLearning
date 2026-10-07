/**
 * components/ui.jsx
 * Centralized reusable UI primitives — Button, Card, Badge, Spinner,
 * Input, SectionTitle, EmptyState, Skeleton, Toast.
 */
import { AlertCircle, CheckCircle2, Info, X } from 'lucide-react'

// ── Button ──────────────────────────────────────────────────────────────────
export function Button({
  children, onClick, variant = 'primary', size = 'md',
  disabled = false, loading = false, className = '', type = 'button', icon,
}) {
  const base = 'inline-flex items-center justify-center gap-2 font-semibold rounded-xl transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-sky-400 focus:ring-offset-1 disabled:opacity-50 disabled:cursor-not-allowed select-none'
  const variants = {
    primary:   'bg-sky-500 hover:bg-sky-600 active:bg-sky-700 text-white shadow-sm hover:shadow-md',
    secondary: 'bg-white hover:bg-sky-50 active:bg-sky-100 text-sky-700 border border-brand-border hover:border-sky-400 shadow-sm',
    ghost:     'bg-transparent hover:bg-sky-50 text-sky-700',
    danger:    'bg-red-500 hover:bg-red-600 text-white shadow-sm',
    success:   'bg-emerald-500 hover:bg-emerald-600 text-white shadow-sm',
  }
  const sizes = {
    sm:  'px-3 py-1.5 text-sm',
    md:  'px-4 py-2 text-sm',
    lg:  'px-6 py-2.5 text-base',
    xl:  'px-8 py-3 text-base',
  }
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled || loading}
      className={`${base} ${variants[variant]} ${sizes[size]} ${className}`}
    >
      {loading ? <Spinner size={4} color="currentColor" /> : icon}
      {children}
    </button>
  )
}

// ── Card ────────────────────────────────────────────────────────────────────
export function Card({ children, className = '', hover = false, padding = true }) {
  return (
    <div className={`bg-white rounded-2xl border border-brand-border shadow-card ${hover ? 'card-hover cursor-pointer' : ''} ${padding ? 'p-5' : ''} ${className}`}>
      {children}
    </div>
  )
}

// ── KPI / Stat Card ─────────────────────────────────────────────────────────
export function StatCard({ label, value, sub, icon: Icon, color = 'sky', trend }) {
  const colorMap = {
    sky:     'bg-sky-100 text-sky-600',
    green:   'bg-emerald-100 text-emerald-600',
    amber:   'bg-amber-100 text-amber-600',
    red:     'bg-red-100 text-red-500',
    purple:  'bg-purple-100 text-purple-600',
    indigo:  'bg-indigo-100 text-indigo-600',
  }
  return (
    <Card className="flex items-start gap-4 card-hover">
      {Icon && (
        <div className={`w-11 h-11 rounded-xl flex items-center justify-center flex-shrink-0 ${colorMap[color]}`}>
          <Icon size={22} />
        </div>
      )}
      <div className="flex-1 min-w-0">
        <p className="text-xs font-semibold text-brand-muted uppercase tracking-wide truncate">{label}</p>
        <p className="text-2xl font-bold text-brand-dark mt-0.5 leading-tight">{value}</p>
        {sub && <p className="text-xs text-brand-muted mt-1">{sub}</p>}
        {trend && (
          <p className={`text-xs font-medium mt-1 ${trend.up ? 'text-emerald-600' : 'text-red-500'}`}>
            {trend.up ? '↑' : '↓'} {trend.label}
          </p>
        )}
      </div>
    </Card>
  )
}

// ── Badge ───────────────────────────────────────────────────────────────────
export function Badge({ children, variant = 'default', size = 'sm' }) {
  const variants = {
    default:  'bg-sky-100 text-sky-700',
    success:  'bg-emerald-100 text-emerald-700',
    warning:  'bg-amber-100 text-amber-700',
    danger:   'bg-red-100 text-red-600',
    purple:   'bg-purple-100 text-purple-700',
    gray:     'bg-gray-100 text-gray-600',
    low:      'bg-emerald-100 text-emerald-700',
    medium:   'bg-amber-100 text-amber-700',
    high:     'bg-red-100 text-red-600',
  }
  const sizes = { sm: 'px-2.5 py-0.5 text-xs', md: 'px-3 py-1 text-sm' }
  return (
    <span className={`inline-flex items-center gap-1 font-semibold rounded-full ${variants[variant]} ${sizes[size]}`}>
      {children}
    </span>
  )
}

// ── Severity Badge helper ────────────────────────────────────────────────────
export function SeverityBadge({ category }) {
  const map = {
    Low:    { variant: 'low',    dot: 'bg-emerald-500', label: 'Low' },
    Medium: { variant: 'medium', dot: 'bg-amber-400',   label: 'Medium' },
    High:   { variant: 'high',   dot: 'bg-red-500',     label: 'High' },
  }
  const { variant = 'default', dot = 'bg-sky-400', label = category } = map[category] || {}
  return (
    <Badge variant={variant}>
      <span className={`w-1.5 h-1.5 rounded-full ${dot}`} />
      {label}
    </Badge>
  )
}

// ── Spinner ─────────────────────────────────────────────────────────────────
export function Spinner({ size = 6, color = 'text-sky-500', className = '' }) {
  return (
    <svg
      className={`w-${size} h-${size} animate-spin ${color} ${className}`}
      xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"
    >
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
    </svg>
  )
}

// ── Loading full-page ────────────────────────────────────────────────────────
export function LoadingScreen({ message = 'Loading…' }) {
  return (
    <div className="flex flex-col items-center justify-center h-64 gap-4">
      <Spinner size={8} />
      <p className="text-brand-muted text-sm">{message}</p>
    </div>
  )
}

// ── Section Title ────────────────────────────────────────────────────────────
export function SectionTitle({ children, sub, action }) {
  return (
    <div className="flex items-center justify-between mb-4">
      <div>
        <h2 className="text-base font-bold text-brand-dark">{children}</h2>
        {sub && <p className="text-xs text-brand-muted mt-0.5">{sub}</p>}
      </div>
      {action && <div>{action}</div>}
    </div>
  )
}

// ── Page Header ──────────────────────────────────────────────────────────────
export function PageHeader({ title, subtitle, icon: Icon, action }) {
  return (
    <div className="flex items-start justify-between mb-6">
      <div className="flex items-center gap-3">
        {Icon && (
          <div className="w-10 h-10 rounded-xl bg-sky-100 flex items-center justify-center text-sky-600 flex-shrink-0">
            <Icon size={22} />
          </div>
        )}
        <div>
          <h1 className="text-xl font-bold text-brand-dark">{title}</h1>
          {subtitle && <p className="text-sm text-brand-muted mt-0.5">{subtitle}</p>}
        </div>
      </div>
      {action && <div className="flex-shrink-0">{action}</div>}
    </div>
  )
}

// ── Input ────────────────────────────────────────────────────────────────────
export function Input({ label, error, className = '', ...props }) {
  return (
    <div className="flex flex-col gap-1">
      {label && <label className="text-xs font-semibold text-brand-muted uppercase tracking-wide">{label}</label>}
      <input
        className={`w-full px-3.5 py-2.5 rounded-xl border border-brand-border bg-white text-brand-dark text-sm placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-sky-400 focus:border-sky-400 transition-all duration-150 ${error ? 'border-red-400' : ''} ${className}`}
        {...props}
      />
      {error && <p className="text-xs text-red-500">{error}</p>}
    </div>
  )
}

// ── Select ───────────────────────────────────────────────────────────────────
export function Select({ label, error, children, className = '', ...props }) {
  return (
    <div className="flex flex-col gap-1">
      {label && <label className="text-xs font-semibold text-brand-muted uppercase tracking-wide">{label}</label>}
      <select
        className={`w-full px-3.5 py-2.5 rounded-xl border border-brand-border bg-white text-brand-dark text-sm focus:outline-none focus:ring-2 focus:ring-sky-400 focus:border-sky-400 transition-all duration-150 ${className}`}
        {...props}
      >
        {children}
      </select>
      {error && <p className="text-xs text-red-500">{error}</p>}
    </div>
  )
}

// ── Empty State ──────────────────────────────────────────────────────────────
export function EmptyState({ icon: Icon, title, subtitle, action }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 px-8 text-center">
      {Icon && (
        <div className="w-16 h-16 rounded-2xl bg-sky-50 flex items-center justify-center text-sky-400 mb-4">
          <Icon size={32} />
        </div>
      )}
      <p className="text-base font-semibold text-brand-dark">{title}</p>
      {subtitle && <p className="text-sm text-brand-muted mt-1 max-w-xs">{subtitle}</p>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  )
}

// ── Error Box ────────────────────────────────────────────────────────────────
export function ErrorBox({ message, onRetry }) {
  return (
    <div className="bg-red-50 border border-red-200 rounded-2xl p-5 flex gap-3 items-start">
      <AlertCircle size={20} className="text-red-500 flex-shrink-0 mt-0.5" />
      <div className="flex-1">
        <p className="text-sm font-semibold text-red-700">Something went wrong</p>
        <p className="text-sm text-red-600 mt-1">{message}</p>
        {onRetry && (
          <button onClick={onRetry} className="mt-3 text-sm font-semibold text-red-700 underline">
            Try again
          </button>
        )}
      </div>
    </div>
  )
}

// ── Alert Banner ─────────────────────────────────────────────────────────────
export function Alert({ type = 'info', children, onClose }) {
  const styles = {
    info:    { bg: 'bg-sky-50 border-sky-200',   icon: <Info size={16} className="text-sky-600" />,   text: 'text-sky-800' },
    success: { bg: 'bg-emerald-50 border-emerald-200', icon: <CheckCircle2 size={16} className="text-emerald-600" />, text: 'text-emerald-800' },
    warning: { bg: 'bg-amber-50 border-amber-200', icon: <AlertCircle size={16} className="text-amber-600" />, text: 'text-amber-800' },
    error:   { bg: 'bg-red-50 border-red-200',   icon: <AlertCircle size={16} className="text-red-600" />,   text: 'text-red-800' },
  }
  const s = styles[type]
  return (
    <div className={`flex items-start gap-2.5 p-3.5 rounded-xl border ${s.bg} ${s.text} text-sm`}>
      <span className="flex-shrink-0 mt-0.5">{s.icon}</span>
      <span className="flex-1">{children}</span>
      {onClose && (
        <button onClick={onClose} className="flex-shrink-0 opacity-60 hover:opacity-100">
          <X size={14} />
        </button>
      )}
    </div>
  )
}

// ── Skeleton row ─────────────────────────────────────────────────────────────
export function Skeleton({ className = '', rows = 1 }) {
  return (
    <div className="space-y-2">
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className={`skeleton h-4 ${className}`} />
      ))}
    </div>
  )
}

// ── Progress bar ─────────────────────────────────────────────────────────────
export function ProgressBar({ value, max = 100, color = 'bg-sky-400', className = '' }) {
  const pct = Math.min(100, Math.max(0, (value / max) * 100))
  return (
    <div className={`w-full bg-sky-100 rounded-full h-2 overflow-hidden ${className}`}>
      <div
        className={`h-full rounded-full transition-all duration-500 ${color}`}
        style={{ width: `${pct}%` }}
      />
    </div>
  )
}

// ── Divider ──────────────────────────────────────────────────────────────────
export function Divider({ className = '' }) {
  return <hr className={`border-brand-border ${className}`} />
}

// ── Table helpers ─────────────────────────────────────────────────────────────
export function Table({ children, className = '' }) {
  return (
    <div className={`overflow-x-auto rounded-2xl border border-brand-border ${className}`}>
      <table className="w-full text-sm">{children}</table>
    </div>
  )
}
export function Th({ children, className = '' }) {
  return (
    <th className={`px-4 py-3 text-left text-xs font-semibold text-brand-muted uppercase tracking-wide bg-sky-50 first:rounded-tl-2xl last:rounded-tr-2xl ${className}`}>
      {children}
    </th>
  )
}
export function Td({ children, className = '' }) {
  return (
    <td className={`px-4 py-3 text-brand-dark border-t border-brand-border ${className}`}>
      {children}
    </td>
  )
}
