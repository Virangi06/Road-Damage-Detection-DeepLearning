import { useState, useMemo } from 'react'
import { useFetch } from '../hooks/useFetch'
import { api } from '../utils/api'
import {
  Card, SectionTitle, PageHeader, LoadingScreen,
  ErrorBox, EmptyState, SeverityBadge, Badge, Button, Input, Select,
} from '../components/ui'
import {
  AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, CartesianGrid,
} from 'recharts'
import {
  Clock, Trash2, RefreshCw, CheckCircle2, AlertTriangle,
  TrendingUp, Search,
} from 'lucide-react'

const SEV_COLORS = { Low: '#10B981', Medium: '#F59E0B', High: '#EF4444' }

function fmt(ts) {
  try { return new Date(ts).toLocaleString('en-IN', { dateStyle: 'medium', timeStyle: 'short' }) }
  catch { return ts?.slice(0, 16) || '—' }
}

function ChartTip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-white border border-brand-border rounded-xl shadow-card px-3 py-2 text-xs">
      {label && <p className="font-semibold text-brand-dark mb-1">{label}</p>}
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.stroke || p.fill }} className="font-medium">
          {p.name}: {typeof p.value === 'number' ? p.value.toFixed(2) : p.value}
        </p>
      ))}
    </div>
  )
}

export default function History() {
  const { data, loading, error, refetch } = useFetch(api.history)
  const [search,    setSearch]    = useState('')
  const [sevFilter, setSevFilter] = useState('All')
  const [dmgFilter, setDmgFilter] = useState('All')
  const [sortBy,    setSortBy]    = useState('Newest')
  const [clearing,  setClearing]  = useState(false)

  const history = data?.history || []

  const filtered = useMemo(() => {
    let h = [...history]
    if (search)             h = h.filter(x => x.image_name?.toLowerCase().includes(search.toLowerCase()))
    if (sevFilter !== 'All') h = h.filter(x => x.severity_category === sevFilter)
    if (dmgFilter === 'Damaged')    h = h.filter(x => x.damage_detected)
    if (dmgFilter === 'No Damage')  h = h.filter(x => !x.damage_detected)
    if (sortBy === 'Newest')        /* already newest-first from API */ ;
    if (sortBy === 'Oldest')        h = [...h].reverse()
    if (sortBy === 'Highest Score') h = [...h].sort((a,b) => (b.severity_score||0) - (a.severity_score||0))
    if (sortBy === 'Lowest Score')  h = [...h].sort((a,b) => (a.severity_score||0) - (b.severity_score||0))
    return h
  }, [history, search, sevFilter, dmgFilter, sortBy])

  const handleClear = async () => {
    if (!confirm('Clear all analysis history?')) return
    setClearing(true)
    try { await api.clearHistory(); refetch() }
    catch (e) { alert(e.message) }
    finally   { setClearing(false) }
  }

  // ── Chart data ─────────────────────────────────────────────────────────────
  const timeline = [...history].reverse().slice(-20).map((h, i) => ({
    i: i + 1,
    label: h.image_name?.replace('.jpg','')?.slice(-12) || String(i+1),
    score: h.severity_score || 0,
  }))

  const sevCounts = { Low: 0, Medium: 0, High: 0 }
  history.forEach(h => { if (sevCounts[h.severity_category] !== undefined) sevCounts[h.severity_category]++ })
  const sevPie = Object.entries(sevCounts).filter(([,v]) => v > 0).map(([k,v]) => ({ name: k, value: v, fill: SEV_COLORS[k] }))

  const typeCounts = {}
  history.forEach(h => (h.damage_types || []).forEach(t => { if (t) typeCounts[t] = (typeCounts[t]||0)+1 }))
  const typeBar = Object.entries(typeCounts).sort(([,a],[,b]) => b-a).map(([k,v]) => ({ name: k, count: v }))

  if (loading) return <LoadingScreen message="Loading history…" />
  if (error)   return <ErrorBox message={error} onRetry={refetch} />

  const total   = history.length
  const damaged = history.filter(h => h.damage_detected).length
  const highs   = history.filter(h => h.severity_category === 'High').length
  const avgScore = total ? (history.reduce((s,h) => s + (h.severity_score||0), 0) / total).toFixed(1) : 0

  return (
    <div className="space-y-5 animate-fade-in">
      <PageHeader
        title="Analysis History"
        subtitle="All previous road image analyses — saved automatically after each inference"
        icon={Clock}
        action={
          <div className="flex gap-2">
            <Button onClick={refetch}   variant="secondary" size="sm" icon={<RefreshCw size={14} />}>Refresh</Button>
            {total > 0 && (
              <Button onClick={handleClear} variant="danger" size="sm" loading={clearing} icon={<Trash2 size={14} />}>
                Clear
              </Button>
            )}
          </div>
        }
      />

      {total === 0 ? (
        <EmptyState
          icon={Clock}
          title="No analysis history yet"
          subtitle="Analyse a road image on the Analyze Image page to see results here."
        />
      ) : (
        <>
          {/* KPI strip */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { label: 'Total Analyses',     value: total,    icon: TrendingUp,    color: 'text-sky-600',     bg: 'bg-sky-50' },
              { label: 'Damage Detected',    value: damaged,  icon: AlertTriangle, color: 'text-amber-600',   bg: 'bg-amber-50' },
              { label: 'High Severity',      value: highs,    icon: AlertTriangle, color: 'text-red-500',     bg: 'bg-red-50' },
              { label: 'Avg Severity Score', value: `${avgScore}/100`, icon: TrendingUp, color: 'text-emerald-600', bg: 'bg-emerald-50' },
            ].map(s => (
              <Card key={s.label} className="flex items-center gap-3 card-hover">
                <div className={`w-10 h-10 rounded-xl ${s.bg} flex items-center justify-center ${s.color} flex-shrink-0`}>
                  <s.icon size={18} />
                </div>
                <div>
                  <p className="text-xs text-brand-muted font-medium">{s.label}</p>
                  <p className="text-xl font-bold text-brand-dark">{s.value}</p>
                </div>
              </Card>
            ))}
          </div>

          {/* Charts row */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <Card className="lg:col-span-2 p-5">
              <SectionTitle sub="Last 20 analyses chronologically">Severity Score Timeline</SectionTitle>
              <ResponsiveContainer width="100%" height={180}>
                <AreaChart data={timeline}>
                  <defs>
                    <linearGradient id="scoreGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor="#87CEEB" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#87CEEB" stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#EEF9FF" />
                  <XAxis dataKey="label" tick={{ fontSize: 9, fill: '#64748B' }} interval="preserveStartEnd" />
                  <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: '#64748B' }} />
                  <Tooltip content={<ChartTip />} />
                  <Area type="monotone" dataKey="score" name="Severity Score"
                    stroke="#87CEEB" strokeWidth={2} fill="url(#scoreGrad)" />
                </AreaChart>
              </ResponsiveContainer>
            </Card>

            <Card className="p-5">
              <SectionTitle>Severity Distribution</SectionTitle>
              <ResponsiveContainer width="100%" height={180}>
                <PieChart>
                  <Pie data={sevPie} dataKey="value" nameKey="name"
                    cx="50%" cy="50%" innerRadius={50} outerRadius={75} paddingAngle={3}>
                    {sevPie.map((d,i) => <Cell key={i} fill={d.fill} />)}
                  </Pie>
                  <Tooltip content={<ChartTip />} />
                  <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 11 }} />
                </PieChart>
              </ResponsiveContainer>
            </Card>
          </div>

          {typeBar.length > 0 && (
            <Card className="p-5">
              <SectionTitle>Damage Type Frequency</SectionTitle>
              <ResponsiveContainer width="100%" height={160}>
                <BarChart data={typeBar} layout="vertical" barCategoryGap="20%">
                  <XAxis type="number" tick={{ fontSize: 10, fill: '#64748B' }} axisLine={false} tickLine={false} />
                  <YAxis type="category" dataKey="name" width={140} tick={{ fontSize: 11, fill: '#64748B' }} axisLine={false} tickLine={false} />
                  <Tooltip content={<ChartTip />} />
                  <Bar dataKey="count" name="Count" fill="#87CEEB" radius={[0,6,6,0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          )}

          {/* Filters */}
          <Card className="p-4">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="relative sm:col-span-1">
                <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-brand-muted" />
                <input
                  type="text"
                  placeholder="Search image name…"
                  value={search}
                  onChange={e => setSearch(e.target.value)}
                  className="w-full pl-8 pr-3 py-2 text-sm rounded-xl border border-brand-border bg-white focus:outline-none focus:ring-2 focus:ring-sky-400 transition-all"
                />
              </div>
              <Select value={sevFilter} onChange={e => setSevFilter(e.target.value)}>
                <option>All</option>
                <option>Low</option><option>Medium</option><option>High</option>
              </Select>
              <Select value={dmgFilter} onChange={e => setDmgFilter(e.target.value)}>
                <option>All</option>
                <option>Damaged</option><option>No Damage</option>
              </Select>
              <Select value={sortBy} onChange={e => setSortBy(e.target.value)}>
                <option>Newest</option><option>Oldest</option>
                <option>Highest Score</option><option>Lowest Score</option>
              </Select>
            </div>
            <p className="text-xs text-brand-muted mt-2">
              Showing <strong>{filtered.length}</strong> of <strong>{total}</strong> records
            </p>
          </Card>

          {/* Table */}
          <Card className="overflow-x-auto p-0">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-sky-50">
                  {['Date & Time','Image','Damage','Regions','Severity','Score','Priority','Condition','Time (ms)'].map(h => (
                    <th key={h} className="px-4 py-3 text-left text-xs font-semibold text-brand-muted uppercase tracking-wide whitespace-nowrap">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filtered.map((h, i) => (
                  <tr key={i} className="border-t border-brand-border hover:bg-sky-50/40 transition-colors">
                    <td className="px-4 py-3 text-xs text-brand-muted whitespace-nowrap">{fmt(h.timestamp)}</td>
                    <td className="px-4 py-3 font-medium text-brand-dark max-w-[160px] truncate">{h.image_name}</td>
                    <td className="px-4 py-3">
                      {h.damage_detected
                        ? <Badge variant="warning"><AlertTriangle size={10} /> Yes</Badge>
                        : <Badge variant="success"><CheckCircle2 size={10} /> No</Badge>}
                    </td>
                    <td className="px-4 py-3 text-brand-muted">{h.regions ?? 0}</td>
                    <td className="px-4 py-3"><SeverityBadge category={h.severity_category || 'Low'} /></td>
                    <td className="px-4 py-3 font-semibold text-brand-dark">{(h.severity_score || 0).toFixed(1)}/100</td>
                    <td className="px-4 py-3 text-brand-muted">{h.repair_priority ?? 1}/10</td>
                    <td className="px-4 py-3 text-xs text-brand-muted max-w-[160px] truncate">{h.overall_condition}</td>
                    <td className="px-4 py-3 text-xs text-brand-muted">{(h.inference_ms || 0).toFixed(0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {filtered.length === 0 && (
              <div className="py-12 text-center text-sm text-brand-muted">
                No records match the current filters.
              </div>
            )}
          </Card>
        </>
      )}
    </div>
  )
}
