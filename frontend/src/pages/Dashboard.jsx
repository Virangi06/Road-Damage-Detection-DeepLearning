import { useFetch } from '../hooks/useFetch'
import { api } from '../utils/api'
import {
  StatCard, Card, SectionTitle, PageHeader,
  LoadingScreen, ErrorBox, Badge, ProgressBar,
} from '../components/ui'
import {
  BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis,
  Tooltip, ResponsiveContainer, Legend,
} from 'recharts'
import {
  Images, Tag, Zap, Target, TrendingUp,
  AlertTriangle, CheckCircle2, Clock, LayoutDashboard,
} from 'lucide-react'
import { useNavigate } from 'react-router-dom'

// ── Pastel chart palette ─────────────────────────────────────────────────────
const CHART_COLORS = ['#87CEEB', '#7DD3C0', '#A78BFA', '#F59E0B', '#F87171']
const SEV_COLORS   = { Low: '#10B981', Medium: '#F59E0B', High: '#EF4444' }

// ── Custom tooltip ───────────────────────────────────────────────────────────
function ChartTip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-white border border-brand-border rounded-xl shadow-card px-3 py-2 text-xs">
      {label && <p className="font-semibold text-brand-dark mb-1">{label}</p>}
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color }} className="font-medium">
          {p.name}: {typeof p.value === 'number' ? p.value.toLocaleString() : p.value}
        </p>
      ))}
    </div>
  )
}

export default function Dashboard() {
  const { data, loading, error, refetch } = useFetch(api.stats)
  const navigate = useNavigate()

  if (loading) return <LoadingScreen message="Loading dashboard…" />
  if (error)   return <ErrorBox message={error} onRetry={refetch} />

  const { dataset = {}, models = {}, history = {} } = data || {}

  // Class distribution chart data
  const classData = Object.entries(dataset.class_counts || {}).map(([k, v]) => ({
    name: k.replace(' (', '\n('),
    count: v,
    short: k.split(' ')[0],
  }))

  // Country data
  const countryData = Object.entries(dataset.country_counts || {}).map(([k, v]) => ({
    name: k, value: v,
  }))

  // Model comparison data
  const modelData = [
    { name: 'CNN\nBaseline', accuracy: 89.0, f1: 63.3, color: '#87CEEB' },
    { name: 'ViT\nBaseline', accuracy: 92.0, f1: 69.5, color: '#7DD3C0' },
    { name: 'Hybrid\nCNN+ViT', accuracy: 87.0, f1: 60.2, color: '#A78BFA' },
  ]

  // Severity distribution from history
  const sevMap = { Low: 0, Medium: 0, High: 0 }
  // We don't have per-sev from /stats directly so derive from known history counts
  const sevData = [
    { name: 'Low',    value: Math.max(1, history.total_analyzed - history.damage_detected - history.high_severity), fill: SEV_COLORS.Low },
    { name: 'Medium', value: Math.max(0, history.damage_detected - history.high_severity), fill: SEV_COLORS.Medium },
    { name: 'High',   value: history.high_severity || 0, fill: SEV_COLORS.High },
  ].filter(d => d.value > 0)

  // Box size data
  const boxData = Object.entries(dataset.box_sizes || {}).map(([k, v]) => ({
    name: k, value: v,
  }))

  return (
    <div className="space-y-6 animate-fade-in">
      <PageHeader
        title="Dashboard"
        subtitle="Road damage detection system overview — RDD2022"
        icon={LayoutDashboard}
        action={
          <button
            onClick={() => navigate('/analyze')}
            className="inline-flex items-center gap-2 px-4 py-2 bg-sky-500 hover:bg-sky-600 text-white text-sm font-semibold rounded-xl shadow-sm transition-all duration-200"
          >
            <AlertTriangle size={15} />
            Analyze Image
          </button>
        }
      />

      {/* ── KPI row ── */}
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-5 gap-4">
        <StatCard
          label="Total Dataset Images"
          value={dataset.total_images?.toLocaleString() ?? '—'}
          sub={`${dataset.train_images?.toLocaleString()} train`}
          icon={Images}
          color="sky"
        />
        <StatCard
          label="Total Annotations"
          value={dataset.total_annotations?.toLocaleString() ?? '—'}
          sub={`${dataset.num_classes} damage classes`}
          icon={Tag}
          color="purple"
        />
        <StatCard
          label="Best Model Accuracy"
          value={`${models.best_accuracy ?? 92}%`}
          sub={models.best_model ?? 'ViT Baseline'}
          icon={Target}
          color="green"
        />
        <StatCard
          label="Best Macro F1"
          value={models.best_macro_f1 ?? '0.6953'}
          sub="ViT Baseline"
          icon={TrendingUp}
          color="indigo"
        />
        <StatCard
          label="Fastest Inference"
          value={`${models.fastest_fps ?? 80.6} FPS`}
          sub="CNN Baseline (12.4 ms)"
          icon={Zap}
          color="amber"
        />
      </div>

      {/* ── History strip (only when analyses exist) ── */}
      {history.total_analyzed > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard label="Images Analyzed"   value={history.total_analyzed}  icon={CheckCircle2} color="sky" />
          <StatCard label="Damage Detected"   value={history.damage_detected} icon={AlertTriangle} color="amber" />
          <StatCard label="High Severity"     value={history.high_severity}   icon={AlertTriangle} color="red" />
          <StatCard label="Avg Severity Score" value={`${history.avg_severity_score}/100`} icon={Clock} color="purple" />
        </div>
      )}

      {/* ── Charts row 1 ── */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">

        {/* Class distribution bar */}
        <Card className="lg:col-span-3 p-5">
          <SectionTitle>Training Set — Class Distribution</SectionTitle>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={classData} layout="vertical" barCategoryGap="20%">
              <XAxis type="number" tick={{ fontSize: 11, fill: '#64748B' }} axisLine={false} tickLine={false} />
              <YAxis type="category" dataKey="short" tick={{ fontSize: 11, fill: '#64748B' }} width={62} axisLine={false} tickLine={false} />
              <Tooltip content={<ChartTip />} />
              <Bar dataKey="count" name="Annotations" radius={[0, 6, 6, 0]}>
                {classData.map((_, i) => <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>

        {/* Country donut */}
        <Card className="lg:col-span-2 p-5">
          <SectionTitle>Images by Country</SectionTitle>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie
                data={countryData} dataKey="value" nameKey="name"
                cx="50%" cy="50%" innerRadius={55} outerRadius={85}
                paddingAngle={3}
              >
                {countryData.map((_, i) => <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />)}
              </Pie>
              <Tooltip content={<ChartTip />} />
              <Legend iconType="circle" iconSize={8}
                wrapperStyle={{ fontSize: 11, color: '#64748B' }} />
            </PieChart>
          </ResponsiveContainer>
        </Card>
      </div>

      {/* ── Charts row 2 ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">

        {/* Model comparison */}
        <Card className="p-5">
          <SectionTitle sub="Validation accuracy & macro F1 (×100)">Model Comparison</SectionTitle>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={modelData} barGap={4}>
              <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#64748B' }} axisLine={false} tickLine={false} />
              <YAxis domain={[0, 100]} tick={{ fontSize: 11, fill: '#64748B' }} axisLine={false} tickLine={false} />
              <Tooltip content={<ChartTip />} />
              <Legend iconType="circle" iconSize={8}
                wrapperStyle={{ fontSize: 11, color: '#64748B' }} />
              <Bar dataKey="accuracy" name="Accuracy (%)" fill="#87CEEB" radius={[4,4,0,0]} />
              <Bar dataKey="f1"       name="Macro F1 ×100" fill="#A78BFA" radius={[4,4,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        {/* Box size donut OR severity if history */}
        <Card className="p-5">
          {history.total_analyzed > 0 ? (
            <>
              <SectionTitle>Severity Distribution (History)</SectionTitle>
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={sevData} dataKey="value" nameKey="name"
                    cx="50%" cy="50%" innerRadius={55} outerRadius={85} paddingAngle={3}>
                    {sevData.map((d, i) => <Cell key={i} fill={d.fill} />)}
                  </Pie>
                  <Tooltip content={<ChartTip />} />
                  <Legend iconType="circle" iconSize={8}
                    wrapperStyle={{ fontSize: 11, color: '#64748B' }} />
                </PieChart>
              </ResponsiveContainer>
            </>
          ) : (
            <>
              <SectionTitle>Bounding Box Size Distribution</SectionTitle>
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={boxData} dataKey="value" nameKey="name"
                    cx="50%" cy="50%" innerRadius={55} outerRadius={85} paddingAngle={3}>
                    {boxData.map((_, i) => <Cell key={i} fill={CHART_COLORS[i]} />)}
                  </Pie>
                  <Tooltip content={<ChartTip />} />
                  <Legend iconType="circle" iconSize={8}
                    wrapperStyle={{ fontSize: 11, color: '#64748B' }} />
                </PieChart>
              </ResponsiveContainer>
            </>
          )}
        </Card>
      </div>

      {/* ── How it works strip ── */}
      <Card className="p-5">
        <SectionTitle sub="End-to-end pipeline">How the System Works</SectionTitle>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 mt-1">
          {[
            { step: '01', label: 'Upload Image',     desc: 'Road photograph provided', color: 'bg-sky-100 text-sky-600' },
            { step: '02', label: 'Detect Damage',    desc: 'Faster R-CNN localises boxes', color: 'bg-purple-100 text-purple-600' },
            { step: '03', label: 'Classify Type',    desc: 'Hybrid CNN+ViT classifies', color: 'bg-teal-100 text-teal-600' },
            { step: '04', label: 'Assess Severity',  desc: '0–100 score computed', color: 'bg-amber-100 text-amber-600' },
            { step: '05', label: 'View Results',     desc: 'Annotated image & report', color: 'bg-emerald-100 text-emerald-600' },
          ].map(s => (
            <div key={s.step} className="flex flex-col items-center text-center p-3 rounded-xl bg-brand-bg">
              <span className={`text-xs font-bold px-2 py-0.5 rounded-full mb-2 ${s.color}`}>{s.step}</span>
              <p className="text-xs font-semibold text-brand-dark">{s.label}</p>
              <p className="text-xs text-brand-muted mt-0.5 leading-snug">{s.desc}</p>
            </div>
          ))}
        </div>
      </Card>

      {/* ── Severity guide ── */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {[
          { label: 'Low Severity',    range: 'Score 0 – 29.9',  desc: 'Minor cracks, routine monitoring.',  bg: 'bg-emerald-50', border: 'border-emerald-200', text: 'text-emerald-700', dot: 'bg-emerald-500' },
          { label: 'Medium Severity', range: 'Score 30 – 64.9', desc: 'Multiple cracks, schedule repair.',   bg: 'bg-amber-50',   border: 'border-amber-200',   text: 'text-amber-700',   dot: 'bg-amber-400' },
          { label: 'High Severity',   range: 'Score 65 – 100',  desc: 'Potholes / alligator cracking, immediate action.', bg: 'bg-red-50', border: 'border-red-200', text: 'text-red-700', dot: 'bg-red-500' },
        ].map(s => (
          <div key={s.label} className={`${s.bg} border ${s.border} rounded-2xl p-4`}>
            <div className="flex items-center gap-2 mb-1">
              <span className={`w-2 h-2 rounded-full ${s.dot}`} />
              <p className={`text-sm font-bold ${s.text}`}>{s.label}</p>
            </div>
            <p className={`text-xs font-medium ${s.text} opacity-80`}>{s.range}</p>
            <p className="text-xs text-brand-muted mt-1">{s.desc}</p>
          </div>
        ))}
      </div>
    </div>
  )
}
