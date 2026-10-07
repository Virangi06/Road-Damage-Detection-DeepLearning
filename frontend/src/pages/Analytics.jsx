import { useState } from 'react'
import { useFetch } from '../hooks/useFetch'
import { api } from '../utils/api'
import {
  Card, SectionTitle, PageHeader, LoadingScreen,
  ErrorBox, Badge,
} from '../components/ui'
import {
  LineChart, Line, BarChart, Bar, ScatterChart, Scatter,
  XAxis, YAxis, Tooltip, Legend, ResponsiveContainer,
  CartesianGrid, Cell, ReferenceLine,
} from 'recharts'
import { BarChart2 } from 'lucide-react'

const COLORS   = ['#87CEEB', '#7DD3C0', '#A78BFA']
const CLS_COLORS = ['#87CEEB', '#7DD3C0', '#A78BFA', '#F59E0B', '#F87171']

function ChartTip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-white border border-brand-border rounded-xl shadow-card px-3 py-2 text-xs min-w-[120px]">
      {label !== undefined && <p className="font-semibold text-brand-dark mb-1">Epoch {label}</p>}
      {payload.map((p, i) => (
        <p key={i} className="font-medium" style={{ color: p.color }}>
          {p.name}: {typeof p.value === 'number' ? p.value.toFixed(4) : p.value}
        </p>
      ))}
    </div>
  )
}

const TABS = ['Model Comparison', 'Training Curves', 'Per-Class F1', 'Efficiency']

const MODEL_KEYS = [
  'CNN Baseline (EfficientNet-B0)',
  'ViT Baseline (vit_tiny)',
  'Hybrid CNN+ViT (Proposed)',
]
const MODEL_SHORT = ['CNN', 'ViT', 'Hybrid']
const CLASS_NAMES = [
  'D00 Longitudinal', 'D10 Transverse',
  'D20 Alligator', 'D40 Pothole', 'D43/D44 Other',
]
const CLASS_KEYS = [
  'D00 (Longitudinal Crack)', 'D10 (Transverse Crack)',
  'D20 (Alligator Crack)', 'D40 (Pothole)', 'D43/D44 (Other Damage)',
]

export default function Analytics() {
  const [tab, setTab] = useState(0)
  const { data, loading, error, refetch } = useFetch(api.modelInfo)

  if (loading) return <LoadingScreen message="Loading analytics…" />
  if (error)   return <ErrorBox message={error} onRetry={refetch} />

  const bench = data?.benchmark || {}
  const th    = data?.training_history || {}
  const dh    = data?.detection_history || {}

  // ── Build comparison table data ────────────────────────────────────────────
  const compRows = MODEL_KEYS.map((key, i) => {
    const m = bench[key]?.metrics || {}
    const p = bench[key]?.parameters || {}
    const e = bench[key]?.efficiency || {}
    return {
      name: MODEL_SHORT[i], fullName: key,
      accuracy:   m.accuracy   ?? 0,
      macroF1:    m.macro_f1   ?? 0,
      precision:  m.macro_precision ?? 0,
      recall:     m.macro_recall    ?? 0,
      params:     p.total_params_m  ?? 0,
      latency:    e.latency_ms_per_crop ?? 0,
      fps:        e.fps              ?? 0,
      fileSize:   p.file_size_mb     ?? 0,
    }
  })

  // ── Per-class F1 bar chart data ────────────────────────────────────────────
  const clsData = CLASS_KEYS.map((cls, ci) => {
    const row = { name: CLASS_NAMES[ci] }
    MODEL_KEYS.forEach((key, mi) => {
      row[MODEL_SHORT[mi]] = bench[key]?.class_breakdown?.[cls]?.f1_score ?? 0
    })
    return row
  })

  // ── Training curves ────────────────────────────────────────────────────────
  const buildCurve = (modelKey) => {
    const d = th[modelKey] || {}
    return (d.epochs || []).map((ep, i) => ({
      epoch:      ep,
      train_loss: d.train_loss?.[i] ?? 0,
      val_loss:   d.val_loss?.[i]   ?? 0,
      train_acc:  d.train_acc?.[i]  ?? 0,
      val_acc:    d.val_acc?.[i]    ?? 0,
    }))
  }

  // ── Efficiency scatter data ────────────────────────────────────────────────
  const effData = compRows.map((r, i) => ({
    name: r.name, x: r.latency, y: r.accuracy, z: r.params, color: COLORS[i],
  }))

  // ── Detection training (Phase 2) ──────────────────────────────────────────
  const det2 = (dh.epochs || []).map((ep, i) => ({
    epoch:     ep,
    train_loss: dh.train_loss?.[i] ?? 0,
    val_loss:   dh.val_loss?.[i]   ?? 0,
  }))

  return (
    <div className="space-y-5 animate-fade-in">
      <PageHeader
        title="Analytics & Performance"
        subtitle="Quantitative benchmark comparison of CNN, ViT, and Hybrid CNN+ViT architectures"
        icon={BarChart2}
      />

      {/* Tab bar */}
      <div className="flex gap-1 p-1 bg-sky-50 rounded-xl border border-brand-border w-fit">
        {TABS.map((t, i) => (
          <button
            key={t}
            onClick={() => setTab(i)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-150
              ${tab === i ? 'bg-white text-sky-700 shadow-sm' : 'text-brand-muted hover:text-sky-600'}`}
          >
            {t}
          </button>
        ))}
      </div>

      {/* ── Tab 0: Model Comparison ── */}
      {tab === 0 && (
        <div className="space-y-5">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            {/* Accuracy bar */}
            <Card className="p-5">
              <SectionTitle>Validation Accuracy (%)</SectionTitle>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={compRows} barCategoryGap="35%">
                  <CartesianGrid vertical={false} stroke="#EEF9FF" />
                  <XAxis dataKey="name" tick={{ fontSize: 12, fill: '#64748B' }} axisLine={false} tickLine={false} />
                  <YAxis domain={[80, 95]} tick={{ fontSize: 11, fill: '#64748B' }} axisLine={false} tickLine={false} />
                  <Tooltip content={<ChartTip />} />
                  <Bar dataKey="accuracy" name="Accuracy %" radius={[6,6,0,0]}>
                    {compRows.map((_, i) => <Cell key={i} fill={COLORS[i]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>

            {/* Macro F1 bar */}
            <Card className="p-5">
              <SectionTitle>Macro F1-Score</SectionTitle>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={compRows} barCategoryGap="35%">
                  <CartesianGrid vertical={false} stroke="#EEF9FF" />
                  <XAxis dataKey="name" tick={{ fontSize: 12, fill: '#64748B' }} axisLine={false} tickLine={false} />
                  <YAxis domain={[0, 0.8]} tick={{ fontSize: 11, fill: '#64748B' }} axisLine={false} tickLine={false} />
                  <Tooltip content={<ChartTip />} />
                  <Bar dataKey="macroF1" name="Macro F1" radius={[6,6,0,0]}>
                    {compRows.map((_, i) => <Cell key={i} fill={COLORS[i]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>

          {/* Full metrics table */}
          <Card className="p-5 overflow-x-auto">
            <SectionTitle>Full Benchmark Table</SectionTitle>
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-sky-50 text-brand-muted text-xs font-semibold uppercase tracking-wide">
                  {['Architecture','Accuracy','Macro P','Macro R','Macro F1','Params','Size','Latency','FPS'].map(h => (
                    <th key={h} className="px-3 py-2.5 text-left first:rounded-l-xl last:rounded-r-xl">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {compRows.map((r, i) => (
                  <tr key={r.name} className="border-t border-brand-border hover:bg-sky-50/50 transition-colors">
                    <td className="px-3 py-2.5 font-semibold text-brand-dark">{r.fullName}</td>
                    <td className="px-3 py-2.5">
                      <Badge variant={r.accuracy >= 91 ? 'success' : 'default'}>{r.accuracy}%</Badge>
                    </td>
                    <td className="px-3 py-2.5 text-brand-muted">{r.precision.toFixed(4)}</td>
                    <td className="px-3 py-2.5 text-brand-muted">{r.recall.toFixed(4)}</td>
                    <td className="px-3 py-2.5 font-semibold text-brand-dark">{r.macroF1.toFixed(4)}</td>
                    <td className="px-3 py-2.5 text-brand-muted">{r.params}M</td>
                    <td className="px-3 py-2.5 text-brand-muted">{r.fileSize} MB</td>
                    <td className="px-3 py-2.5 text-brand-muted">{r.latency} ms</td>
                    <td className="px-3 py-2.5 text-brand-muted">{r.fps}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </div>
      )}

      {/* ── Tab 1: Training Curves ── */}
      {tab === 1 && (
        <div className="space-y-5">
          {/* Phase 2 — Faster R-CNN */}
          <Card className="p-5">
            <SectionTitle sub="Phase 2 — Faster R-CNN ResNet-50 FPN (2 epochs)">
              Detection Training Loss
            </SectionTitle>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={det2}>
                <CartesianGrid strokeDasharray="3 3" stroke="#EEF9FF" />
                <XAxis dataKey="epoch" tick={{ fontSize: 11, fill: '#64748B' }} label={{ value: 'Epoch', position: 'insideBottom', offset: -2, fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11, fill: '#64748B' }} domain={['auto','auto']} />
                <Tooltip content={<ChartTip />} />
                <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 11 }} />
                <Line type="monotone" dataKey="train_loss" name="Train Loss" stroke="#87CEEB" strokeWidth={2} dot={{ r: 4 }} />
                <Line type="monotone" dataKey="val_loss"   name="Val Loss"   stroke="#A78BFA" strokeWidth={2} dot={{ r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          {/* Phase 3 — loss + acc for all 3 models */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            {[
              { key: 'CNN Baseline',  color: COLORS[0], label: 'CNN Baseline (EfficientNet-B0)' },
              { key: 'ViT Baseline',  color: COLORS[1], label: 'ViT Baseline (vit_tiny)' },
              { key: 'Hybrid CNN+ViT',color: COLORS[2], label: 'Hybrid CNN+ViT (Proposed)' },
            ].map(m => {
              const curve = buildCurve(m.key)
              return (
                <Card key={m.key} className="p-5">
                  <SectionTitle sub={m.label}>Loss & Accuracy Curves</SectionTitle>
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={curve}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#EEF9FF" />
                      <XAxis dataKey="epoch" tick={{ fontSize: 11, fill: '#64748B' }} />
                      <YAxis yAxisId="loss" orientation="left"  tick={{ fontSize: 10, fill: '#64748B' }} domain={[0, 1.2]} />
                      <YAxis yAxisId="acc"  orientation="right" tick={{ fontSize: 10, fill: '#64748B' }} domain={[60, 100]} />
                      <Tooltip content={<ChartTip />} />
                      <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 10 }} />
                      <Line yAxisId="loss" type="monotone" dataKey="train_loss" name="Train Loss" stroke={m.color} strokeWidth={2} strokeDasharray="4 2" dot={{ r: 3 }} />
                      <Line yAxisId="loss" type="monotone" dataKey="val_loss"   name="Val Loss"   stroke={m.color} strokeWidth={2} dot={{ r: 3 }} />
                      <Line yAxisId="acc"  type="monotone" dataKey="val_acc"    name="Val Acc %"  stroke="#F59E0B" strokeWidth={2} dot={{ r: 3 }} />
                    </LineChart>
                  </ResponsiveContainer>
                </Card>
              )
            })}

            {/* Phase 2 table */}
            <Card className="p-5">
              <SectionTitle sub="Phase 2 — Faster R-CNN">Detection Training Log</SectionTitle>
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-sky-50 text-brand-muted text-xs font-semibold">
                    {['Epoch','Train Loss','Val Loss','LR'].map(h => (
                      <th key={h} className="px-3 py-2 text-left">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {[
                    { epoch:1, tl:0.544,  vl:0.3123, lr:'3.00e-4' },
                    { epoch:2, tl:0.3075, vl:0.2661, lr:'2.25e-4', best:true },
                  ].map(r => (
                    <tr key={r.epoch} className={`border-t border-brand-border ${r.best ? 'bg-emerald-50' : ''}`}>
                      <td className="px-3 py-2 font-medium">{r.epoch}</td>
                      <td className="px-3 py-2 text-brand-muted">{r.tl}</td>
                      <td className="px-3 py-2 font-semibold">{r.vl} {r.best && <Badge variant="success" size="sm">Best</Badge>}</td>
                      <td className="px-3 py-2 text-brand-muted font-mono text-xs">{r.lr}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          </div>
        </div>
      )}

      {/* ── Tab 2: Per-Class F1 ── */}
      {tab === 2 && (
        <div className="space-y-5">
          <Card className="p-5">
            <SectionTitle sub="All 5 damage categories">Per-Class F1-Score Comparison</SectionTitle>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={clsData} barCategoryGap="20%">
                <CartesianGrid vertical={false} stroke="#EEF9FF" />
                <XAxis dataKey="name" tick={{ fontSize: 10, fill: '#64748B' }} axisLine={false} tickLine={false} />
                <YAxis domain={[0, 1.05]} tick={{ fontSize: 11, fill: '#64748B' }} axisLine={false} tickLine={false} />
                <Tooltip content={<ChartTip />} />
                <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 11 }} />
                {MODEL_SHORT.map((s, i) => (
                  <Bar key={s} dataKey={s} name={s} fill={COLORS[i]} radius={[4,4,0,0]} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* D43/D44 note */}
          <div className="bg-amber-50 border border-amber-200 rounded-2xl p-4 text-sm text-amber-800">
            <strong>⚠️ D43/D44 (Other Damage): F1 = 0.0 across all models.</strong>{' '}
            Root causes: class imbalance (4,628 vs 18,201 samples for D00), high intra-class visual heterogeneity, 3-epoch training limit.
          </div>

          {/* Detailed table */}
          <Card className="p-5 overflow-x-auto">
            <SectionTitle>Detailed Per-Class Metrics</SectionTitle>
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-sky-50 text-brand-muted font-semibold uppercase tracking-wide">
                  {['Class','CNN P','CNN R','CNN F1','ViT P','ViT R','ViT F1','Hybrid P','Hybrid R','Hybrid F1'].map(h => (
                    <th key={h} className="px-3 py-2 text-left">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {CLASS_KEYS.map((cls, ci) => {
                  const rows = MODEL_KEYS.map(k => bench[k]?.class_breakdown?.[cls] || { precision: 0, recall: 0, f1_score: 0 })
                  return (
                    <tr key={cls} className="border-t border-brand-border hover:bg-sky-50/40 transition-colors">
                      <td className="px-3 py-2 font-semibold text-brand-dark text-xs whitespace-nowrap">{CLASS_NAMES[ci]}</td>
                      {rows.map((r, ri) => (
                        <>
                          <td key={`p${ri}`} className="px-3 py-2 text-brand-muted">{r.precision.toFixed(3)}</td>
                          <td key={`r${ri}`} className="px-3 py-2 text-brand-muted">{r.recall.toFixed(3)}</td>
                          <td key={`f${ri}`} className={`px-3 py-2 font-bold ${r.f1_score > 0.8 ? 'text-emerald-600' : r.f1_score > 0.4 ? 'text-amber-600' : 'text-red-500'}`}>
                            {r.f1_score.toFixed(3)}
                          </td>
                        </>
                      ))}
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </Card>
        </div>
      )}

      {/* ── Tab 3: Efficiency ── */}
      {tab === 3 && (
        <div className="space-y-5">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            <Card className="p-5">
              <SectionTitle sub="Lower latency = faster inference">Accuracy vs. Latency (ms/crop)</SectionTitle>
              <ResponsiveContainer width="100%" height={260}>
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" stroke="#EEF9FF" />
                  <XAxis dataKey="x" name="Latency (ms)" type="number" tick={{ fontSize: 11, fill: '#64748B' }} label={{ value: 'Latency (ms)', position: 'insideBottom', offset: -5, fontSize: 11 }} domain={[0, 40]} />
                  <YAxis dataKey="y" name="Accuracy (%)" type="number" tick={{ fontSize: 11, fill: '#64748B' }} domain={[84, 95]} label={{ value: 'Accuracy %', angle: -90, position: 'insideLeft', fontSize: 11 }} />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} content={({ active, payload }) => {
                    if (!active || !payload?.[0]) return null
                    const d = payload[0].payload
                    return (
                      <div className="bg-white border border-brand-border rounded-xl shadow-card p-3 text-xs">
                        <p className="font-bold text-brand-dark">{d.name}</p>
                        <p className="text-brand-muted">Latency: {d.x} ms</p>
                        <p className="text-brand-muted">Accuracy: {d.y}%</p>
                        <p className="text-brand-muted">Params: {d.z}M</p>
                      </div>
                    )
                  }} />
                  {effData.map((d, i) => (
                    <Scatter key={i} data={[d]} name={d.name} fill={d.color}>
                      <Cell fill={d.color} />
                    </Scatter>
                  ))}
                  <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 11 }} />
                </ScatterChart>
              </ResponsiveContainer>
            </Card>

            <Card className="p-5">
              <SectionTitle sub="Smaller = more efficient">Accuracy vs. Parameters (Millions)</SectionTitle>
              <ResponsiveContainer width="100%" height={260}>
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" stroke="#EEF9FF" />
                  <XAxis dataKey="x" name="Parameters (M)" type="number" tick={{ fontSize: 11, fill: '#64748B' }} label={{ value: 'Params (M)', position: 'insideBottom', offset: -5, fontSize: 11 }} domain={[2, 12]} />
                  <YAxis dataKey="y" name="Accuracy (%)" type="number" tick={{ fontSize: 11, fill: '#64748B' }} domain={[84, 95]} />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} content={({ active, payload }) => {
                    if (!active || !payload?.[0]) return null
                    const d = payload[0].payload
                    return (
                      <div className="bg-white border border-brand-border rounded-xl shadow-card p-3 text-xs">
                        <p className="font-bold">{d.name}</p>
                        <p className="text-brand-muted">Params: {d.z}M</p>
                        <p className="text-brand-muted">Accuracy: {d.y}%</p>
                      </div>
                    )
                  }} />
                  {effData.map((d, i) => (
                    <Scatter key={i} data={[{ ...d, x: d.z }]} name={d.name} fill={d.color} />
                  ))}
                  <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 11 }} />
                </ScatterChart>
              </ResponsiveContainer>
            </Card>
          </div>

          {/* Efficiency table */}
          <Card className="p-5 overflow-x-auto">
            <SectionTitle>Architecture Efficiency Summary</SectionTitle>
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-sky-50 text-brand-muted text-xs font-semibold uppercase tracking-wide">
                  {['Architecture','Accuracy (%)','Params (M)','File (MB)','Latency (ms)','FPS'].map(h => (
                    <th key={h} className="px-3 py-2.5 text-left">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {compRows.map(r => (
                  <tr key={r.name} className="border-t border-brand-border hover:bg-sky-50/50 transition-colors">
                    <td className="px-3 py-2.5 font-semibold text-brand-dark">{r.fullName}</td>
                    <td className="px-3 py-2.5"><Badge variant={r.accuracy >= 91 ? 'success' : 'default'}>{r.accuracy}%</Badge></td>
                    <td className="px-3 py-2.5 text-brand-muted">{r.params}M</td>
                    <td className="px-3 py-2.5 text-brand-muted">{r.fileSize} MB</td>
                    <td className="px-3 py-2.5 text-brand-muted">{r.latency} ms</td>
                    <td className="px-3 py-2.5 text-brand-muted">{r.fps}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </div>
      )}
    </div>
  )
}
