import { useState, useRef, useCallback } from 'react'
import { api } from '../utils/api'
import {
  Button, Card, SectionTitle, PageHeader,
  SeverityBadge, Badge, Alert, Spinner, ProgressBar,
} from '../components/ui'
import {
  Upload, X, Search, Download, RotateCcw,
  AlertTriangle, CheckCircle2, Image as ImageIcon,
  ZoomIn, Info,
} from 'lucide-react'

// ── Severity colour helpers ───────────────────────────────────────────────────
const SEV_STYLES = {
  Low:    { bg: 'bg-emerald-50', border: 'border-emerald-200', text: 'text-emerald-700', bar: 'bg-emerald-400' },
  Medium: { bg: 'bg-amber-50',   border: 'border-amber-200',   text: 'text-amber-700',   bar: 'bg-amber-400' },
  High:   { bg: 'bg-red-50',     border: 'border-red-200',     text: 'text-red-700',      bar: 'bg-red-400' },
}

// ── Drop-zone ─────────────────────────────────────────────────────────────────
function DropZone({ onFile }) {
  const inputRef  = useRef()
  const [drag, setDrag] = useState(false)

  const handleDrop = useCallback((e) => {
    e.preventDefault(); setDrag(false)
    const file = e.dataTransfer.files[0]
    if (file) onFile(file)
  }, [onFile])

  const handleFile = (e) => {
    const file = e.target.files[0]
    if (file) onFile(file)
  }

  return (
    <div
      onDragOver={e => { e.preventDefault(); setDrag(true) }}
      onDragLeave={() => setDrag(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
      className={`
        relative flex flex-col items-center justify-center gap-3 p-10
        rounded-2xl border-2 border-dashed cursor-pointer transition-all duration-200
        ${drag
          ? 'border-sky-400 bg-sky-50 scale-[1.01]'
          : 'border-brand-border bg-white hover:border-sky-300 hover:bg-sky-50/50'}
      `}
    >
      <div className={`w-14 h-14 rounded-2xl flex items-center justify-center transition-colors ${drag ? 'bg-sky-100 text-sky-600' : 'bg-sky-50 text-sky-400'}`}>
        <Upload size={26} />
      </div>
      <div className="text-center">
        <p className="text-sm font-semibold text-brand-dark">Drop your road image here</p>
        <p className="text-xs text-brand-muted mt-1">or click to browse — JPG, JPEG, PNG</p>
      </div>
      <p className="text-xs text-brand-muted/70">Max 200 MB</p>
      <input ref={inputRef} type="file" accept=".jpg,.jpeg,.png" onChange={handleFile} className="hidden" />
    </div>
  )
}

// ── Image viewer with toggle ──────────────────────────────────────────────────
function ImageViewer({ original, annotated, label }) {
  const [showAnnotated, setShowAnnotated] = useState(true)
  const [zoomed, setZoomed] = useState(false)

  const current = (showAnnotated && annotated) ? annotated : original

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <p className="text-xs font-semibold text-brand-muted uppercase tracking-wide">{label}</p>
        <div className="flex items-center gap-2">
          {annotated && (
            <div className="flex rounded-lg border border-brand-border overflow-hidden text-xs">
              <button
                onClick={() => setShowAnnotated(false)}
                className={`px-2.5 py-1 font-medium transition-colors ${!showAnnotated ? 'bg-sky-100 text-sky-700' : 'bg-white text-brand-muted hover:bg-sky-50'}`}
              >Original</button>
              <button
                onClick={() => setShowAnnotated(true)}
                className={`px-2.5 py-1 font-medium transition-colors ${showAnnotated ? 'bg-sky-100 text-sky-700' : 'bg-white text-brand-muted hover:bg-sky-50'}`}
              >Detected</button>
            </div>
          )}
          <button onClick={() => setZoomed(z => !z)} className="p-1.5 rounded-lg hover:bg-sky-50 text-brand-muted hover:text-sky-600 transition-colors">
            <ZoomIn size={14} />
          </button>
        </div>
      </div>
      <div className={`relative overflow-hidden rounded-xl border border-brand-border bg-gray-50 transition-all duration-300 ${zoomed ? 'max-h-[600px]' : 'max-h-80'}`}>
        <img src={current} alt="Road analysis" className="w-full object-contain" />
        {showAnnotated && annotated && (
          <div className="absolute top-2 left-2">
            <span className="text-xs bg-black/60 text-white px-2 py-0.5 rounded-full font-medium backdrop-blur-sm">
              AI Detection
            </span>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Detection row ─────────────────────────────────────────────────────────────
function DetectionRow({ index, detection }) {
  const conf = detection.cls_confidence_pct ?? (detection.cls_confidence ?? 0) * 100
  const [x1, y1, x2, y2] = (detection.bbox || [0,0,0,0]).map(Math.round)
  return (
    <div className="flex items-center gap-3 p-3 rounded-xl bg-brand-bg hover:bg-sky-50/60 transition-colors">
      <div className="w-6 h-6 rounded-full bg-sky-100 text-sky-700 text-xs font-bold flex items-center justify-center flex-shrink-0">
        {index}
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-semibold text-brand-dark truncate">{detection.class_name || detection.class_full || '—'}</p>
        <p className="text-xs text-brand-muted">({x1},{y1}) → ({x2},{y2}) · {x2-x1}×{y2-y1}px</p>
      </div>
      <div className="text-right flex-shrink-0">
        <p className="text-sm font-bold text-brand-dark">{conf.toFixed(1)}%</p>
        <p className="text-xs text-brand-muted">confidence</p>
      </div>
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function Analyze() {
  const [file,       setFile]       = useState(null)
  const [preview,    setPreview]    = useState(null)
  const [status,     setStatus]     = useState('idle')   // idle|analyzing|done|error
  const [result,     setResult]     = useState(null)
  const [errorMsg,   setErrorMsg]   = useState(null)
  const [stage,      setStage]      = useState('')

  const handleFile = useCallback((f) => {
    // validate
    const ext = f.name.split('.').pop().toLowerCase()
    if (!['jpg','jpeg','png'].includes(ext)) {
      setErrorMsg('Unsupported file type. Please upload JPG or PNG.')
      return
    }
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setResult(null)
    setErrorMsg(null)
    setStatus('idle')
  }, [])

  const reset = () => {
    setFile(null); setPreview(null)
    setResult(null); setErrorMsg(null)
    setStatus('idle'); setStage('')
  }

  const analyze = async () => {
    if (!file) return
    setStatus('analyzing'); setErrorMsg(null)
    try {
      setStage('Preprocessing image…')
      await new Promise(r => setTimeout(r, 300))
      setStage('Running deep learning model…')
      const fd = new FormData()
      fd.append('image', file)
      const res = await api.predict(fd)
      setStage('Assessing severity…')
      await new Promise(r => setTimeout(r, 200))
      setResult(res)
      setStatus('done')
    } catch (err) {
      setErrorMsg(err.message || 'Inference failed')
      setStatus('error')
    }
  }

  const downloadAnnotated = () => {
    if (!result?.images?.annotated) return
    const a = document.createElement('a')
    a.href = result.images.annotated
    a.download = `road_analysis_${file?.name || 'result.jpg'}`
    a.click()
  }

  const sv  = result?.severity    || {}
  const det = result?.detections  || []
  const sevStyle = SEV_STYLES[sv.severity_category] || SEV_STYLES.Low

  return (
    <div className="space-y-6 animate-fade-in">
      <PageHeader
        title="Analyze Road Image"
        subtitle="Upload a road photograph for AI-powered damage detection and severity assessment"
        icon={Search}
        action={result && (
          <Button onClick={reset} variant="secondary" size="sm" icon={<RotateCcw size={14} />}>
            New Analysis
          </Button>
        )}
      />

      {/* ── Error alert ── */}
      {errorMsg && (
        <Alert type="error" onClose={() => setErrorMsg(null)}>
          {errorMsg}
        </Alert>
      )}

      {/* ── Upload panel (hidden after result) ── */}
      {!result && (
        <Card className="p-5">
          <SectionTitle>Upload Image</SectionTitle>

          {!file ? (
            <DropZone onFile={handleFile} />
          ) : (
            <div className="space-y-4">
              <div className="flex items-start gap-4">
                <div className="relative flex-shrink-0">
                  <img src={preview} alt="Preview" className="w-24 h-24 object-cover rounded-xl border border-brand-border" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-semibold text-brand-dark truncate">{file.name}</p>
                  <p className="text-xs text-brand-muted mt-1">{(file.size / 1024).toFixed(0)} KB · {file.type || 'image'}</p>
                  <button onClick={reset} className="mt-2 text-xs text-red-500 hover:text-red-700 flex items-center gap-1 transition-colors">
                    <X size={12} /> Remove
                  </button>
                </div>
              </div>

              {status === 'analyzing' ? (
                <div className="bg-sky-50 border border-sky-200 rounded-xl p-4 space-y-3">
                  <div className="flex items-center gap-2">
                    <Spinner size={4} />
                    <p className="text-sm font-medium text-sky-700">{stage}</p>
                  </div>
                  <ProgressBar value={stage.includes('deep') ? 50 : stage.includes('Assessing') ? 85 : 15} />
                  <p className="text-xs text-sky-600">This may take 10–30 seconds on CPU…</p>
                </div>
              ) : (
                <Button
                  onClick={analyze}
                  variant="primary"
                  size="lg"
                  className="w-full"
                  icon={<Search size={16} />}
                >
                  Analyze Road Damage
                </Button>
              )}
            </div>
          )}
        </Card>
      )}

      {/* ── Results ── */}
      {result && (
        <div className="space-y-5">
          {/* Success banner */}
          <Alert type="success">
            Analysis complete in {(result.inference_ms / 1000).toFixed(1)}s
            {result.damage_detected ? ` · ${det.length} damage region${det.length !== 1 ? 's' : ''} detected` : ' · No damage detected'}
          </Alert>

          {/* Image + severity side by side */}
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-5">
            {/* Image viewer */}
            <Card className="lg:col-span-3 p-5">
              <ImageViewer
                original={result.images?.original}
                annotated={result.images?.annotated}
                label="Result"
              />
              <div className="flex gap-2 mt-3">
                {result.images?.annotated && (
                  <Button onClick={downloadAnnotated} variant="secondary" size="sm" icon={<Download size={14} />}>
                    Download Annotated
                  </Button>
                )}
              </div>
            </Card>

            {/* Severity panel */}
            <div className="lg:col-span-2 space-y-4">
              {/* Severity banner */}
              <div className={`${sevStyle.bg} border ${sevStyle.border} rounded-2xl p-5`}>
                <div className="flex items-center justify-between mb-3">
                  <p className={`text-sm font-bold ${sevStyle.text}`}>Severity Assessment</p>
                  <SeverityBadge category={sv.severity_category || 'Low'} />
                </div>
                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className={sevStyle.text + ' font-medium'}>Severity Score</span>
                      <span className={`font-bold ${sevStyle.text}`}>{sv.severity_score?.toFixed(1) ?? 0} / 100</span>
                    </div>
                    <ProgressBar value={sv.severity_score ?? 0} color={sevStyle.bar} />
                  </div>
                </div>
              </div>

              {/* Metric cards */}
              <div className="grid grid-cols-2 gap-3">
                {[
                  { label: 'Repair Priority', value: `${sv.repair_priority ?? 1}/10` },
                  { label: 'Regions Found',   value: sv.region_count ?? 0 },
                  { label: 'Affected Area',   value: `${((sv.normalized_area_ratio ?? 0) * 100).toFixed(2)}%` },
                  { label: 'Inference',       value: `${(result.inference_ms / 1000).toFixed(1)}s` },
                ].map(m => (
                  <div key={m.label} className="bg-white border border-brand-border rounded-xl p-3 text-center">
                    <p className="text-xs text-brand-muted font-medium">{m.label}</p>
                    <p className="text-lg font-bold text-brand-dark mt-0.5">{m.value}</p>
                  </div>
                ))}
              </div>

              {/* Overall condition */}
              <Card className="p-4 bg-sky-50 border-sky-200">
                <p className="text-xs font-semibold text-brand-muted uppercase tracking-wide mb-1">Overall Condition</p>
                <p className="text-sm font-bold text-brand-dark">{result.overall_condition || '—'}</p>
              </Card>

              {/* Max severity class */}
              {sv.max_severity_class && (
                <Card className="p-4">
                  <p className="text-xs font-semibold text-brand-muted uppercase tracking-wide mb-1">Most Severe Damage</p>
                  <p className="text-sm font-semibold text-brand-dark">{sv.max_severity_class}</p>
                  <p className="text-xs text-brand-muted mt-0.5">Class weight: {sv.max_class_weight ?? '—'}</p>
                </Card>
              )}
            </div>
          </div>

          {/* Detections table */}
          <Card className="p-5">
            <SectionTitle sub={`${det.length} region${det.length !== 1 ? 's' : ''} detected`}>
              Detected Damage Regions
            </SectionTitle>
            {det.length === 0 ? (
              <div className="flex items-center gap-2 text-sm text-emerald-700 bg-emerald-50 rounded-xl px-4 py-3">
                <CheckCircle2 size={16} />
                No damage detected — road surface appears clear
              </div>
            ) : (
              <div className="space-y-2">
                {det.map((d, i) => <DetectionRow key={i} index={i + 1} detection={d} />)}
              </div>
            )}
          </Card>

          {/* Severity formula breakdown */}
          <Card className="p-5">
            <SectionTitle icon={Info}>Severity Score Breakdown</SectionTitle>
            <div className="grid grid-cols-1 sm:grid-cols-4 gap-3">
              {[
                {
                  label: 'Area Component',
                  formula: `50 × ${(sv.normalized_area_ratio ?? 0).toFixed(4)}`,
                  value: (50 * (sv.normalized_area_ratio ?? 0)).toFixed(2),
                  max: 50,
                },
                {
                  label: 'Region Density',
                  formula: `25 × (${sv.region_count ?? 0}/5)`,
                  value: (25 * Math.min(sv.region_count ?? 0, 5) / 5).toFixed(2),
                  max: 25,
                },
                {
                  label: 'Class Weight',
                  formula: `25 × ${sv.max_class_weight ?? 0}`,
                  value: (25 * (sv.max_class_weight ?? 0)).toFixed(2),
                  max: 25,
                },
                {
                  label: 'Total Score',
                  formula: `min(100, sum)`,
                  value: sv.severity_score?.toFixed(2) ?? '0',
                  max: 100,
                  highlight: true,
                },
              ].map(s => (
                <div key={s.label} className={`p-3 rounded-xl border ${s.highlight ? 'bg-sky-50 border-sky-200' : 'bg-brand-bg border-brand-border'}`}>
                  <p className="text-xs text-brand-muted font-medium">{s.label}</p>
                  <p className="text-xs text-brand-muted/70 font-mono mt-0.5">{s.formula}</p>
                  <p className={`text-lg font-bold mt-1 ${s.highlight ? 'text-sky-700' : 'text-brand-dark'}`}>
                    {s.value} <span className="text-xs font-normal text-brand-muted">/ {s.max}</span>
                  </p>
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}
    </div>
  )
}
