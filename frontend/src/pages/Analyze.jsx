/**
 * Analyze.jsx — Road Image Analysis Page
 *
 * Full workflow:
 *   Upload → Preview → Analyze → Progress → Results
 *
 * Shows detailed error messages when the API is unavailable or
 * returns an error, so the user knows exactly what to do.
 */

import { useState, useRef, useCallback, useEffect } from 'react'
import { api } from '../utils/api'
import {
  Card, SectionTitle, PageHeader, SeverityBadge,
  Badge, Alert, Spinner, ProgressBar, Button,
} from '../components/ui'
import {
  Upload, X, Search, Download, RotateCcw,
  AlertTriangle, CheckCircle2, ZoomIn, ZoomOut,
  Info, Clock, Maximize2, Terminal,
} from 'lucide-react'

// ── Severity colour map ────────────────────────────────────────────────────────
const SEV = {
  Low:    { bg: 'bg-emerald-50', border: 'border-emerald-200', text: 'text-emerald-700', bar: 'bg-emerald-400', icon: '🟢' },
  Medium: { bg: 'bg-amber-50',   border: 'border-amber-200',   text: 'text-amber-700',   bar: 'bg-amber-400',   icon: '🟡' },
  High:   { bg: 'bg-red-50',     border: 'border-red-200',     text: 'text-red-700',      bar: 'bg-red-400',     icon: '🔴' },
}

// ── Stages shown during analysis ───────────────────────────────────────────────
const STAGES = [
  { pct: 10, msg: 'Validating image…' },
  { pct: 25, msg: 'Sending to AI server…' },
  { pct: 40, msg: 'Running Faster R-CNN detection…' },
  { pct: 65, msg: 'Classifying damage regions (Hybrid CNN+ViT)…' },
  { pct: 85, msg: 'Computing severity score…' },
  { pct: 95, msg: 'Generating annotated image…' },
]

// ── API Status check component ─────────────────────────────────────────────────
function ApiStatus({ onReady }) {
  const [status, setStatus] = useState('checking')  // checking | online | offline
  const [detail, setDetail] = useState('')

  useEffect(() => {
    api.health()
      .then(d => {
        if (d.models_loaded) {
          setStatus('online')
          onReady(true)
        } else {
          setStatus('loading')
          setDetail(d.model_error || 'Models are loading, please wait…')
        }
      })
      .catch(err => {
        setStatus('offline')
        setDetail(err.message)
        onReady(false)
      })
  }, [])

  if (status === 'online') return null   // hide when all good

  const styles = {
    checking: { type: 'info',    text: 'Checking API status…' },
    loading:  { type: 'warning', text: 'API online — models still loading, please wait and retry.' },
    offline:  { type: 'error',   text: 'API server is not running.' },
  }
  const s = styles[status] || styles.checking

  return (
    <Alert type={s.type}>
      <div className="space-y-1">
        <p className="font-semibold">{s.text}</p>
        {status === 'offline' && (
          <>
            <p className="text-xs">Start the Flask API server by running in a new terminal:</p>
            <div className="flex items-center gap-2 bg-white/60 rounded-lg px-3 py-1.5 font-mono text-xs mt-1 border border-current/20">
              <Terminal size={12} />
              <span className="select-all">python api.py</span>
            </div>
            {detail && <p className="text-xs opacity-80 mt-1">Details: {detail}</p>}
          </>
        )}
        {status === 'loading' && (
          <p className="text-xs">{detail}</p>
        )}
      </div>
    </Alert>
  )
}

// ── Drop Zone ─────────────────────────────────────────────────────────────────
function DropZone({ onFile }) {
  const inputRef = useRef()
  const [drag, setDrag] = useState(false)

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDrag(false)
    const file = e.dataTransfer.files?.[0]
    if (file) onFile(file)
  }, [onFile])

  return (
    <div
      onDragOver={e => { e.preventDefault(); setDrag(true) }}
      onDragLeave={() => setDrag(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
      className={`
        flex flex-col items-center justify-center gap-3 p-10
        rounded-2xl border-2 border-dashed cursor-pointer
        transition-all duration-200 select-none
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
      <p className="text-xs text-brand-muted/70">Works best on road surface photos from cars or drones</p>
      <input
        ref={inputRef}
        type="file"
        accept=".jpg,.jpeg,.png,image/jpeg,image/png"
        onChange={e => { const f = e.target.files?.[0]; if (f) onFile(f) }}
        className="hidden"
      />
    </div>
  )
}

// ── Progress indicator ────────────────────────────────────────────────────────
function AnalysisProgress({ stage, elapsed }) {
  return (
    <div className="bg-sky-50 border border-sky-200 rounded-2xl p-5 space-y-3">
      <div className="flex items-center gap-3">
        <Spinner size={5} color="text-sky-500" />
        <div>
          <p className="text-sm font-semibold text-sky-800">{stage.msg}</p>
          <p className="text-xs text-sky-600 mt-0.5">
            Elapsed: {elapsed}s · CPU inference typically takes 10–60 seconds
          </p>
        </div>
      </div>
      <ProgressBar value={stage.pct} color="bg-sky-400" />
      <p className="text-xs text-sky-600">
        Please keep this tab open. Do not refresh the page.
      </p>
    </div>
  )
}

// ── Image viewer ───────────────────────────────────────────────────────────────
function ImageViewer({ original, annotated }) {
  const [showAnnotated, setShowAnnotated] = useState(true)
  const [expanded, setExpanded] = useState(false)

  const current = (showAnnotated && annotated) ? annotated : original

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <p className="text-xs font-semibold text-brand-muted uppercase tracking-wide">
          {showAnnotated && annotated ? 'AI Detection Result' : 'Original Image'}
        </p>
        <div className="flex items-center gap-2">
          {annotated && (
            <div className="flex rounded-lg border border-brand-border overflow-hidden text-xs">
              <button
                onClick={() => setShowAnnotated(false)}
                className={`px-2.5 py-1 font-medium transition-colors ${!showAnnotated ? 'bg-sky-100 text-sky-700' : 'bg-white text-brand-muted hover:bg-sky-50'}`}
              >
                Original
              </button>
              <button
                onClick={() => setShowAnnotated(true)}
                className={`px-2.5 py-1 font-medium transition-colors ${showAnnotated ? 'bg-sky-100 text-sky-700' : 'bg-white text-brand-muted hover:bg-sky-50'}`}
              >
                Detected
              </button>
            </div>
          )}
          <button
            onClick={() => setExpanded(x => !x)}
            className="p-1.5 rounded-lg hover:bg-sky-50 text-brand-muted hover:text-sky-600 transition-colors"
            title={expanded ? 'Collapse' : 'Expand'}
          >
            {expanded ? <ZoomOut size={14} /> : <ZoomIn size={14} />}
          </button>
        </div>
      </div>

      <div className={`relative overflow-hidden rounded-xl border border-brand-border bg-gray-50 transition-all duration-300 ${expanded ? 'max-h-[700px]' : 'max-h-80'}`}>
        {current ? (
          <img
            src={current}
            alt={showAnnotated ? 'Annotated road' : 'Original road'}
            className="w-full object-contain"
          />
        ) : (
          <div className="h-48 flex items-center justify-center text-brand-muted text-sm">
            Image not available
          </div>
        )}
        {showAnnotated && annotated && (
          <div className="absolute top-2 left-2">
            <span className="text-xs bg-black/60 text-white px-2 py-0.5 rounded-full font-medium backdrop-blur-sm">
              🔍 AI Detection
            </span>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Detection row ──────────────────────────────────────────────────────────────
function DetRow({ idx, d }) {
  const conf = typeof d.cls_confidence_pct === 'number'
    ? d.cls_confidence_pct
    : (d.cls_confidence ?? 0) * 100
  const bbox = (d.bbox || [0, 0, 0, 0]).map(Math.round)
  const [x1, y1, x2, y2] = bbox
  const confColor = conf >= 70 ? 'text-emerald-700' : conf >= 40 ? 'text-amber-700' : 'text-red-600'

  return (
    <div className="flex items-center gap-3 p-3 rounded-xl bg-brand-bg hover:bg-sky-50/60 transition-colors">
      <div className="w-7 h-7 rounded-full bg-sky-100 text-sky-700 text-xs font-bold flex items-center justify-center flex-shrink-0">
        {idx}
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-semibold text-brand-dark">
          {d.class_name || d.class_full || 'Unknown damage'}
        </p>
        <p className="text-xs text-brand-muted">
          Box: ({x1},{y1}) → ({x2},{y2}) &nbsp;·&nbsp; {x2-x1}×{y2-y1} px
        </p>
      </div>
      <div className={`text-right flex-shrink-0 font-bold text-sm ${confColor}`}>
        {conf.toFixed(1)}%
      </div>
    </div>
  )
}

// ── Main page ──────────────────────────────────────────────────────────────────
export default function Analyze() {
  const [apiReady,   setApiReady]   = useState(false)
  const [file,       setFile]       = useState(null)
  const [preview,    setPreview]    = useState(null)
  const [status,     setStatus]     = useState('idle')   // idle|analyzing|done|error
  const [result,     setResult]     = useState(null)
  const [errorMsg,   setErrorMsg]   = useState(null)
  const [stageIdx,   setStageIdx]   = useState(0)
  const [elapsed,    setElapsed]    = useState(0)
  const timerRef = useRef(null)

  // Stage cycling during analysis
  useEffect(() => {
    if (status === 'analyzing') {
      let i = 0
      let secs = 0
      const stageTimer = setInterval(() => {
        secs++
        setElapsed(secs)
        // advance stage every ~8 seconds
        if (secs % 8 === 0 && i < STAGES.length - 1) {
          i++
          setStageIdx(i)
        }
      }, 1000)
      timerRef.current = stageTimer
      return () => clearInterval(stageTimer)
    } else {
      clearInterval(timerRef.current)
    }
  }, [status])

  const handleFile = useCallback((f) => {
    const ext = (f.name || '').split('.').pop().toLowerCase()
    if (!['jpg','jpeg','png'].includes(ext)) {
      setErrorMsg('Unsupported file type. Please upload JPG or PNG.')
      return
    }
    if (f.size > 200 * 1024 * 1024) {
      setErrorMsg('File too large. Maximum 200 MB.')
      return
    }
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setResult(null)
    setErrorMsg(null)
    setStatus('idle')
    setStageIdx(0)
    setElapsed(0)
  }, [])

  const reset = () => {
    setFile(null); setPreview(null)
    setResult(null); setErrorMsg(null)
    setStatus('idle'); setStageIdx(0); setElapsed(0)
    clearInterval(timerRef.current)
  }

  const analyze = async () => {
    if (!file || status === 'analyzing') return

    setStatus('analyzing')
    setErrorMsg(null)
    setStageIdx(0)
    setElapsed(0)

    try {
      const fd = new FormData()
      fd.append('image', file)
      const res = await api.predict(fd)

      clearInterval(timerRef.current)

      if (!res.success) {
        setErrorMsg(res.error || 'Inference returned an error.')
        setStatus('error')
        return
      }

      setResult(res)
      setStatus('done')
    } catch (err) {
      clearInterval(timerRef.current)
      setErrorMsg(err.message || 'Analysis failed.')
      setStatus('error')
    }
  }

  const downloadAnnotated = () => {
    const url = result?.images?.annotated
    if (!url) return
    const a = document.createElement('a')
    a.href  = url
    a.download = `road_analysis_${file?.name || 'result.jpg'}`
    a.click()
  }

  // Derived result values
  const sv      = result?.severity || {}
  const det     = result?.detections || []
  const sevCat  = sv.severity_category || 'Low'
  const sevSt   = SEV[sevCat] || SEV.Low
  const areaRat = sv.normalized_area_ratio ?? 0
  const areaPts = (50 * areaRat).toFixed(2)
  const regPts  = (25 * Math.min(sv.region_count ?? 0, 5) / 5).toFixed(2)
  const wPts    = (25 * (sv.max_class_weight ?? 0)).toFixed(2)

  return (
    <div className="space-y-5 animate-fade-in max-w-5xl">
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

      {/* API status — always shown at top */}
      <ApiStatus onReady={setApiReady} />

      {/* Error alert */}
      {errorMsg && (
        <Alert type="error" onClose={() => setErrorMsg(null)}>
          <div className="space-y-1">
            <p className="font-semibold">{errorMsg}</p>
            {errorMsg.includes('5000') && (
              <div className="flex items-center gap-2 bg-white/60 rounded-lg px-3 py-1.5 font-mono text-xs mt-1 border border-red-200">
                <Terminal size={12} />
                <span className="select-all">python api.py</span>
              </div>
            )}
          </div>
        </Alert>
      )}

      {/* ── Upload + settings panel ── */}
      {!result && (
        <Card className="p-5">
          <SectionTitle>Upload Road Image</SectionTitle>

          {!file ? (
            <DropZone onFile={handleFile} />
          ) : (
            <div className="space-y-4">
              {/* File preview row */}
              <div className="flex items-start gap-4 p-4 bg-brand-bg rounded-xl border border-brand-border">
                <img
                  src={preview}
                  alt="Preview"
                  className="w-20 h-20 object-cover rounded-xl border border-brand-border flex-shrink-0"
                />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-semibold text-brand-dark truncate">{file.name}</p>
                  <p className="text-xs text-brand-muted mt-1">
                    {(file.size / 1024).toFixed(0)} KB &nbsp;·&nbsp; {file.type || 'image'}
                  </p>
                  <button
                    onClick={reset}
                    className="mt-2 text-xs text-red-500 hover:text-red-700 flex items-center gap-1 transition-colors"
                  >
                    <X size={12} /> Remove
                  </button>
                </div>
                <Badge variant={apiReady ? 'success' : 'warning'} size="sm">
                  {apiReady ? 'API Ready' : 'API Offline'}
                </Badge>
              </div>

              {/* Progress OR analyze button */}
              {status === 'analyzing' ? (
                <AnalysisProgress stage={STAGES[stageIdx]} elapsed={elapsed} />
              ) : (
                <button
                  onClick={analyze}
                  disabled={!apiReady}
                  className={`
                    w-full flex items-center justify-center gap-2.5 py-3 px-6
                    rounded-xl font-semibold text-sm transition-all duration-200
                    ${apiReady
                      ? 'bg-sky-500 hover:bg-sky-600 active:bg-sky-700 text-white shadow-sm hover:shadow-md cursor-pointer'
                      : 'bg-gray-200 text-gray-400 cursor-not-allowed'}
                  `}
                >
                  <Search size={16} />
                  {apiReady ? 'Analyze Road Damage' : 'Waiting for API…'}
                </button>
              )}

              {!apiReady && status !== 'analyzing' && (
                <p className="text-xs text-center text-brand-muted">
                  Start the API server first: <code className="bg-sky-50 px-1.5 py-0.5 rounded text-sky-700 font-mono">python api.py</code>
                </p>
              )}
            </div>
          )}
        </Card>
      )}

      {/* ── Results ── */}
      {result && status === 'done' && (
        <div className="space-y-5">
          {/* Success banner */}
          <Alert type="success">
            <span className="font-semibold">Analysis complete</span>
            {' '}in {(result.inference_ms / 1000).toFixed(1)}s
            {det.length > 0
              ? ` — ${det.length} damage region${det.length !== 1 ? 's' : ''} detected`
              : ' — road surface appears clear'}
          </Alert>

          {/* Image + severity panel */}
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-5">

            {/* Image viewer */}
            <Card className="lg:col-span-3 p-5">
              <ImageViewer
                original={result.images?.original}
                annotated={result.images?.annotated}
              />
              <div className="flex flex-wrap gap-2 mt-3">
                {result.images?.annotated && (
                  <Button
                    onClick={downloadAnnotated}
                    variant="secondary" size="sm"
                    icon={<Download size={14} />}
                  >
                    Download Annotated
                  </Button>
                )}
                <Button onClick={reset} variant="ghost" size="sm" icon={<RotateCcw size={14} />}>
                  Analyze Another
                </Button>
              </div>
            </Card>

            {/* Severity panel */}
            <div className="lg:col-span-2 space-y-3">

              {/* Severity banner */}
              <div className={`${sevSt.bg} border ${sevSt.border} rounded-2xl p-5`}>
                <div className="flex items-center justify-between mb-3">
                  <p className={`font-bold ${sevSt.text}`}>
                    {sevSt.icon} {sevCat} Severity
                  </p>
                  <SeverityBadge category={sevCat} />
                </div>
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className={sevSt.text + ' font-medium'}>Score</span>
                    <span className={`font-bold ${sevSt.text}`}>
                      {(sv.severity_score ?? 0).toFixed(1)} / 100
                    </span>
                  </div>
                  <ProgressBar value={sv.severity_score ?? 0} color={sevSt.bar} />
                </div>
              </div>

              {/* 4 metric chips */}
              <div className="grid grid-cols-2 gap-2">
                {[
                  { label: 'Repair Priority', value: `${sv.repair_priority ?? 1}/10` },
                  { label: 'Regions Found',   value: sv.region_count ?? 0 },
                  { label: 'Affected Area',   value: `${(areaRat * 100).toFixed(2)}%` },
                  { label: 'Inference',       value: `${(result.inference_ms / 1000).toFixed(1)}s` },
                ].map(m => (
                  <div key={m.label} className="bg-white border border-brand-border rounded-xl p-3 text-center">
                    <p className="text-xs text-brand-muted font-medium">{m.label}</p>
                    <p className="text-lg font-bold text-brand-dark mt-0.5">{String(m.value)}</p>
                  </div>
                ))}
              </div>

              {/* Overall condition */}
              <div className="bg-sky-50 border border-sky-200 rounded-xl p-3">
                <p className="text-xs font-semibold text-brand-muted uppercase tracking-wide mb-1">
                  Overall Condition
                </p>
                <p className="text-sm font-bold text-brand-dark">
                  {result.overall_condition || '—'}
                </p>
              </div>

              {/* Max severity class */}
              {sv.max_severity_class && sv.max_severity_class !== 'None' && (
                <div className="bg-white border border-brand-border rounded-xl p-3">
                  <p className="text-xs font-semibold text-brand-muted uppercase tracking-wide mb-1">
                    Dominant Damage Class
                  </p>
                  <p className="text-sm font-semibold text-brand-dark">{sv.max_severity_class}</p>
                  <p className="text-xs text-brand-muted mt-0.5">
                    Severity weight: {sv.max_class_weight ?? '—'}
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Detections list */}
          <Card className="p-5">
            <SectionTitle
              sub={`${det.length} region${det.length !== 1 ? 's' : ''} found`}
            >
              Detected Damage Regions
            </SectionTitle>
            {det.length === 0 ? (
              <div className="flex items-center gap-2 text-sm text-emerald-700 bg-emerald-50 rounded-xl px-4 py-3 border border-emerald-200">
                <CheckCircle2 size={16} className="flex-shrink-0" />
                No damage regions detected — road surface appears clear.
              </div>
            ) : (
              <div className="space-y-2 mt-2">
                {det.map((d, i) => <DetRow key={i} idx={i + 1} d={d} />)}
              </div>
            )}
          </Card>

          {/* Severity score formula breakdown */}
          <Card className="p-5">
            <SectionTitle sub="How the score was calculated">
              Severity Score Breakdown
            </SectionTitle>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-2">
              {[
                {
                  label:   'Area Component',
                  formula: `50 × ${areaRat.toFixed(4)}`,
                  value:   areaPts,
                  max:     50,
                },
                {
                  label:   'Region Density',
                  formula: `25 × (${sv.region_count ?? 0}/5)`,
                  value:   regPts,
                  max:     25,
                },
                {
                  label:   'Class Weight',
                  formula: `25 × ${sv.max_class_weight ?? 0}`,
                  value:   wPts,
                  max:     25,
                },
                {
                  label:     'Total Score',
                  formula:   'min(100, sum)',
                  value:     (sv.severity_score ?? 0).toFixed(2),
                  max:       100,
                  highlight: true,
                },
              ].map(s => (
                <div
                  key={s.label}
                  className={`p-3 rounded-xl border ${s.highlight ? 'bg-sky-50 border-sky-200' : 'bg-brand-bg border-brand-border'}`}
                >
                  <p className="text-xs text-brand-muted font-medium">{s.label}</p>
                  <p className="text-xs text-brand-muted/70 font-mono mt-0.5 leading-snug">{s.formula}</p>
                  <p className={`text-lg font-bold mt-1 ${s.highlight ? 'text-sky-700' : 'text-brand-dark'}`}>
                    {s.value}
                    <span className="text-xs font-normal text-brand-muted ml-1">/ {s.max}</span>
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
