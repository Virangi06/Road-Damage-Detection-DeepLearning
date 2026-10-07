import { useState } from 'react'
import { Card, SectionTitle, PageHeader, Badge, ProgressBar } from '../components/ui'
import { Info, ChevronDown, ChevronUp, CheckCircle2, Layers } from 'lucide-react'

// ── Architecture card ─────────────────────────────────────────────────────────
function ArchCard({ title, subtitle, badge, rows, highlight = false, accentColor = 'sky' }) {
  const accents = {
    sky:    { border: 'border-sky-200',   bg: 'bg-sky-50',    text: 'text-sky-700',    dot: 'bg-sky-400' },
    teal:   { border: 'border-teal-200',  bg: 'bg-teal-50',   text: 'text-teal-700',   dot: 'bg-teal-400' },
    purple: { border: 'border-purple-200',bg: 'bg-purple-50', text: 'text-purple-700', dot: 'bg-purple-400' },
  }
  const a = accents[accentColor]
  return (
    <Card className={`p-5 flex flex-col gap-4 card-hover ${highlight ? `border-2 ${a.border}` : ''}`}>
      <div className="flex items-start justify-between gap-2">
        <div>
          <div className="flex items-center gap-2">
            <span className={`w-2 h-2 rounded-full ${a.dot}`} />
            <h3 className="text-sm font-bold text-brand-dark">{title}</h3>
            {badge && <Badge variant={accentColor === 'teal' ? 'success' : 'default'} size="sm">{badge}</Badge>}
          </div>
          <p className="text-xs text-brand-muted mt-0.5 ml-4">{subtitle}</p>
        </div>
      </div>
      <div className="space-y-2">
        {rows.map(([k, v]) => (
          <div key={k} className="flex items-start justify-between gap-4 text-xs">
            <span className="text-brand-muted font-medium whitespace-nowrap">{k}</span>
            <span className="text-brand-dark font-semibold text-right">{v}</span>
          </div>
        ))}
      </div>
    </Card>
  )
}

// ── Collapsible Q&A ───────────────────────────────────────────────────────────
function QA({ q, a }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="border-b border-brand-border last:border-none">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between gap-3 py-3.5 text-left group"
      >
        <span className="text-sm font-medium text-brand-dark group-hover:text-sky-700 transition-colors pr-2">{q}</span>
        {open
          ? <ChevronUp size={16} className="flex-shrink-0 text-sky-500" />
          : <ChevronDown size={16} className="flex-shrink-0 text-brand-muted" />}
      </button>
      {open && (
        <div className="pb-4 pl-1">
          <p className="text-sm text-brand-muted leading-relaxed bg-sky-50 rounded-xl px-4 py-3">{a}</p>
        </div>
      )}
    </div>
  )
}

const TABS = ['Architectures', 'Dataset', 'Severity', 'Viva Q&A', 'Tech Stack']

const QA_LIST = [
  ['What is Deep Learning?', 'A subset of ML using multi-layer neural networks to automatically learn hierarchical feature representations from raw data — no manual feature engineering required.'],
  ['Why use Deep Learning for road damage detection?', 'Traditional CV needs hand-crafted features that break across lighting/weather. DL learns robust features from 38,000+ annotated images and generalises across 6 countries.'],
  ['What is a CNN (EfficientNet-B0)?', 'Applies learnable convolutional filters spatially to detect local patterns (edges, textures, crack lines). EfficientNet-B0 uses compound-scaled MBConv blocks for an optimal efficiency-accuracy balance.'],
  ['What is a Vision Transformer (ViT)?', 'Splits the 224×224 image into 196 patches of 16×16 px, embeds them, adds a CLS token + positional embeddings, then applies multi-head self-attention so every patch attends to every other patch simultaneously.'],
  ['Why does ViT outperform CNN on road damage?', 'Crack patterns span large road areas with global spatial dependencies. ViT captures these in one attention pass; CNNs need many stacked layers to achieve a similar receptive field.'],
  ['What is the Hybrid CNN+ViT model?', 'Runs EfficientNet-B0 (1280-dim local features) and ViT-Tiny (192-dim global features) in parallel on the same 224×224 crop, concatenates to 1472-dim, then classifies via a BatchNorm-Dropout-Linear fusion head.'],
  ['What is Faster R-CNN?', 'Two-stage detector: Stage 1 (RPN) proposes candidate regions; Stage 2 (ROI head) classifies each proposal and refines box coordinates. ResNet-50 FPN provides multi-scale features.'],
  ['What is the severity score formula?', 'Score = min(100, 50·A_norm + 25·(N/5) + 25·W_max). Three components: damaged area coverage (0–50 pts), region density (0–25 pts), damage class severity weight (0–25 pts). Pothole has max weight 1.0.'],
  ['Why is severity rule-based rather than a neural model?', 'RDD2022 only has bounding box class labels — not civil engineering severity ratings. The formula is a documented proxy using available detection information.'],
  ['What is Grad-CAM?', 'Registers hooks on the final EfficientNet conv layer. After a forward pass, computes gradient of class score w.r.t. feature activations, global-average-pools them as weights, and produces a spatial heatmap showing which regions influenced the prediction.'],
  ['What is CLAHE?', 'Contrast Limited Adaptive Histogram Equalisation. Applied to the L channel in LAB colour space (clipLimit=2.0, tileGridSize=8×8). Enhances low-contrast cracks without oversaturating bright regions.'],
  ['Why does D43/D44 have 0% F1?', 'Three compounding factors: (1) severe class imbalance (4,628 vs 18,201 samples for D00), (2) high intra-class visual diversity (rutting, edge cracks, pavement markings look completely different), (3) only 3 training epochs.'],
  ['What is mAP?', 'Mean Average Precision — averages the area under the Precision-Recall curve across all classes. mAP@50 uses IoU ≥ 0.50 as the True Positive threshold. Standard object detection metric.'],
  ['What is IoU?', 'Intersection over Union = (Area of Overlap) / (Area of Union) between predicted and ground-truth box. Determines TP (IoU ≥ threshold) vs FP (IoU < threshold).'],
  ['What is Transfer Learning?', 'Initialising model weights from ImageNet pretraining (1M+ images) before fine-tuning on RDD2022. The model starts with learned visual features instead of random weights — critical when target data is limited.'],
  ['What is AdamW?', 'Adam with decoupled weight decay. Weight decay is applied directly to weights (not mixed into the gradient), providing better regularisation than standard Adam + L2.'],
  ['How does the inference pipeline work end-to-end?', '1) Faster R-CNN detects boxes → 2) Crop each with 10% padding → 3) Resize + CLAHE + normalise to 224×224 → 4) Hybrid CNN+ViT classifies → 5) SeverityAssessor computes 0–100 score → 6) OpenCV draws annotations → 7) React dashboard shows results.'],
  ['What are the main project limitations?', '1) 0% F1 on D43/D44. 2) Only 3 training epochs. 3) mAP@50 = 0.0 (evaluation script issue, not model failure). 4) Severity is rule-based. 5) BBox area overestimates actual crack area. 6) No GPS/GIS integration.'],
  ['What future improvements would you make?', '1) Class-weighted loss for D43/D44. 2) Instance segmentation (Mask R-CNN) for pixel-accurate area. 3) More training epochs. 4) Real-time video inference. 5) GPS/GIS integration. 6) Civil engineer severity label collection.'],
  ['What is the research question and conclusion?', 'Does Hybrid CNN+ViT outperform standalone models? In the 3-epoch experiment, ViT (92%) slightly outperforms Hybrid (87%). However, Hybrid shows more consistent F1 across D00/D10/D40. With more epochs + class-weighted loss, Hybrid is expected to exceed both baselines.'],
]

export default function ModelInfo() {
  const [tab, setTab] = useState(0)

  return (
    <div className="space-y-5 animate-fade-in">
      <PageHeader
        title="Model & System Information"
        subtitle="Architecture reference, dataset details, severity methodology, and viva preparation"
        icon={Info}
      />

      {/* Tab bar */}
      <div className="flex flex-wrap gap-1 p-1 bg-sky-50 rounded-xl border border-brand-border w-fit">
        {TABS.map((t, i) => (
          <button key={t} onClick={() => setTab(i)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-150
              ${tab === i ? 'bg-white text-sky-700 shadow-sm' : 'text-brand-muted hover:text-sky-600'}`}>
            {t}
          </button>
        ))}
      </div>

      {/* ── Tab 0: Architectures ── */}
      {tab === 0 && (
        <div className="space-y-5">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <ArchCard
              title="1. CNN Baseline"
              subtitle="EfficientNet-B0"
              accentColor="sky"
              rows={[
                ['Backbone',   'EfficientNet-B0'],
                ['Pretrained', 'ImageNet'],
                ['Input',      '224 × 224 × 3'],
                ['Features',   '1280-dim'],
                ['Head',       'Dropout(0.3) → Linear(5)'],
                ['Params',     '4.01 M'],
                ['File size',  '16.2 MB'],
                ['Accuracy',   '89.0%'],
                ['Macro F1',   '0.6330'],
                ['Latency',    '12.4 ms · 80.6 FPS'],
              ]}
            />
            <ArchCard
              title="2. ViT Baseline"
              subtitle="vit_tiny_patch16_224 (timm)"
              badge="⭐ Best"
              accentColor="teal"
              highlight
              rows={[
                ['Backbone',    'vit_tiny_patch16_224'],
                ['Pretrained',  'ImageNet (timm)'],
                ['Input',       '224×224 → 196 patches'],
                ['Patch size',  '16 × 16 px'],
                ['Features',    '192-dim CLS token'],
                ['Params',      '5.72 M'],
                ['File size',   '22.8 MB'],
                ['Accuracy',    '92.0% 🏆'],
                ['Macro F1',    '0.6953 🏆'],
                ['Latency',     '18.7 ms · 53.5 FPS'],
              ]}
            />
            <ArchCard
              title="3. Hybrid CNN+ViT"
              subtitle="Proposed Architecture"
              accentColor="purple"
              rows={[
                ['CNN stream',  'EfficientNet-B0 → 1280-dim'],
                ['ViT stream',  'vit_tiny → 192-dim'],
                ['Fusion',      'Concat → 1472-dim'],
                ['Head',        'BN→Drop(0.3)→FC(512)→BN→Drop(0.2)→FC(5)'],
                ['Params',      '9.73 M'],
                ['File size',   '38.9 MB'],
                ['Accuracy',    '87.0%'],
                ['Macro F1',    '0.6023'],
                ['Latency',     '31.2 ms · 32.1 FPS'],
              ]}
            />
          </div>

          {/* Faster R-CNN */}
          <Card className="p-5">
            <SectionTitle sub="Phase 2 — Object Detection Baseline">Faster R-CNN ResNet-50 FPN</SectionTitle>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-2">
              {[
                { k: 'Architecture', v: 'Faster R-CNN + ResNet-50 FPN' },
                { k: 'Detection Head', v: 'FastRCNNPredictor · 6 classes' },
                { k: 'Optimizer', v: 'AdamW · lr=3e-4' },
                { k: 'Best Val Loss', v: '0.2661 (Epoch 2)' },
              ].map(s => (
                <div key={s.k} className="bg-brand-bg rounded-xl p-3">
                  <p className="text-xs text-brand-muted font-medium">{s.k}</p>
                  <p className="text-sm font-semibold text-brand-dark mt-0.5">{s.v}</p>
                </div>
              ))}
            </div>
          </Card>

          {/* Training config */}
          <Card className="p-5">
            <SectionTitle sub="Applied to all three classifiers">Training Configuration</SectionTitle>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-2">
              {[
                ['Loss', 'CrossEntropyLoss'], ['Optimizer', 'AdamW · lr=3e-4'],
                ['Scheduler', 'CosineAnnealingLR'], ['Epochs', '3'],
                ['Batch Size', '32'], ['Input', '224 × 224 px'],
                ['Augmentation', 'H-flip + ColorJitter ±0.2'],
                ['Preprocessing', 'CLAHE (LAB) + 10% padding'],
              ].map(([k, v]) => (
                <div key={k} className="bg-brand-bg rounded-xl p-3">
                  <p className="text-xs text-brand-muted font-medium">{k}</p>
                  <p className="text-sm font-semibold text-brand-dark mt-0.5">{v}</p>
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}

      {/* ── Tab 1: Dataset ── */}
      {tab === 1 && (
        <div className="space-y-5">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            {[
              { k: 'Dataset',        v: 'RDD2022' },
              { k: 'Total Images',   v: '38,385' },
              { k: 'Annotations',    v: '65,712 boxes' },
              { k: 'Countries',      v: '6 nations' },
            ].map(s => (
              <Card key={s.k} className="p-4 text-center card-hover">
                <p className="text-xs text-brand-muted font-medium">{s.k}</p>
                <p className="text-lg font-bold text-brand-dark mt-1">{s.v}</p>
              </Card>
            ))}
          </div>

          {/* Class table */}
          <Card className="p-5 overflow-x-auto">
            <SectionTitle>Damage Class Taxonomy</SectionTitle>
            <table className="w-full text-sm mt-2">
              <thead>
                <tr className="bg-sky-50">
                  {['ID','Code','Name','Train Count','Severity Weight'].map(h => (
                    <th key={h} className="px-4 py-2.5 text-left text-xs font-semibold text-brand-muted uppercase">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {[
                  [0,'D00','Longitudinal Crack','18,201',0.4],
                  [1,'D10','Transverse Crack',   '8,386', 0.5],
                  [2,'D20','Alligator Crack',    '7,527', 0.8],
                  [3,'D40','Pothole',            '7,554', 1.0],
                  [4,'D43/D44','Other Damage',   '4,628', 0.6],
                ].map(([id,code,name,count,w]) => (
                  <tr key={id} className="border-t border-brand-border hover:bg-sky-50/40 transition-colors">
                    <td className="px-4 py-3 text-brand-muted text-xs">{id}</td>
                    <td className="px-4 py-3 font-mono text-xs font-semibold text-sky-700">{code}</td>
                    <td className="px-4 py-3 font-medium text-brand-dark">{name}</td>
                    <td className="px-4 py-3 text-brand-muted">{count}</td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <ProgressBar value={w * 100} className="w-20" color={w === 1.0 ? 'bg-red-400' : w >= 0.8 ? 'bg-amber-400' : 'bg-sky-400'} />
                        <span className="text-xs font-semibold text-brand-dark">{w}</span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </div>
      )}

      {/* ── Tab 2: Severity ── */}
      {tab === 2 && (
        <div className="space-y-5">
          <div className="bg-amber-50 border border-amber-200 rounded-2xl p-4 text-sm text-amber-800">
            <strong>Important:</strong> Severity assessment is <strong>rule-based</strong> — not a trained neural network.
            It is a quantitative proxy for road condition, not validated against civil engineering standards.
          </div>

          <Card className="p-5">
            <SectionTitle>Severity Score Formula</SectionTitle>
            <div className="bg-sky-50 rounded-xl p-4 font-mono text-sm text-sky-800 border border-sky-200 mt-2">
              Score = min(100, <span className="font-bold">50 × A_norm</span> + <span className="font-bold">25 × (N / 5)</span> + <span className="font-bold">25 × W_max</span>)
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mt-4">
              {[
                { comp: 'Area Component', sym: 'A_norm', desc: 'Total damage bbox area ÷ image area', max: '50 pts' },
                { comp: 'Region Density', sym: 'N_regions', desc: 'Count of damage regions, capped at 5', max: '25 pts' },
                { comp: 'Class Weight',   sym: 'W_class_max', desc: 'Max damage class severity weight', max: '25 pts' },
              ].map(s => (
                <div key={s.comp} className="bg-brand-bg rounded-xl p-3 border border-brand-border">
                  <p className="text-xs font-bold text-brand-dark">{s.comp}</p>
                  <p className="text-xs font-mono text-sky-600 mt-0.5">{s.sym}</p>
                  <p className="text-xs text-brand-muted mt-1">{s.desc}</p>
                  <Badge variant="default" size="sm" className="mt-2">Max: {s.max}</Badge>
                </div>
              ))}
            </div>
          </Card>

          {/* Severity categories */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {[
              { cat:'Low',    range:'0–29.9',  action:'Monitor regularly',       bg:'bg-emerald-50', border:'border-emerald-200', text:'text-emerald-700' },
              { cat:'Medium', range:'30–64.9', action:'Schedule maintenance',     bg:'bg-amber-50',   border:'border-amber-200',   text:'text-amber-700' },
              { cat:'High',   range:'65–100',  action:'Immediate repair required',bg:'bg-red-50',     border:'border-red-200',     text:'text-red-700' },
            ].map(s => (
              <div key={s.cat} className={`${s.bg} border ${s.border} rounded-2xl p-4`}>
                <p className={`text-sm font-bold ${s.text}`}>{s.cat} Severity</p>
                <p className={`text-xs ${s.text} opacity-80 mt-0.5`}>Score: {s.range}</p>
                <p className="text-xs text-brand-muted mt-2">{s.action}</p>
              </div>
            ))}
          </div>

          {/* Worked example */}
          <Card className="p-5">
            <SectionTitle>Worked Example</SectionTitle>
            <div className="bg-brand-bg rounded-xl p-4 text-xs font-mono text-brand-dark space-y-1 border border-brand-border mt-2">
              <p>Image: 640×480 px · Pothole [50,100,200,250] · Alligator [300,200,500,380]</p>
              <p className="text-brand-muted">Pothole area:    150×150 = 22,500 px²</p>
              <p className="text-brand-muted">Alligator area:  200×180 = 36,000 px²</p>
              <p className="text-brand-muted">Total damage:    58,500 px² · Image area: 307,200 px²</p>
              <p className="mt-2">A_norm = 58,500 / 307,200 = <span className="text-sky-700 font-bold">0.1904</span></p>
              <p>Area term   = 50 × 0.1904 = <span className="text-sky-700 font-bold">9.52</span></p>
              <p>Region term = 25 × (2/5)  = <span className="text-sky-700 font-bold">10.00</span></p>
              <p>Weight term = 25 × 1.0    = <span className="text-sky-700 font-bold">25.00</span> (Pothole = max)</p>
              <p className="mt-2 font-bold text-amber-700">Score = min(100, 44.52) = 44.52 → MEDIUM</p>
              <p className="font-bold text-amber-700">Priority = Clamp(4 + 2, 1, 10) = 6/10</p>
            </div>
          </Card>
        </div>
      )}

      {/* ── Tab 3: Viva Q&A ── */}
      {tab === 3 && (
        <div className="space-y-4">
          <div className="bg-sky-50 border border-sky-200 rounded-2xl p-4 text-sm text-sky-800">
            <strong>30-Second Explanation:</strong>{' '}
            "Our project automates road damage inspection using deep learning on the RDD2022 dataset
            (38,000+ road images from 6 countries). We detect damage with Faster R-CNN, classify it into
            five types using a Hybrid CNN+ViT model, and estimate severity on a 0–100 scale.
            The ViT baseline achieved 92% accuracy. All results are accessible through this web dashboard."
          </div>

          <Card className="p-5 divide-y divide-brand-border">
            {QA_LIST.map(([q, a]) => <QA key={q} q={q} a={a} />)}
          </Card>
        </div>
      )}

      {/* ── Tab 4: Tech Stack ── */}
      {tab === 4 && (
        <div className="space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {[
              { name: 'Python 3.11',      role: 'Core language',                 tag: 'Backend' },
              { name: 'PyTorch 2.0+',     role: 'DL framework · training & inference', tag: 'Backend' },
              { name: 'TorchVision',      role: 'Faster R-CNN · transforms',     tag: 'Backend' },
              { name: 'timm 0.9+',        role: 'ViT model library',             tag: 'Backend' },
              { name: 'OpenCV 4.8+',      role: 'CLAHE · image annotation',      tag: 'Backend' },
              { name: 'Flask',            role: 'REST API server',               tag: 'API' },
              { name: 'Streamlit 1.64+',  role: 'Original Streamlit dashboard',  tag: 'UI' },
              { name: 'React 18',         role: 'Frontend framework',            tag: 'Frontend' },
              { name: 'Tailwind CSS 3',   role: 'Utility-first styling',         tag: 'Frontend' },
              { name: 'Recharts',         role: 'Interactive charts',            tag: 'Frontend' },
              { name: 'Lucide React',     role: 'Icon library',                  tag: 'Frontend' },
              { name: 'Vite 5',           role: 'Build tool + dev proxy',        tag: 'Frontend' },
              { name: 'Scikit-learn',     role: 'Metrics · P/R/F1/confusion',    tag: 'Backend' },
              { name: 'Plotly',           role: 'Streamlit charts',              tag: 'UI' },
              { name: 'NumPy / Pandas',   role: 'Data manipulation',             tag: 'Backend' },
            ].map(s => (
              <div key={s.name} className="flex items-center gap-3 p-3 rounded-xl border border-brand-border bg-white hover:bg-sky-50/50 transition-colors">
                <div className="w-8 h-8 rounded-lg bg-sky-100 flex items-center justify-center flex-shrink-0">
                  <Layers size={16} className="text-sky-600" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-semibold text-brand-dark truncate">{s.name}</p>
                  <p className="text-xs text-brand-muted truncate">{s.role}</p>
                </div>
                <Badge variant={s.tag === 'Frontend' ? 'default' : s.tag === 'API' ? 'success' : 'purple'} size="sm">
                  {s.tag}
                </Badge>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
