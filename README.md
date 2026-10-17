# Time-Series Foundation Models vs. Simple Baselines for ICS Attack Detection

MS Data Science capstone, University of Arizona (College of Information Science).
Author: Tanishk Singh · Advisor: Prof. Rozhin Yasaei (MITHRA Lab) · PhD contact: Afjal Sarower.

## The question

Under honest evaluation and with running cost counted, does a pretrained
time-series foundation model (TSFM) — with or without **Forecast Fallback** —
actually beat PCA, windowed PCA, and a small CNN at detecting attacks on
industrial control systems? And which *kinds* of anomaly does each method catch?

"Honest evaluation" is not decoration. It means, for every method alike:

- **VUS-PR** as the primary metric (range-aware, threshold-independent), with
  range-based F1 and ROC-AUC reported alongside.
- **No point-adjusted F1.** Under PA, a random scorer ranks 26th of 33 in TSB-AD.
- Thresholds fitted on **training / benign-validation data only** — never on test.
  Both Lowe et al. and Uray et al. tuned on test, which inflates their F1.
- Results broken out **by anomaly type**, not pooled into one headline number.

## Why the gap is real

| Prior work | What it showed | What it left open |
|---|---|---|
| Lowe, Fung & Bauer (ICML 2026 wksp) | Zero-shot TSFMs copy attacked input into their forecasts, so error collapses inside long attacks. Forecast Fallback recovers 0.1–0.45 ROC-AUC. | No PCA or CNN baseline; thresholds taken from test. |
| Wen & Chen (arXiv 2608.22968) | A 13.8 KB TCN-AE beat 138 MB MOMENT on C-MAPSS (AUROC 0.957 vs 0.731). | No security data, no attacks. |
| Fung et al. (ESORICS 2022) | Architecture matters little; preprocessing and metric choice matter a lot. | Predates TSFMs. Code reused here. |
| Uray et al. (arXiv 2607.12454) | TimesFM on SWaT collapses inside attacks; embeddings + Isolation Forest reach F1 0.480 vs PCA's 0.833. | One model, one dataset, test-tuned threshold. |
| Liu & Paparrizos — TSB-AD (NeurIPS 2024) | VUS-PR is the most reliable metric; simple methods are strong. | Most TSFMs never tested multivariately. |
| Kravchik & Shabtai (IEEE TDSC) | Windowed PCA F1 0.879 on SWaT, beating the autoencoder. | No TSFMs. "Try PCA first." |
| Sarfraz et al. (ICML 2024) | Plain PCA best on SWaT (0.833) and WADI-127 (0.501). | Deep models mostly learn linear maps. |
| Pinet et al. (MiLeTS @ KDD 2026) | Of 373 real anomaly segments, **none** was purely cross-sensor. Proposes NOISEFLIP / NPROLL injections. | No TSFM, PCA or CNN; synthetic sinusoids only. |

No paper puts a TSFM **with Forecast Fallback** against **PCA and a CNN** on the
same ICS data with **marginal-preserving cross-sensor injections** scored **per
anomaly type**. That combination is the target.

## Scope

1. **Direction 1 — head-to-head** (committed). TSFM ±FF vs PCA, windowed PCA, small CNN. Real attacks plus synthetic cross-sensor injections. Cost measured: latency, memory, energy.
2. **Direction 2 — root cause** (committed). Ablations isolating *which* property explains the ranking: fixed vs adapting notion of "normal", per-sensor vs joint, history window, pretrained vs plant-fitted, linear vs nonlinear structure.
3. **Direction 3 — enhancement** (future work). Better FF thresholding, boundary-spike triggers, multivariate fallback.
4. **Direction 4 — new architectures** (future work). Spiking nets, state-space models.

A **null result is an acceptable outcome** ("PCA is enough unless the anomaly is
type X"). An *unexplainable* result is not — hence Direction 2 runs either way.
PCA must be tuned carefully: published PCA F1 on SWaT spans 0.23 to 0.88 purely
on setup, so an under-tuned baseline would invalidate any TSFM win.

## Data — not in this repo

Raw datasets are **gitignored**: ~750 MB total, four files exceed GitHub's 100 MB
limit, and SWaT/WADI are redistribution-restricted under an iTrust agreement.
Keep them locally at these paths:

```
$$SWAT Dataset A2-Dec 2015/    SWaT_Dataset_Normal_v1.xlsx, SWaT_Dataset_Attack_v0.xlsx,
                               List_of_attacks_Final(Sheet1).csv
WADI.A2_19 Nov 2019/           WADI_14days_new.csv, WADI_attackdataLABLE.csv, table_WADI.pdf
HAI/                           (not yet downloaded — github.com/icsdataset/hai)
```

### Verified dataset traps

Checked directly against the files in this working copy. Each one silently
corrupts results if missed.

**SWaT** — attack set 449,919 rows × 51 sensors @ 1 Hz:

- Labels are `Normal` (395,298), `Attack` (54,584), and **`A ttack` — with a
  space — (37)**. Filtering `label == 'Attack'` silently drops those 37 rows.
  Match on `'ttack' in label`.
- 35 contiguous attack segments. Min 101 s, median 444 s, **none under 60 s** —
  every attack is long-context, precisely the regime where TSFMs are reported
  to fail. Good news for the hypothesis under test.
- **One segment (rows 227,831–263,730) runs 35,900 s (~10 h) and contains 66 %
  of all attack-labeled timesteps.** Any pointwise metric is dominated by this
  single event; report segment-level results beside it.
- `List_of_attacks_Final(Sheet1).csv` documents 41 attacks, 5 marked
  "No Physical Impact" (what Fung et al.'s attack cleaning removes), and names
  the official taxonomy — Single/Multi Stage × Single/Multi Point — which gives
  a per-anomaly-type axis on *real* attacks, not just synthetic ones.
- `openpyxl` needs >10 min on these workbooks; streaming
  `xl/worksheets/sheet1.xml` out of the zip does it in ~5 s. Convert to Parquet once.

**WADI** — 127 sensors (the "WADI-127" of Sarfraz et al.):

- The normal file has a **264,000-row gap**: `Row` jumps 335,999 → 600,000, with
  784,571 rows actually present. **Windows must not span the discontinuity.**
- Labels are **inverted**: `1` = no attack (162,826), `-1` = attack (9,977),
  5.8 % attack rate across 14 contiguous segments (88–1,741 s).
- `Date`/`Time` columns are Excel-mangled (`Time` is mm:ss; CSV uses M/D/Y while
  `table_WADI.pdf` uses D/M/Y). **Align labels by row index, not timestamp:**
  the data is 1 Hz starting `2017-10-09 18:00:00`, after which the documented
  attack times match the label runs with a constant ~3–4 s offset (verified on
  attacks 1, 2 and 15, spanning the whole file).

## Environment

Local GPU is a **GTX 1660 Ti, 6 GB** (Turing, no tensor cores) — this bounds
TSFM batch size and is the main schedule risk. Forecast Fallback is inherently
sequential (it feeds predictions back into the input), so it cannot be batched
away: at horizon 1, SWaT's test set alone is 449,919 forward passes per
configuration. **Decide the stride/subset policy before launching the grid.**

No existing conda env is usable as-is: the CUDA-enabled one lacks pandas and
sklearn, the one with pandas has CPU-only torch. A fresh env is needed.

## Status

Datasets for SWaT and WADI are in hand. HAI is not yet downloaded. Code is at
the scaffolding stage — see the project board / milestones for current progress.

## License

MIT — see [LICENSE](LICENSE).
