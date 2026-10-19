# Time-Series Foundation Models vs. Simple Baselines for ICS Attack Detection

MS Data Science capstone, University of Arizona (College of Information Science).
Author: Tanishk Singh. Advisor: Prof. Rozhin Yasaei (MITHRA Lab). PhD contact: Afjal Sarower.

## About

Time-series foundation models (TSFMs) are pretrained on huge, general-purpose
collections of time series and then applied to a new domain with little or no
fine-tuning. They are an attractive idea for industrial control systems (ICS)
because labeled attack data is scarce and expensive to collect. This project
asks a direct question: under a fair, honest evaluation, with running cost
counted, does a pretrained TSFM, with or without a technique called Forecast
Fallback, actually outperform much simpler methods such as PCA, windowed PCA,
and a small CNN at detecting attacks on ICS data? And just as importantly,
which kinds of anomaly does each method actually catch, and which does it miss?

"Honest evaluation" is a deliberate choice of words here, not a throwaway
phrase. It means, applied equally to every method under test:

- **VUS-PR** as the primary metric (range-aware and threshold-independent),
  with range-based F1 and ROC-AUC reported alongside it.
- **No point-adjusted F1.** Under the point-adjust protocol, a random scorer
  ranks 26th of 33 methods on the TSB-AD benchmark; it rewards noise.
- Detection thresholds fitted only on training or held-out normal data, never
  on the test set. Two of the papers this project builds on tuned thresholds
  on test data, which inflates their reported F1.
- Results broken out by anomaly type, rather than collapsed into one headline
  number that hides where a method actually struggles.

## The gap

| Prior work | What it showed | What it left open |
|---|---|---|
| Lowe, Fung & Bauer (ICML 2026 workshop) | Zero-shot TSFMs copy attacked input straight into their forecasts, so the forecast error collapses during long attacks. Forecast Fallback recovers 0.1 to 0.45 ROC-AUC. | No PCA or CNN baseline; thresholds taken from the test set. |
| Wen & Chen (arXiv 2608.22968) | A 13.8 KB TCN-AE beat a 138 MB MOMENT model on C-MAPSS (AUROC 0.957 vs. 0.731). | No security data, no attacks. |
| Fung et al. (ESORICS 2022) | Architecture matters relatively little; preprocessing and metric choice matter a great deal. | Predates TSFMs. Code from this paper is reused here for the classical baselines. |
| Uray et al. (arXiv 2607.12454) | TimesFM on SWaT collapses during attacks; embeddings plus Isolation Forest reach F1 0.480 against PCA's 0.833. | One model, one dataset, threshold tuned on test. |
| Liu & Paparrizos, TSB-AD (NeurIPS 2024) | VUS-PR is the most reliable anomaly-detection metric studied; simple methods are often strong. | Most TSFMs were never tested on multivariate data. |
| Kravchik & Shabtai (IEEE TDSC) | Windowed PCA reached F1 0.879 on SWaT, beating an autoencoder. | No TSFMs considered. The implicit lesson is "try PCA first." |
| Sarfraz et al. (ICML 2024) | Plain PCA performed best on SWaT (0.833) and WADI-127 (0.501). | Deep models mostly end up learning something close to a linear map anyway. |
| Pinet et al. (MiLeTS @ KDD 2026) | Of 373 real anomaly segments studied, none was purely cross-sensor. Proposes NOISEFLIP and NPROLL synthetic injections. | No TSFM, PCA, or CNN tested; only synthetic sinusoidal signals used. |

No existing paper puts a TSFM, with Forecast Fallback, against PCA and a CNN,
on the same ICS data, using marginal-preserving cross-sensor injections, scored
per anomaly type. That combination is the target of this project.

## Scope and methodology

The work is organized into two research directions and, within them, two
implementation phases.

**Direction 1, head-to-head comparison.** A TSFM (with and without Forecast
Fallback) is compared against PCA, windowed PCA, and a small CNN, on real
attacks plus synthetic cross-sensor injections designed to leave each
channel's own marginal distribution unchanged. Cost is measured alongside
accuracy: latency, memory, and energy on a fixed local GPU.

**Direction 2, root cause.** Ablations isolate which specific property
actually explains the ranking between methods: a fixed versus an adapting
notion of "normal," per-sensor versus joint modeling, the length of history
considered, a pretrained versus a plant-fitted model, and linear versus
nonlinear structure.

A null result is an acceptable outcome of this project ("PCA is enough unless
the anomaly is of type X"). An unexplainable result is not, which is why
Direction 2 runs regardless of what Direction 1 finds. PCA in particular is
tuned carefully: published PCA F1 scores on SWaT range from 0.23 to 0.88
depending purely on setup, so an under-tuned baseline would invalidate any
apparent TSFM advantage.

These directions are being executed in phases:

- **Phase A**, classical baselines on SWaT. A shared data loader, PCA and
  windowed PCA, a small CNN autoencoder, a shared detection rule, and a
  metrics harness, all built as a config-driven package rather than
  notebook-embedded code, so the same pipeline carries forward unchanged.
- **Phase B**, TSFM and Forecast Fallback. A pretrained TSFM (Chronos-2) is
  run zero-shot with the Forecast Fallback rule, first validated on a
  representative slice of data, reproducing the collapse-and-recovery effect
  reported by Lowe et al. before scaling up.

Later phases extend the same pipeline to the full model times Forecast
Fallback times injection times ablation grid, across datasets, once Phase A
and Phase B have produced a working, reproducible result.

## Data

This repository does not include the raw datasets. They are large (roughly
750 MB combined, with several individual files over GitHub's 100 MB limit)
and, in the case of SWaT and WADI, redistribution-restricted under a signed
iTrust data-use agreement. The project draws on three public ICS datasets:

- **SWaT** (Secure Water Treatment), a scaled-down water treatment testbed
  operated by iTrust at the Singapore University of Technology and Design,
  with roughly 450,000 timesteps across 51 sensors and actuators at 1 Hz,
  including 35 labeled attack segments.
- **WADI** (Water Distribution), a related iTrust testbed covering a larger
  water distribution network, with 127 sensors and actuators and 14 labeled
  attack segments.
- **HAI** (HIL-based Augmented ICS), a dataset built on a hardware-in-the-loop
  testbed combining boiler, turbine, and water-treatment processes.

SWaT is the primary dataset for this work; WADI and HAI are brought in later
to confirm that findings generalize beyond a single testbed.

## Reference papers

| Authors | Contribution | Code available |
|---|---|---|
| Goh, Adepu, Junejo, Mathur (CRITIS 2016) | Origin paper describing the SWaT testbed itself | No, dataset description only |
| Kravchik & Shabtai (CPS-SPC 2018 / IEEE TDSC) | Dynamic PCA, 1D-CNN, and autoencoder baselines on SWaT and related datasets | No |
| Fung, Srinarasi, Lucas, Phee, Bauer (ESORICS 2022) | Comprehensive evaluation of reconstruction-based ICS anomaly detectors | Yes |
| Liu & Paparrizos, TSB-AD (NeurIPS 2024) | Benchmark introducing the VUS-PR metric and a RobustPCA baseline | Yes |
| Lowe, Fung, Bauer (ICML 2026 workshop) | Zero-shot TSFMs on ICS data and the Forecast Fallback technique | No |
| Wen & Chen (arXiv 2608.22968) | Cost-aware comparison of classical methods against TSFMs for industrial monitoring | No |

## Expected outcomes

A results table comparing PCA, windowed PCA, a small CNN, and a TSFM (with and
without Forecast Fallback) on SWaT, reported overall, per attack segment, and
per anomaly type, with accuracy and running cost shown side by side. Either
outcome, a TSFM advantage or a null result in favor of the simpler baselines,
is reported together with the ablations from Direction 2 that explain why.

## License

MIT, see [LICENSE](LICENSE).
