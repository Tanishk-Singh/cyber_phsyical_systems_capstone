# Capstone context handoff: Time-Series Foundation Models vs. Simple Baselines for ICS Attack Detection

Written 2026-10-16 for Claude Code. It condenses a long planning conversation (paper reading, literature checks, direction-setting, proposal drafting). The goal now is to **set up the codebase and run the experiments**. Everything below is either something the user (Tanishk) said, something checked against the papers' PDFs, or something that came from a web-research run and has NOT been checked against the source. Each paper section says which. Where a detail is unknown, it says so rather than guessing.

---

## 0. TL;DR

- **Who:** Tanishk, MS Data Science student at the University of Arizona (College of Information Science), final semester (Fall 2026). ECE undergrad background, about 2 years of industry experience, prior research in federated learning / edge ML (DASS Lab, Prof. J.K. Dass).
- **Advisor:** Professor Rozhin Yasaei (MITHRA Lab, University of Arizona). PhD contact: Afjal Sarower, who shared the starting papers.
- **Deadline:** Capstone presentation **December 7, 2026**. This is the user's own date and is **not yet confirmed** with the advisor. Today is Oct 16, so about 7 weeks remain.
- **Core question:** Under honest evaluation (VUS-PR, thresholds from training/benign-validation data only, no point-adjust) and with cost counted, does a pretrained time-series foundation model (TSFM), with or without **Forecast Fallback**, beat PCA (plain and windowed) and a small CNN at detecting attacks on industrial control systems (ICS)? Which anomaly types does each method catch, especially synthetic cross-sensor anomalies that break relationships between sensors while each sensor stays individually normal?
- **No results exist yet.** Nothing in our research has been run on HAI. Every number below comes from published papers on other datasets (SWaT, WADI, TEP, CTown, BATADAL). The first job is producing the first real HAI numbers.
- **Datasets:** Start with **HAI** (public, GitHub). SWaT/WADI need an iTrust data request that can take weeks (the plan was to submit it around Oct 9; confirm whether that happened). BATADAL, TEP and CTown are public fallbacks.
- **Proposal status:** A solo-capstone proposal/SOW (.docx on a course template) was drafted Oct 7 and, per the user's notes, sent to Professor Yasaei around Oct 12. The user wants her direct approval of the proposal and directions. No advisor feedback has been recorded yet.

---

## 1. People, logistics, environment

| Item | Detail |
|---|---|
| Student | Tanishk (last name left as a placeholder in the proposal), MS Data Science, UA College of Information Science, entered Spring 2025, final semester Fall 2026 |
| Interests | AI efficiency, edge ML, federated learning, model compression, HPC, edge computing (FPGAs, Jetson Orin) |
| Advisor | Prof. Rozhin Yasaei, MITHRA Lab (UA); appointments in Cyber Operations, ECE, Emergency Medicine |
| PhD contact | Afjal Sarower (ECE PhD, MITHRA Lab). Public work is GNN-based hardware-Trojan localization (GINx4TL, IOLTS 2026). No public TSFM/ICS paper by him yet, so this direction may be new to the group. |
| Presentation | Dec 7, 2026 (unconfirmed) |
| Compute the user has used before (confirm what is available now) | DeltaAI HPC cluster (federated learning work); Brev GPU instances (8x A100). Interested in Jetson Orin, but **it is not known whether he has one**. Ask before assuming. |
| Working style notes | Wants natural, human-sounding writing in documents. Prefers clean, minimal visuals with implementation detail stripped out. |

**MITHRA Lab context** (why the project fits the lab): research areas are AI Foundation Models & Graph Learning, Cyber-Physical Systems, Hardware Security & Design Automation, Multimodal AI for Precision Health. Yasaei's notable work: TCAD 2022 (GNN-based hardware-Trojan detection), DAC 2021 (GNN4IP), two TIFS 2021 papers, CAD-IoT (ICCAD 2020), a Graph Deviation Network (GDN) for anomaly detection in additive manufacturing (ACM Transactions on Cyber-Physical Systems), IoT-GRAF (DATE 2024). Graph-based extensions of this project would fit the lab especially well.

**Open logistics to ask the advisor / Afjal** (still pending as far as this conversation knows):
1. What deliverable is expected (report length/format, workshop-style paper, code release)?
2. Is Dec 7 the right presentation date?
3. Whether the lab cares more about accuracy or about running cost on small edge hardware.
4. Whether the proposal's Ground Rules and Advisor Engagement appendix (team-style boilerplate) apply to a solo project.

---

## 2. The project

### 2.1 One-paragraph version
Industrial plants (e.g., water treatment) are attacked by manipulating sensor/actuator values, often for long durations. The usual defense learns normal behavior and flags deviations. TSFMs (Chronos-2, Moirai-2.0, TimesFM, MOMENT, ...) promise zero-shot use without plant-specific training, but published work shows they **copy attacked inputs into their own forecasts**, so forecast error collapses inside long attacks and detection fails. A fix, **Forecast Fallback (FF)**, helps a lot, but it was never compared against simple baselines like PCA or a small CNN. Other work shows simple fitted models often beat TSFMs at a fraction of the cost, but not on attack data. This project tests the head-to-head on attack data with honest metrics and with synthetic cross-sensor anomalies, then investigates why winners win.

### 2.2 The four directions (as agreed with the user)
1. **Direction 1 (LOCKED IN, committed): head-to-head comparison.** TSFM (with/without FF) vs PCA vs windowed PCA vs small CNN on HAI (plus SWaT/WADI if access arrives), including synthetic cross-sensor injections, reported **by anomaly type**, plus cost (latency, memory, energy).
2. **Direction 2 (committed, sized to what Direction 1 shows): root cause.** Ablations to find which architectural property explains the results.
3. **Direction 3 (stretch / future work): enhancement.** Strengthen whichever quality matters (e.g., better thresholds for FF, a boundary-spike trigger, joint multivariate fallback).
4. **Direction 4 (future work): new architectures.** Spiking neural networks (SNNs) and state-space models for the same task. Caveats: SNN efficiency gains need neuromorphic hardware; on GPU/Jetson an SNN is usually slower/less efficient than a small ordinary network (earlier rated medium-to-high risk). State-space-model literature was **not researched** in this conversation.

Directions 1 and 2 are a realistic semester. Directions 3 and 4 are probably beyond one semester and are framed as future work in the proposal. Confirm scope with the advisor.

### 2.3 Why the gap looks open (literature check, limited searches)
No paper found combines: a TSFM (Chronos-2, Moirai-2.0, ...) + Forecast Fallback + PCA and CNN baselines + SWaT/HAI + marginal-preserving cross-channel injections reported per anomaly type. Closest partial matches: Pinet et al. (injection protocol, but no TSFMs/PCA/CNN, synthetic sinusoids only), mTSBench (PCA, CNN and 2 LLM-based detectors, but no marginal-preserving injection and no per-anomaly-type results), TSB-AD (PCA, CNN, several TSFMs, curated real data only). "None found" is not proof of absence; new 2026 preprints keep appearing. No published HAI results were found for any of these methods.

---

## 3. Experiment specification: Direction 1

### 3.1 Models
- **TSFM, zero-shot:** Chronos-2 as the main model. Also a smaller one such as **Chronos-Bolt-tiny** (about 9M parameters; Bolt family is tiny/mini/small/base at 9M/21M/48M/205M) for the cost comparison. Optional: Moirai-2.0, TSPulse (IBM, about 1M parameters per its model card), TimesFM.
  - Each with and without **Forecast Fallback**.
  - **Channel-independent control:** run the TSFM one sensor at a time vs. all sensors together. Chronos-2 supports grouping (group attention aggregates information across series in the same group; giving a series' variates the same group ID vs. a unique ID each switches between multivariate and univariate mode). Moirai-2.0 reportedly "does not support cross-variate forecasting" (treats multivariate as independent univariate tasks), so expect near-chance results from it on cross-channel-only anomalies. These two claims come from a research run and should be verified against the model papers (Chronos-2: Ansari et al., arXiv 2510.15821; Moirai-2.0: Liu et al., arXiv 2511.11698).
- **PCA** (reconstruction error, single timestep).
- **Windowed / dynamic PCA** (PCA on stacked lagged vectors). Required because lagged *linear* dependence is something PCA can capture once given a window, so plain PCA would be an unfairly weak baseline.
- **Small CNN** (and optionally AE / LSTM, TCN-AE) using the Fung et al. code.

### 3.2 Forecast Fallback (as described in Lowe et al., verified against the PDF)
- Monitor per-feature forecast error (MAE) in real time.
- If a feature's error exceeds its **99.5th-percentile threshold for 3 consecutive time steps**, stop trusting the observed value and use the model's own prediction as the surrogate input for that step. Continue forecasting on the reconstructed input sequence.
- Too-strict FF threshold means FF never triggers (same failure as without FF). Too-loose threshold causes false positives and stale predictions.
- The paper left the best FF thresholding strategy to future work and did not explore multivariate FF.
- Their grid: input length I in {32, 64, 128}, horizon H in {1, 16, 32}.

### 3.3 Detection rule (shared by all methods, from Fung et al., verified)
Flag a timestep as an attack when the error exceeds threshold τ for **w consecutive steps**. Set τ from the distribution of **benign-validation errors** (e.g., 99.5th percentile). Fung's search range: τ-percentile in [0.95, 0.99995], w in [1, 100]. **Never set thresholds from test data.** (Lowe et al. and Uray et al. both tuned on test data, which inflates their F1.)

### 3.4 Data
- **HAI first** (public). Then SWaT and WADI if access arrives. Fallbacks: BATADAL, TEP, CTown (all public).
- Normalize/scale using **training statistics only**. Keep train / benign-validation / test splits explicit. Document feature selection (Fung removed 11 features from SWaT and 10 from WADI for distribution shift).
- If SWaT arrives: remove attacks that never physically affected the plant (Fung et al. "attack cleaning").
- **Synthetic cross-sensor injections** into normal data, which must keep **each channel's own marginal distribution unchanged** (single-channel detectors should be near chance on them):
  - **NOISEFLIP** (Pinet et al.): flip the sign of the shared noise on a random subset of channels. On Pinet's synthetic data this is 9 phase-shifted sinusoids with shared noise; **how to adapt this to real plant data is a design decision** (e.g., flip the sign of the residual from a smoothed baseline on selected channels).
  - **NPROLL** (Pinet et al.): circularly shift (roll) the selected channels within the anomalous segment.
  - **A nonlinear injection to design:** break a nonlinear relationship (e.g., flow vs. sqrt of a pressure difference, threshold/valve behavior) while keeping each sensor's own values looking normal.
  - Validate each injection (e.g., KS test or histogram check that per-channel marginals within the segment are unchanged; confirm channel-independent models score near chance).

### 3.5 Metrics (non-negotiable rules)
- **Primary: VUS-PR** (range-aware AUPRC, robust to small lags, threshold-independent; recommended by TSB-AD).
- **Range-based F1** (Fung et al.; existence reward) alongside. Optionally the Numenta score.
- **ROC-AUC** reported too, so numbers can be compared with Lowe et al.
- **Never report point-adjusted (PA) F1** alone. PA flags the whole attack window if any point is hit and is badly biased (in TSB-AD a random score ranks 26th of 33 under PA-F1).
- Report **by anomaly type**: single-sensor attacks vs. cross-sensor ones; real attacks vs. each injection type.
- Several runs/seeds wherever randomness is involved; report spread.
- Point-F1 can be misleading (Fung Figure 4: two cases with equal point-F1 0.75 but very different quality).
- Reference definitions: ROC-AUC = probability a random attack timestep scores higher than a random normal timestep (below 0.5 = backwards). AUPRC baseline = anomaly rate, not 0.5.

### 3.6 Cost measurements
Per-window latency, memory (VRAM/RAM), model size, and energy where measurable, for each method, on available GPU and on a **Jetson Orin if the user can get time on one**. Wen & Chen's cost-comparison style (below) is a good template: model size, batch-1 p50 latency, peak VRAM at batch 16.

### 3.7 Expectations and what counts as success
- No fixed accuracy target. Reported PCA F1 on SWaT ranges from about 0.23 to 0.83 across papers depending on setup, so success means a **fair, reproducible comparison**: every method tuned the same way, training-only thresholds, spread across runs, clear answer on which method wins for which anomaly type at what cost.
- **A null result is acceptable** (e.g., PCA is enough unless the anomaly is type X). Earlier work (Kravchik & Shabtai; Sarfraz et al.) already suggests PCA is hard to beat on standard ICS data. What must be avoided is a result nobody can explain, hence Direction 2.
- If FF cannot be reproduced on HAI, report that after checking settings against the paper. If a TSFM is too slow on full test data, use a subset of windows and say so.
- **Careful PCA tuning is mandatory**, otherwise reviewers will reasonably doubt any TSFM win.

---

## 4. Direction 2 (root-cause ablations)

Candidate explanatory factors to ablate on top of Direction 1 results:
1. **Fixed vs. adapting reference for "normal"** (FF on/off). PCA fits a global normal subspace; a TSFM re-adapts to each window and absorbs the attack.
2. **Capacity bottleneck** (PCA's limited capacity means it cannot reconstruct novel patterns).
3. **Pretrained general prior vs. fitted on the plant's own data** (TSFMs are pretrained on weather/traffic/etc.). Lowe et al. found LoRA fine-tuning gave negligible gain (0.517 to 0.519 on one setting).
4. **Joint sensor modeling vs. per-sensor** (channel-independent TSFM vs. all-sensors-together; PCA models linear cross-sensor structure natively).
5. **History window / lag** (windowed vs. single-timestep).
6. **Linear vs. nonlinear structure** (the one place a nonlinear windowed model, CNN/LSTM/TSFM, has a structural edge over PCA).
7. **Cost** (PCA is among the fastest in TSB-AD; FF adds hyperparameters and stale-prediction risk).

**Where TSFMs could win:** nonlinear or time-lagged sensor dependencies that PCA cannot capture, point anomalies (per TSB-AD), cold start (no training data). Multivariate TSFMs (Chronos-2, Moirai-2.0) have not been tested multivariately in published benchmarks that we found.

**Important nuance from the literature:** "time-lagged" and "nonlinear" are separate claims. A windowed linear model already handles lag (Kravchik's windowed PCA; Pinet's linear autoencoder reached AUC-ROC 0.97 on NPROLL). Only the nonlinear part genuinely calls for CNN/LSTM/TSFM.

---

## 5. Directions 3 and 4 (parking lot)

Direction 3 candidates (from earlier discussion and a reading-list research run):
- **Adaptive thresholding for FF** (lowest risk; explicitly left open by Lowe et al.). Options: per-feature conformal thresholds, CUSUM trigger, boundary-spike trigger.
- **Boundary-spike trigger:** Uray et al. observed forecast error spikes at anomaly entry/exit boundaries, so FMs may be good change-point detectors; use that to trigger fallback earlier.
- **Cross-channel / graph-guided multivariate FF** (GDN-style learned sensor graph; fits the lab's GNN work). Riskier: Pinet et al. suggest real SWaT/WADI labels may not reward cross-channel modeling, so a null result is possible. Evaluate on synthetic injections too.
- **Evasion robustness of TSFM-based detectors** (adapt Erba et al. concealment attacks to a Chronos/Moirai forecast-error detector with/without FF). A research run found no prior work on attacking a TSFM-based ICS detector, which would make it the most novel option, but it is medium feasibility (black-box attack is the safer scope).
- **Federated ICS detection with small TSFMs:** medium-low feasibility for one semester; only if the advisor wants it.

Direction 4: SNNs and state-space models. See caveats in section 2.2.

---

## 6. Datasets

| Dataset | Type | Features | Access | Notes |
|---|---|---|---|---|
| **HAI** | Hardware-in-the-loop augmented ICS testbed (boiler, turbine, water treatment processes) | roughly 60-90 (not verified; version-dependent) | **Public**, GitHub: github.com/icsdataset/hai (a possibly related repo github.com/xuhongzuo/HAI-dataset also appeared in search results; unverified). Shin et al., CSET 2020/2021. | Less "used up" than SWaT/WADI. Multiple HAI versions exist. **Check the README, pick one version, and record it.** Pinet et al. and most diagnostics used SWaT/WADI, not HAI, so HAI results are new. |
| SWaT | Real water treatment testbed (SUTD iTrust) | 51 | Request from iTrust (can take weeks) | Some attacks had no physical effect and should be removed (Fung et al.). TSB-AD's curated SWaT-M has only 2 series, so it is unreliable for comparison. |
| WADI | Real water distribution | about 123 to 127 | iTrust request | "WADI-127" is used in Sarfraz et al. |
| BATADAL | Simulated water | 43 | Public | |
| TEP | Simulated chemical (Tennessee Eastman) | 52 | Public | |
| CTown | Simulated water | 39 | Public | In Lowe et al.: 39 features, n = 149,760, 14 days; example attack holds a value of 1.29 for 373 consecutive seconds. |

**Dataset pitfalls:** point-adjust inflates results on all of these; benchmarks like SWaT/WADI have been criticized as unsuited for multivariate anomaly-detection evaluation (Wagner et al., TimeSeAD, via Pinet et al.), and Pinet et al. found no purely cross-sensor anomalies in 373 long labeled segments across 8 real benchmarks.

---

## 7. Literature knowledge base

Verification tags: **[PDF]** = checked against the paper PDF the user uploaded in this session. **[RESEARCH]** = reported by a web-research run, not checked against the source. **[EARLIER]** = from the user-pasted reading earlier in the conversation (full text was read then; not re-checked in this session).

### 7.1 Lowe, Fung, Bauer: "On the Application of Time-Series Foundation Models for Detecting Long-Context Anomalies in Industrial Control Systems" (ICML 2026 workshop paper) [PDF]
- Zero-shot **Chronos-2** and **Moirai-2.0** on **SWaT, TEP, CTown**; forecast-error (MAE) detection. Pretrained TSFMs struggle with long-duration anomalies because they repeat anomalous inputs in the forecast.
- Baseline (no FF) ROC-AUC is low across all configs, roughly **0.2 to 0.6** (SWaT about 0.24 to 0.27, below chance).
- **FF raises ROC-AUC by 0.1 to 0.45**; CTown ROC-AUC goes from below 0.5 to **above 0.89 for both models**. F1 with FF and the optimal threshold taken from the ROC curve (i.e., test-derived): Chronos-2 SWaT 0.248, TEP 0.756, CTown 0.742; Moirai-2.0 SWaT 0.722, TEP 0.583, CTown 0.831. Abstract claims F1 0.83 on CTown is competitive with trained ICS models (prior work reports 0.7 to 0.9 on CTown, Erba et al. 2024).
- LoRA fine-tuning of Chronos-2 on TEP: ROC-AUC 0.517 to 0.519 (Table 3: 0.5197 / 0.5193 / 0.5191 / 0.5194 at 10/20/50/100% of validation data). Negligible.
- Open items: thresholding strategy for FF; multivariate FF; threshold choice used test data; no PCA/small-model comparison.
- **Unverified figures from earlier notes, not found in the PDF text (may be in a figure legend): Moirai-2.0 CTown AUC 0.912, Chronos-2 TEP AUC 0.903, Chronos-2 SWaT 0.523. Do not quote these without checking Figure 4.**

### 7.2 Wen & Chen: "Do Time-Series Foundation Models Pay Off for Industrial Monitoring? A Cost-Aware Empirical Study" (arXiv 2608.22968, Aug 2026, preprint) [PDF]
- Frozen/zero-shot TSFMs (MOMENT-small, Chronos-T5 tiny, TimesFM 2.5) vs. fitted lightweight models on **C-MAPSS, MIMII, BDG2**.
- C-MAPSS: **TCN-AE** AUROC 0.9570 / AUPRC 0.8960 / F1 0.7959 vs. **MOMENT reconstruction** 0.7310 / 0.3080 / 0.1190. OCSVM beats MOMENT on MIMII (from earlier reading). On BDG2 forecasting, TimesFM 2.5 has the lowest MAE but AUPRC is similar.
- Cost: model size **138,053 KB (MOMENT) vs. 13.8 KB (TCN-AE)**; batch-1 p50 latency about 0.29 ms (TCN-AE) vs. about 13.4 ms (MOMENT; the 13.4 is from earlier reading); batch-16 peak VRAM 691 MB vs. 2.6 MB.
- Conclusion: TSFMs are task-dependent, not default replacements; compact fitted models win when target-normal data is available.
- Limitations: no CPS-security datasets, no adversarial attacks, no fine-tuning, small/older TSFM variants.

### 7.3 Fung, Srinarasi, Lucas, Phee, Bauer: "Perspectives from a Comprehensive Evaluation of Reconstruction-based Anomaly Detection in Industrial Control Systems" (ESORICS 2022, LNCS 13556, pp. 493-513) [PDF]
- Datasets BATADAL, SWaT, WADI. Models AE, CNN, LSTM.
- Findings: architecture and hyperparameters matter little, much smaller models match large ones (Finding 1a); metric choice changes conclusions (point-F1 misleading; use range-F1 / Numenta); four techniques matter: **feature selection, attack cleaning, benign-data shuffling, early stopping** (Finding 1c / #2). Range-F1 tuning gives better TP:FP ratios than point-F1 tuning.
- Detection rule and search space as in section 3.3. Models: AEs use history h=0; CNNs 8 layers, 32 filters, kernel 3, history 50/100/200 in the search; LSTMs history 50/100 in the search. Their reference configs for comparison are a 32-unit CNN with history 200 and a 2-layer, 256-unit LSTM with history 50.
- Open items: overlap rewards for range metrics; datasets beyond the three.
- **Code: https://github.com/pwwl/ics-anomaly-detection.** Whether it supports HAI out of the box is **unverified**; expect to write a HAI loader.

### 7.4 Uray, Messineo, Kwitt, Huber: "Exploring Zero-Shot Foundation Models for Multivariate Time Series Anomaly Detection" (arXiv 2607.12454, July 2026; EUROCAST 2026 version of record pending; preprint) [PDF]
- **TimesFM 1.2.0** (univariate), zero-shot, **SWaT only**. Strategy 1: per-channel forecast, thresholded error, errors summed across channels; fails (once the window lies entirely inside the anomaly, the FM adapts to the new dynamics and the score collapses; the research run says F1 close to zero, but that number is not in their Table 1). Strategy 2: per-channel encoder embeddings concatenated, then an **Isolation Forest** trained on normal data: **F1 0.480**.
- Their Table 1 (SWaT F1; baselines taken from their ref. [16]): Random 0.218; sensor range deviation 0.231; L2-norm 0.782; 1-NN distance 0.782; **PCA Error 0.833**; 1-layer MLP 0.771; MLPMixer 0.780; single Transformer block 0.787; 1-layer GCN-LSTM 0.829; TimesFM + IF 0.480.
- Forecast error **spikes at anomaly boundaries**, so FMs may be useful for change-point detection.
- Caveats: one model, one dataset; forecasting threshold **optimized on the test set to maximize F1** (inflates numbers).

### 7.5 Liu & Paparrizos: "The Elephant in the Room: Towards A Reliable Time-Series Anomaly Detection Benchmark" (TSB-AD, NeurIPS 2024 Datasets & Benchmarks) [PDF + EARLIER]
- 1,070 curated series from 40 datasets (870 univariate TSB-AD-U, 200 multivariate TSB-AD-M), 40 detectors, 10 metrics. **VUS-PR** is the most robust (least sensitive to lag, unbiased, threshold-independent).
- Univariate mean VUS-PR [PDF Table 5]: Sub-PCA 0.42, KShapeAD 0.40, AutoEncoder 0.26, TranAD 0.26, FITS 0.26, OFA 0.24, LOF 0.17, AnomalyTransformer 0.12. From earlier reading: POLY 0.39, MOMENT-FT 0.39, MOMENT-ZS 0.38, TimesFM 0.30, Chronos 0.27.
- Multivariate mean VUS-PR [PDF Table 5]: CNN 0.31, PCA 0.31, LSTMAD 0.31, OmniAnomaly 0.31; **OFA (the only FM-style detector in the multivariate comparison) 0.21**. Chronos/TimesFM/MOMENT were **not** tested multivariately (earlier reading).
- Findings: statistical methods often strong; simpler architectures beat complex transformers; FMs do better on point anomalies than sequence anomalies (same copy failure); MOMENT suffers data contamination (TSB-AD 0.38 vs. MOMENT-Eval 0.12, earlier reading); fine-tuned MOMENT beats zero-shot (p=6.64e-9) but the VUS-PR gap is small.
- **Code: https://github.com/TheDatumOrg/TSB-AD** (`pip install TSB-AD`), useful for VUS-PR.

### 7.6 Kravchik & Shabtai: "Efficient Cyber Attack Detection in Industrial Control Systems Using Lightweight Neural Networks and PCA" (IEEE TDSC 19(4):2179-2197; online 2021; arXiv 1907.01216) [RESEARCH]
- Datasets SWaT, WADI, BATADAL. Models: PCA reconstruction, **windowed PCA (equivalent to dynamic PCA, window 7 on SWaT)**, 1D CNN, undercomplete AE, VAE, frequency-domain AE.
- SWaT F1: PCA 0.817, windowed PCA **0.879**, AE 0.844, 1D CNN 0.871. WADI F1: PCA 0.653, windowed 0.683, 1D CNN 0.714, AE 0.750. BATADAL score: AE 0.926, PCA 0.898, 1D CNN 0.894.
- Authors expected AEs to beat PCA via nonlinearity; margin was small; on SWaT many relations are linear. They recommend trying PCA first.

### 7.7 Sarfraz et al.: "Position: Quo Vadis, Unsupervised Time Series Anomaly Detection?" (ICML 2024, PMLR 235:43461-43476; arXiv 2405.02678) [RESEARCH]
- PCA reconstruction error had the best F1 on **SWaT (0.833)** and **WADI-127 (0.501)** among methods compared. Replacing trained SOTA deep models with linear "distillations" performed on par; the authors conclude deep models mostly learn linear mappings on current datasets.

### 7.8 Pinet, Cumin, Berlemont, Vaufreydaz: "Anomalies in Multivariate Time Series Benchmarks Are Mostly Univariate" (MiLeTS 2026 workshop @ KDD; arXiv 2606.02670) [RESEARCH]
- 373 long labeled anomaly segments across 8 real benchmarks (GECCO, MSL, SMAP, PSM, SMD, SWAN-SF, SWaT, WADI; no HAI): **none** purely cross-channel; every correlation break co-occurs with a univariate deviation. CrossAD channel-dependent never improved over channel-independent on real benchmarks (7 datasets, WADI excluded).
- Synthetic: 9 phase-shifted sinusoids with shared noise; **NOISEFLIP** and **NPROLL** keep each channel's marginal unchanged. Channel-dependent models detect them (LinearAE-CD AUC-ROC 0.97 on NPROLL, 0.84 on NOISEFLIP; CrossAD-CD 0.85 / 0.55; CATCH 0.77 / 0.66); channel-independent models score near chance (0.47 to 0.60). A linear AE beat both SOTA models.
- No TSFMs, no PCA, no CNN. Workshop paper, not archival peer review.

### 7.9 mTSBench (Zhou, Brif, Lourentzou; TMLR 2026; arXiv 2506.21550) [RESEARCH]
- 24 detectors (PCA, CNN/DeepAnT-style, AE, LSTMAD, USAD, OmniAnomaly, TranAD, AnomalyTransformer, ...) and only two LLM-based FMs (OFA, ALLM4TS); 344 series from 19 sources incl. a GutenTAG synthetic set (30 series x 20 dimensions; amplitude, extremum, frequency, mean, pattern, pattern-shift, platform, trend, variance, mode-correlation; 1/2/5/10/20 dimensions corrupted). PCA ranks 1st overall by mean VUS-PR, CNN 4th. Does not report per anomaly type; its mode-correlation anomalies are not marginal-preserving.
- **GutenTAG AUC-ROC numbers (PCA 0.84, OFA 0.81, CNN 0.76, ALLM4TS 0.75) were read off a heatmap with possible row misalignment. Verify against the figure before quoting.**

### 7.10 Other results used for context
- **GDN** (Deng & Hooi, AAAI 2021) reports PCA F1 of only **0.23 on SWaT and 0.10 on WADI** vs. GDN 0.81 / 0.57. This conflicts with Kravchik (0.817) and Sarfraz (0.833) and probably reflects an under-tuned PCA baseline, so do not cite it as evidence that nonlinear models beat PCA. [RESEARCH]
- **Erba et al., ACSAC 2020**, "Constrained Concealment Attacks against Reconstruction-based Anomaly Detectors in ICS": constrained attacker (subset of sensors); white-box and black-box concealment attacks evade AE/LSTM/CNN detectors, demonstrated on a physical testbed. Code: github.com/scy-phy/ICS-Evasion-Attacks. [RESEARCH]
- TEP: Xiao, Kordon & Sen (PHM Society 2023) report a deep AE (3 input lags) beating standard PCA on TEP faults; simulated chemical process, not attack data. [RESEARCH]
- TSB-AD related: Kim et al. (ICLR 2026) zero-shot Chronos/TimeMoE/Moirai/MOMENT on TSB-AD-M, noting only Moirai explicitly uses multichannel information. TimeRCD (arXiv 2509.21190), ChronosAD (arXiv 2606.01300; supervised, Chronos embeddings + BiLSTM + attention, 11 benchmarks incl. SWaT; reports strong results, in tension with Uray et al.). [RESEARCH]

### 7.11 Extra reading list (from a web-research run; unverified; use as pointers)
- Evaluation critiques: Lamberts et al. "SoK: Evaluations in Industrial Intrusion Detection Research" (J. Systems Research 2023; arXiv 2311.02929; 609 publications, only 1.3 datasets evaluated on average); Wu & Keogh, "Current Time Series Anomaly Detection Benchmarks are Flawed..." (IEEE TKDE 2023; arXiv 2009.13807); Kim et al., "Towards a Rigorous Evaluation of Time-series Anomaly Detection" (AAAI 2022; arXiv 2109.05257; point-adjust makes even random scores look SOTA; proposes PA%K).
- Adversarial: Zhang et al., "Are Time-Series Foundation Models Deployment-Ready?" (arXiv 2505.19397; TSFMs susceptible to adversarial attacks, forecasting only); Liu et al. (Computers & Security vol. 140, 2024) adversarial attacks on LSTM-ED ICS detectors; Erba et al., "On Practical Realization of Evasion Attacks for ICS" (RICSS '24 @ CCS).
- Small TSFMs for edge: Chronos-Bolt (Amazon chronos-forecasting README claims up to 250x faster and 20x more memory efficient than original Chronos of the same size), TSPulse (ICLR 2026; arXiv 2505.13033; about 1M parameters; model card claims +20% on TSB-AD), Tiny-TSM (arXiv 2511.19272, 23M, forecasting-only technical report).
- Datasets: HAI paper (Shin et al., CSET 2020/2021); "A Roadmap to Available ICS Datasets and Testbeds" (arXiv 2608.30332).
- Poisoning/federated: Ozbek et al. (arXiv 2608.23547; 1% training contamination crushes several SWaT detectors but PCA's worst-case F1 drop was only 0.045); Nosrati et al. (arXiv 2605.27486).

### 7.12 Paper status summary
| Paper | Status |
|---|---|
| Lowe et al. (Forecast Fallback) | ICML 2026 workshop paper |
| Wen & Chen (cost-aware) | arXiv preprint |
| Fung et al. | Published, ESORICS 2022 |
| Uray et al. | arXiv preprint (EUROCAST 2026 version of record pending) |
| Liu & Paparrizos (TSB-AD) | Published, NeurIPS 2024 |
| Erba et al. (evasion attacks) | Published, ACSAC 2020 |
| Kravchik & Shabtai | Published, IEEE TDSC |
| Sarfraz et al. | Published, ICML 2024 (position paper) |
| Pinet et al. | Workshop paper, arXiv |
| mTSBench | Published, TMLR 2026 |
| Yasaei et al. CAD-IoT / GDN additive manufacturing | Published (ICCAD 2020 / ACM TCPS) |

---

## 8. What we know about HAI results: nothing yet

Asked on Oct 9 "which method has performed best on HAI so far": **no method has HAI results in anything we read or ran.** The closest evidence is SWaT: PCA-style methods did best (Sarfraz PCA F1 0.833; Kravchik windowed PCA 0.879), TimesFM+IF did worst (0.480), and FF has never been compared with PCA. These numbers are not comparable across papers (different setups and thresholds), and HAI is a different plant, so do not assume the SWaT ranking transfers. The first deliverable is the HAI baseline table.

---

## 9. Proposal/SOW document (already drafted; for consistency)

Built from the user's course template (a Senior-Design-style Project Proposal & Statement of Work .docx) as `Project_Proposal_SOW.docx`, 13 pages, adapted to a solo capstone.
- **Working title:** "Time-Series Foundation Models vs. Simple Baselines for ICS Attack Detection". Dated October 7, 2026.
- **Sections:** Executive Summary (297 words; includes Table 1 of the four directions), Literature Review (8-paper table, "where this project fits", three design implications, user-insights placeholder, 9 references), Research Project Deliverables (format, analysis, accuracy, null results, data availability), Timeline & Gantt (Gantt chart still a placeholder), Ethics checklist (28 questions; all N except 22 and 28 = Maybe for over-reading results and dual-use; 21/25/27 N/A as no participants), Approvals, Appendix (template boilerplate left as is).
- **Placeholders left:** user's last name, instructor name, advisor signature date, Gantt chart, user-insights section.
- **Planned milestones in the proposal (proposed dates, all unconfirmed):**

| Milestone | Date |
|---|---|
| Proposal sent to Prof. Yasaei, with questions on report format and expectations | Oct 9 |
| SWaT and WADI data request submitted to iTrust | Oct 9 |
| Environment set up, HAI downloaded, Fung et al. code running | **Oct 16 (today)** |
| Baselines on HAI done: PCA, windowed PCA, CNN (training-only thresholds, VUS-PR) | Oct 23 |
| Foundation model running on HAI, with and without Forecast Fallback | Oct 30 |
| Synthetic cross-sensor injections built and checked (each sensor's own values unchanged) | Nov 6 |
| First full results table by anomaly type; decide whether SWaT/WADI arrived or to use another public dataset | Nov 13 |
| Root-cause ablations finished | Nov 20 |
| Cost measurements done (latency, memory, energy) | Nov 25 |
| Full report draft to advisor | Nov 30 |
| Slides ready and practiced | Dec 4 |
| Capstone presentation (date to be confirmed) | Dec 7 |
| Proposal signed by advisor | to be confirmed |

The schedule is tight, so order matters more than any single date: baselines first, then the foundation model, then the injections. Thanksgiving week (Nov 26) is a known squeeze. If SWaT/WADI have not arrived by mid-November, use another public dataset and label results clearly by dataset.

Promised deliverables in the proposal: a written report (introduction, methods, results, discussion) with a results table broken down by anomaly type and at least two figures; the Dec 7 presentation; a code repository with configs so the comparison can be rerun.

---

## 10. Suggested setup plan for Claude Code

This is a suggested plan consistent with everything above; adjust to the machine you land on. Ask the user which compute is available (local GPU, DeltaAI, Brev, Jetson) before assuming.

### 10.1 Proposed repo layout
```
capstone-ics-tsfm/
  README.md                 # question, plan, how to rerun
  environment.yml | pyproject.toml
  data/                     # gitignored
    raw/hai/  raw/swat/ ...
    processed/
  src/ics_tsfm/
    data/        # loaders (hai.py first; swat/wadi/batadal/tep/ctown later), splits, scaling (fit on train only)
    injection/   # noiseflip.py, nproll.py, nonlinear.py, validate_marginals.py
    models/      # pca.py, windowed_pca.py, cnn.py (reuse Fung code), tsfm_chronos.py, forecast_fallback.py
    detect/      # tau/w rule, threshold fitting on benign-validation only
    metrics/     # VUS-PR (TSB-AD), range-F1, ROC-AUC; NO point-adjust
    cost/        # latency, memory, energy harness
  experiments/   # YAML configs: dataset, split, model, injection, seed
  results/       # parquet/CSV per run + aggregated tables
  notebooks/     # analysis, per-anomaly-type plots
  tests/         # unit tests (leakage checks, injection marginal checks, metric sanity)
```

### 10.2 Order of work
1. **Environment.** Python env; install PyTorch, scikit-learn, pandas/pyarrow, `TSB-AD`, Chronos package (check Amazon's chronos-forecasting README for the current Chronos-2 / Chronos-Bolt API; do not assume the install command), and Fung's repo dependencies. Clone `pwwl/ics-anomaly-detection` and `TheDatumOrg/TSB-AD`. Pin versions. Log hardware.
2. **HAI.** Download from `icsdataset/hai` (check README for versions, file layout, labels, sampling rate). Choose one version and record it. Define: train (normal only), benign-validation (normal only, for τ), test (with attacks). Inspect attack count, durations (long vs. short), per-attack affected sensors, feature types (continuous vs. discrete/actuator). Decide feature selection; document it. Fit scalers on train only.
3. **Fung baselines on HAI** (write a HAI loader for the repo, or port the models): AE / CNN / LSTM, with range-F1 and VUS-PR. This is the ground truth table.
4. **PCA and windowed PCA** with careful tuning on validation (number of components, window length). Same τ/w detection rule.
5. **TSFM zero-shot** (Chronos-2 first; Chronos-Bolt-tiny for cost). **First sanity check: reproduce the copy failure** (error collapses inside long attacks; baseline ROC-AUC near or below 0.5). Then implement **FF** exactly as in 3.2 and confirm it lifts ROC-AUC the way Lowe et al. report (0.1 to 0.45). Use per-feature thresholds from benign-validation data, not test data.
6. **Metrics harness** shared by all models: VUS-PR (TSB-AD), range-F1, ROC-AUC, all reported per anomaly type. No point-adjust.
7. **Injections:** implement NOISEFLIP/NPROLL adaptation and a nonlinear injection; add tests asserting per-channel marginals are unchanged inside injected segments and that a channel-independent detector is near chance. Run all models on injected data.
8. **Channel-independent control** (TSFM per-sensor vs. grouped) and **ablations** from section 4.
9. **Cost harness** (latency batch=1, memory, model size, energy on Jetson if available).
10. Keep everything config-driven with fixed seeds; save per-timestep scores so metrics can be recomputed.

### 10.3 Leakage and correctness checks to bake in
- Scalers, PCA, thresholds, feature selection: fit on train / benign-validation only.
- Test set untouched until the final evaluation; no per-method threshold tuning on test.
- Unit-test the detection rule (τ, w) and the FF trigger on toy sequences.
- Check labels alignment (lags between attack labels and effect) since VUS-PR is lag-tolerant but F1 is not.
- Confirm no data from attack periods leaks into "normal" training (Fung's attack-cleaning concern).

---

## 11. Open questions and unverified items

1. HAI details not verified in this conversation: which version to use, exact feature count, sampling rate, label format, and whether Fung's repo handles it.
2. Whether the iTrust request for SWaT/WADI was actually submitted (planned for Oct 9) and its status.
3. Whether Dec 7 is correct; what report format the advisor wants; whether Professor Yasaei has replied to the proposal.
4. Which compute is available; whether a Jetson Orin is accessible.
5. How to adapt NOISEFLIP to real data (it was defined on synthetic sinusoids with shared noise).
6. Unverified numbers (Lowe's 0.912 / 0.903 / 0.523; mTSBench GutenTAG heatmap values; Kravchik/Sarfraz/Pinet/GDN/Erba numbers not checked against their PDFs).
7. Whether Moirai-2.0 / Chronos-2 channel-handling claims hold in the current library versions.
8. State-space model and SNN literature was not researched.
9. Authorship/AI-use policy for the written deliverables was not discussed.
