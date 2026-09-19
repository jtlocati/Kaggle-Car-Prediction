# PS S6E9 — Public Code Survey (scraped 2026-09-19)

Source: Code tab, sorted by public score descending. 180 notebooks enumerated via the
`kernels.KernelsService/ListKernels` endpoint; source of the top notebooks read directly.

---

## THE HEADLINE FINDING

**The entire 0.94650–0.94656 top of the Code tab is public-LB noise mining, not modelling.**

Tell-tale signal: those notebooks run in **21–29 seconds** and take other notebooks'
*outputs* as inputs (`DATA_SOURCE_TYPE_KERNEL_VERSION`). They train nothing.

Everything up there sits on ONE file: `jazivxt/s6e9-zoom-zoom-baseline/submission_latest_best.csv`
(public 0.94651).

`megayak/s6e9-0-94656-reading-the-public-split` states it outright in its own header:

> "What is exploited here is the sampling noise of the public 20%. It moves the public
> score and, by construction, does nothing for the private one."

Its method: two exact AUC identities that let a pair of submissions *read* the labels of the
public split. Then swap adjacent rank-bands where the public sample's rate happens to invert
against the model's ordering. Measured result: **two hits in fifteen probe submissions**,
mean just below zero — exactly what the noise model predicts. The author is explicit that
the hits "are not skill."

Same-family notebooks: `chinzorigtganbat/s6e9-does-breaking-ties-help`,
`nina2025/ps-s6e9-top-subm-...round(5)` and `round(4)` (rounding to manufacture ties, then
breaking them), `megayak/s6e9-the-ceiling-test-is-circular`.

### Consequence
- Do not fork anything above ~0.94647. It is fitted to 57,314 rows you cannot see.
- The honest public ceiling is ~0.94647 (jazivxt), and even that is 80% external blend.
- Cleanest genuinely-trained public score: **~0.94638–0.94639**.
- ~150 teams sitting at 0.94650–0.94656 on public are carrying this exposure into the
  private LB. That is your opportunity, not your benchmark.

---

## CV → LB CALIBRATION (clean notebooks, no kernel inputs)

| Local CV | Public LB | Notebook |
|---|---|---|
| 0.94583 | 0.94590 | evgendvorkin/s6e9-single-xgb-cv-0-94583 |
| 0.946072 | 0.94614 | tamerlanomralinov/s6e9-cv-0-946072-lb-0-94614 |
| 0.94607 | 0.94638 | najiama/pure-lgbm-model-cv-0-94607-lb-0-94638 |
| 0.94610 | 0.94636 | mizushimatoshihiko/s6e9-competition-only-lgbm |
| 0.946132 | (n/a) | jazivxt/single-model-zoom-zoom (own 5-fold OOF) |
| 0.94618 | 0.94633 | najiama/s6e9-electric-vehicle-oof |
| 0.9463 | 0.94633 | megayak/s6e9-one-lightgbm-from-raw-data |

**Read this table carefully.** CV 0.946070 → LB 0.94638, and CV 0.946072 → LB 0.94614.
Two essentially identical CVs, **0.00024 apart on the LB**. That is a live empirical
demonstration of the paired-SE argument: public LB differences of this size carry no
information. Higher CV does not monotonically map to higher LB here.

Working rule: **CV 0.9461 ≈ LB 0.9463 ± 0.0002.** Target CV >= 0.9463 for a safe top-350.

---

## THE BEST GENUINE NOTEBOOKS (ranked by what you'd learn)

### 1. jazivxt/single-model-zoom-zoom — LB 0.94647, OOF 0.946132, 54 votes, 1152s
Grandmaster. The anchor everything else parasitises. Read this one first. Full recipe below.

Caveats: its `submission.csv` is 80% external reference + 20% own model, so 0.94647 is NOT
a pure model score — the honest number is the 5-fold OOF 0.946132. Its "pass 2" refolds by
pass-1 OOF prediction deciles, which the author admits is **not independent validation**.
Do not copy that part.

### 2. najiama/pure-lgbm-model-cv-0-94607-lb-0-94638 — 78 votes, 1383s
Cleanest honest CV/LB pair on the board. Good baseline to reproduce.

### 3. najiama/xgboost-triple-te-dynamic-pruning-lb-0-94639 — 21 votes, 2961s
The Triple-TE idea in XGBoost form.

### 4. yekenot/ps-s6-e9-realmlp-pytorch — LB 0.94621, 42 votes, 598s
Your diversity model. Also `yekenot/ps-s6-e9-realmlp-pytabkit` (0.94620, 496s). Neural, so
decorrelated from the GBDTs. Forum reports RealMLP is stronger on class 0 where TabM is
stronger on class 1.

### 5. megayak/s6e9-one-lightgbm-from-raw-data-cv-0-9463 — LB 0.94633, 1551s
Highest clean CV on the board (0.9463). From raw data, no external inputs.

### 6. chinzorigtganbat/s6e9-leak-safe-features-logit-blending — LB 0.94639, 2752s
Explicitly leak-safe encoding plus logit-space blending. Worth it for the leakage discipline.

### 7. hermengardo/single-xgb-eda-ps6e9 — 0.94627, 56 votes, 1104s
Best EDA-plus-model combination.

### 8. maiernator/s6e9-ctboost-not-catboost-astra-baseline — 0.94612 in only 217s
Best score-per-second on the board. Good for fast iteration loops.

### 9. cdeotte/fable-5-1-xgb-starter — the educational starter (base_margin / DGP recipe)

### 10. georgymamarin/s6e9-what-the-board-paid-for-eleven-submissions — 62 votes, 4985s
An LB-noise accounting study. Directly relevant to the shakeup question.

---

## THE ZOOM-ZOOM FEATURE RECIPE (111 features, LightGBM)

13 raw columns. Categoricals:
`Gender, City_Type, Current_Car_Type, Home_Charging_Possible, Subsidy_Available, Range_Anxiety_Level`

### A. Row-local, target-free

```
inc_d1 = income % 10          inc_d2 = income // 10 % 10      inc_d3 = income // 100 % 10
inc_mod100 = income % 100     inc_mod1000 = income % 1000
km10 = round(Daily_Commute_km * 10)
km_d1 = km10 % 10             km_mod100 = km10 % 100
inc_q{50,100,250,500,1000,2500,5000} = income // divisor    # quantisation ladder
km_q{5,10,25,50}                     = km10   // divisor
is_30k_spike         = (income == 30000)
is_millionaire_cliff = (income >= 170537)
is_dead_zone         = (38000 <= income <= 42000)
is_env_hater         = (Environmental_Concern_Level == 1)
```

The final model DROPS `inc_d1` — 111 features vs the 112-feature control: 0.946132 vs
0.946124, four folds improved and one declined.

### B. Encoding keys

`k_inc_exact, k_inc100, k_inc1000, k_km_int` plus each categorical and `Age`,
`Number_of_Cars_Owned`, `Charging_Stations_Near_Home`, `Charging_Stations_Near_Work`,
`Environmental_Concern_Level`.

### C. Fold-local encodings (computed INSIDE the training fold only)

- `fq_inc`, `fq_km` — raw value_counts of income and commute x10
- `{key}_fe` — normalised frequency encoding per key
- `{key}_te{auto,10,100}` — **this is the "Triple TE"**:
  `sklearn.preprocessing.TargetEncoder(shuffle=True, cv=5, smooth=s)` for s in
  {`auto`, `10.0`, `100.0`}. Three smoothing levels of the same encoding, all three fed
  to the model.

### D. The actual innovation — cross-fitted asymmetric income statistics

Bin income into **q = 8192 and q = 16384 equal-width bins**. Per bin, compute smoothed
target statistics, cross-fitted over an inner 5-fold:

```
central    = (sums + smooth*prior) / (counts + smooth)
left/right = the same quantity on the adjacent bins
symmetric  = Gaussian-kernel (sigma 0.8) convolution over neighbouring bins
slope      = right - left
curvature  = central - 0.5*(left + right)
log1p(counts)
+ bin position / n_bins
```

8 columns per resolution, 15 income-stat features in total.

**This is the real resolution lever** — not LightGBM's `max_bin`, which stays at the
default 255. Correction to the conventional advice: the high-resolution work happens in
the target-statistic features, not in the histogram binning.

### E. Model parameters

```python
LGBMClassifier(n_estimators=3500, learning_rate=0.02, max_depth=5, num_leaves=32,
               min_child_samples=10, subsample=0.8, subsample_freq=1,
               colsample_bytree=0.3, reg_alpha=0.071, reg_lambda=2.0,
               max_bin=255, feature_pre_filter=False)   # early_stopping(100)

XGBClassifier(n_estimators=2400, max_depth=6, learning_rate=0.03, min_child_weight=12,
              subsample=0.82, colsample_bytree=0.55, reg_alpha=0.08, reg_lambda=3,
              max_bin=512, tree_method="hist", enable_categorical=True,
              early_stopping_rounds=120)

CatBoostClassifier(iterations=600, depth=6, learning_rate=0.07, l2_leaf_reg=5, rsm=0.8,
                   random_strength=0.35, bootstrap_type="Bayesian",
                   bagging_temperature=0.45, od_type="Iter", od_wait=140)
```

Note `colsample_bytree=0.3` on LightGBM — aggressive, because many of the 111 features are
near-duplicate encodings of the same few columns.

---

## KNOWN DATASET STRUCTURE (from forum and code)

- DGP recipe: `score = 1.2*(income/100k) + 0.6*concern + 2*subsidy - 1*med_anxiety - 3*high_anxiety + noise`, buy if `score > 5.5`
- `income >= 170537` → always buys
- No subsidy → under 1% buy rate
- Income spike at exactly 30000; dead zone 38000–42000
- Three cells with **zero** buyers in training (3,575 test rows): income 31,004–41,970;
  commute >= 83 km; income 30000 without subsidy and with low concern or medium/high anxiety
- External data allowed: `itzzomkar/ev-adoption-behavior-and-range-anxiety` (the ~10k real rows)
