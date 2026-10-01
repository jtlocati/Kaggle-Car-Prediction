# Kaggle Playground S6E9: Predicting Electric Vehicle Purchases

A rank-space ensemble of 19 out-of-fold (OOF) models, weighted with a nested hill climb. It finished **270th of 3,576 teams (top 7.6%)** on the private leaderboard, up 245 places from its final public rank. Every blending decision was made on honest cross-validation, never on the public leaderboard, and that held up when the hidden 80% of the test set was revealed.

## Results

| | Rank | Score (ROC AUC) |
|---|---|---|
| **Private leaderboard (final)** | **🥉 270 / 3,576 (top 7.6%)** | **0.94546** |
| Public leaderboard (final) | 515 / 3,576 | 0.94649 |
| Nested CV, final blend | | 0.946462 OOF (10/10 folds over the Stage 3 blend) |
| Winning private score | 1 | 0.94602 |

Progress by stage:

| Stage | What changed | OOF AUC | Public | Nested check |
|---|---|---|---|---|
| 1 | Equal-weight rank blend of 17 public models | 0.946385 | 0.94644 | baseline |
| 2 | Own CatBoost model added as candidate #18 | 0.945538 (alone) | not submitted | got 0 weight |
| 3 | Hill climbing over 18 models | 0.946419 | 0.94645 | 10/10 folds, +0.0000305 vs Stage 1 |
| 4 | Two new public models (heuljax, BlamerX) added | 0.946462 | 0.94649 | 10/10 folds, +0.0000368 vs Stage 3 |
| 5 | Team car (hill climb + 1/4 logistic stack) | | not submitted | failed, 0/10 folds |

Final selected submissions: Stage 4 (`submission_hill_big.csv`) and Stage 3 (`submission_hill.csv`) as the hedge.

## Approach

### The idea

The competition data is synthetic, and the strongest public models were already within a few hundred-thousandths of each other. Instead of training another model of the same kind, this project treats other people's published predictions as a panel of experts and learns how much to trust each one.

Each public model comes as two files:

- **OOF predictions:** a prediction for every training row, made by a model that never saw that row. Because the answers are known, these show honestly how good each model is.
- **Test predictions:** the same model's predictions on the rows Kaggle grades.

The weights are learned on the OOF files and applied to the test files.

### Pipeline

```
data/external/*  ->  src/regularise.py  ->  src/hill_climb.py  ->  outputs/submission_hill_big.csv
(public OOF +       (align every model     (nested hill climb,
 test files)         by id, rank to 0..1)   gated vs the last blend)
```

1. **Align and rank (`src/regularise.py`).** Every model's predictions are merged onto one master id list (so row *i* is the same person in every model) and converted to ranks between 0 and 1. ROC AUC only depends on order, so ranking loses nothing and puts models with different output scales (one public model outputs raw scores from -5.6 to 3.1) on equal footing.
2. **Shared folds.** All scoring uses one fixed 10-fold split (`StratifiedKFold(10, shuffle=True, random_state=42)`, taken from the six-views library), so every model is judged on the same rows.
3. **Hill climbing (`src/hill_climb.py`).** Start from the best single model. Each round, try adding every model once more and keep whichever addition raises AUC the most. A model picked twice out of seven picks gets weight 2/7. Stop when no addition helps by at least 0.0000005.
4. **Nested check.** Weights learned and scored on the same rows always look too good, so for each fold the climber learns weights on the other nine and is scored only on the hidden one. A new blend is accepted only if it beats the previous blend in at least 7 of 10 folds with a positive mean gain.

### Final blend weights

| Model | Source | Weight |
|---|---|---|
| heuljax XGBoost (173 features) | public notebook | 0.286 |
| ensemble (six-views A to F mix) | megayak library | 0.143 |
| 01_blend | najiama library | 0.143 |
| B: XGBoost on view A features | megayak library | 0.143 |
| BlamerX XGBoost + window encodings | public notebook | 0.143 |
| RealMLP neural net (3 seeds) | megayak library | 0.143 |

13 other candidates, including my own CatBoost, received weight 0.

## What worked and what did not

**Worked**

- **Trusting CV over the public board.** Roughly 150 teams sat above this blend on the public leaderboard using notebooks fitted to the public 20%. On the private 80%, this blend moved up 245 places.
- **Models that see the data differently.** The single biggest gain of the project was heuljax (+0.0000786 on its first pick), and the first pick in Stage 3 was RealMLP, the only neural net. Strength alone was not enough; difference from the pack is what earned weight.
- **A hard validation gate.** Every change had to win the nested check before it was used, which made the decisions easy and the results trustworthy.

**Did not work**

- **Own CatBoost model (0.945538 OOF).** It was the most different model in the pool (correlation 0.994 with the blend) but too weak to help at any weight.
- **Team car (hill climb + 1/4 logistic stack).** Helped in a public 83-model notebook, but lost all 10 folds here: with only 20 models, the stack had nothing the hill climb was missing.
- **Extra libraries (medvax, digit-leak GBDTs).** Loaded and aligned correctly, but never picked by the climber.

## Repository layout

```
src/
  paths.py        folder locations, the only place paths are defined
  regularise.py   loads every model's OOF + test predictions, aligns by id, ranks to 0..1
  baseline.py     Stage 1: equal-weight blend and per-model report
  catboost.py     Stage 2: own CatBoost model on the shared 10 folds (GPU optional)
  hill_climb.py   Stages 3 and 4: nested hill climb, gated, writes the submission
data/             competition data and public OOF libraries (not committed)
outputs/          submissions and fold reports (not committed)
```

## How to run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Competition data (join the competition first, then download into `data/raw/`):

```powershell
kaggle competitions download -c playground-series-s6e9 -p data\raw
```

Public OOF sources (into `data/external/`):

```powershell
kaggle datasets download -d najiama/s6e9-oof -p data\external\s6e9-oof --unzip
kaggle datasets download -d megayak/s6e9-six-feature-views-oof-library -p data\external\six-views --unzip
kaggle kernels output blamerx/s6e9-xgboost-window-encodings-0-946-cv -p data\external\blamerx
kaggle kernels output heuljax/kps6e09-xgb-sample -p data\external\heuljax
```

heuljax ships `.parquet` files; they were converted once to `data/external/heuljax/oof.csv` (`id, oof_pred`) and `test.csv` (`id, test_pred`) so no extra package is needed.

Then:

```powershell
python -m src.catboost      # optional: trains the CatBoost candidate (about 30 min on CPU)
python -m src.hill_climb    # about 10 min on CPU, writes outputs/submission_hill_big.csv
```

## Credits

The blend is built on predictions published by other competitors: [najiama](https://www.kaggle.com/datasets/najiama/s6e9-oof), [megayak](https://www.kaggle.com/datasets/megayak/s6e9-six-feature-views-oof-library) (six feature views and RealMLP view G, originally by [yekenot](https://www.kaggle.com/code/yekenot/ps-s6-e9-realmlp-pytorch)), [BlamerX](https://www.kaggle.com/code/blamerx/s6e9-xgboost-window-encodings-0-946-cv) and [heuljax](https://www.kaggle.com/code/heuljax/kps6e09-xgb-sample). The idea of adding these two models came from lucifer19's [EV Grand Prix](https://www.kaggle.com/code/lucifer19/ev-grand-prix-48-engine-cpu-pit-stop-blend) notebook. Hill climbing for ensemble selection follows Caruana et al. (2004).
