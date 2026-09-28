## Super Sick Model Ensemble for Kaggle Competiton
Current Kaggle standing: 460 / 3262 (top 14.1%)
Public AUC: 0.94644 (OOF AUC: 0.946385)

## Model Ensemble
Blending 17 public models from two OOF libraries, plus 1 custom model.

### najiama library (8 models)
- 01 Blend (a pre-made blend)
- LightGBM V1
- LightGBM V3
- LightGBM V5
- LightGBM V6
- Sergey LightGBM
- XGBoost, triple target encoding (5 folds)
- XGBoost, triple target encoding (10 folds)

### megayak six views (9 models)
- A: LightGBM, triple target encoding + digit features
- B: XGBoost on A's features
- C: LightGBM, value windows, no digits
- D: LightGBM, income ladders, no exact income
- E: LightGBM, different ladders
- F: LightGBM, exact income rate as starting score
- Ensemble (author's mix of A to F)
- RealMLP neural net (1 seed)
- RealMLP neural net (3 seeds)

### Custom (1 model)
- CatBoost (trained on the same 10 folds)