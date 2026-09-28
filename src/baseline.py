#takes the 17 models, trains on a 10 set fold, prodused prediction
import numpy
import pandas
from sklearn.metrics import roc_auc_score
from src import paths
from src.regularise import ID_COL, TARGET_COL, loadLib



#Load all models ranked predictions
print(f"laoding OOF lib")
library=loadLib()
y = library["y"]
fold = library["fold"]
oof_table = library["oof_table"]
test_table = library["test_table"]
model_names = list(oof_table.columns)

fold_ints = sorted(set(fold))

#Takes important info gained from parseing, rund them, returns AUC result
def ScoreByFold(predictions, y, fold, fold_ints):
    overallAUC=roc_auc_score(y, predictions)

    fold_aucs = []
    for fold_number in fold_ints:
        in_fold = (fold == fold_number)
        auc = roc_auc_score(y[in_fold], predictions[in_fold])
        fold_aucs.append(auc)
    return overallAUC, fold_aucs


report_rows = []
for model in model_names:
    predictions = oof_table[model].to_numpy()
    overallAUC, foldaucs = ScoreByFold(predictions, y, fold, fold_ints)

    row = {"model": model, "oof_auc": overallAUC}
    for fold_num, auc in zip(fold_ints, foldaucs):
        row[f"fold{fold_num}"] = auc

    report_rows.append(row)

model_report = pandas.DataFrame(report_rows).sort_values("oof_auc", ascending=False)


print("\nSingle-model OOF AUC:")
for _, row in model_report.iterrows():
    print(f"{row['oof_auc']:.6f} {row['model']}")


#begin weighting models
oof_sum = numpy.zeros(len(y))
test_sum  = numpy.zeros(len(test_table))
for model in model_names:
    oof_sum += oof_table[model].to_numpy()
    test_sum += test_table[model].to_numpy()

model_count = len(model_names)
#blends pred for all rows 
eqal_oof = oof_sum / model_count
equal_test = test_sum / model_count

equal_auc, equal_fold_auc = ScoreByFold(eqal_oof, y, fold, fold_ints)
best_single_auc = model_report["oof_auc"].iloc[0]
best_single_name = model_report["model"].iloc[0]
print(f"\nEqual blend of {model_count} models: OOF AUC {equal_auc:.7f}")
print(f"Best single model ({best_single_name}):  {best_single_auc:.7f}")
print(f"Blend minus best single: {equal_auc - best_single_auc:+.7f}")
print("Blend AUC per fold:")
for fold_number, auc in zip(fold_ints, equal_fold_auc):
    print(f"fold {fold_number}: {auc:.6f}")

#save to outputs
submission = pandas.DataFrame({ID_COL: library["test_ids"], TARGET_COL: equal_test})
submission.to_csv(paths.OUTPUTS_DIR / "submission_equal.csv", index=False)

# The per-model table, so you can open it in Excel later.
model_report.to_csv(paths.OUTPUTS_DIR / "stage1_model_report.csv", index=False)

baseline_rows = []
for fold_number, auc in zip(fold_ints, equal_fold_auc):
    baseline_rows.append({"fold": fold_number, "equal_blend_auc": auc})
pandas.DataFrame(baseline_rows).to_csv(paths.OUTPUTS_DIR / "stage1_baseline_folds.csv", index=False)

print(f"\nWrote submission_equal.csv, stage1_model_report.csv, stage1_baseline_folds.csv")
