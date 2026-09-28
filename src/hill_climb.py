#Hill climb for finding model weighting
import numpy 
import pandas
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from src import paths
from src.regularise import COUS_MODELS, ID_COL, loadLib, TARGET_COL

#Constants for Climb
#disasllow a model to be selected more than MAX_STEPS times
MAX_STEPS = 30

#cut selection ny noise
MIN_GAIN = 0.0000005

#ensure that model is not selected by faulty fold
FOLDS_TO_WIN = 7


#load info
library = loadLib()
predictions = library["y"]
fold = library["fold"]
oof_table = library["oof_table"]
test_table = library["test_table"]
model_names = list(oof_table.columns)
fold_nums = sorted(set(fold)) #[0]->9

#regularise models so that col=models && rows=ppl

oof_RegTable = oof_table[model_names].to_numpy()
test_RegTable = test_table[model_names].to_numpy()
print(f"models regularised")

#load prevous baseline from satge 1
Satge1 = numpy.zeros(len(model_names))
public_count = 0
for i, name in enumerate(model_names):
    if name not in COUS_MODELS:
        Satge1[i] = 1.0
        public_count += 1
#ensure weighting = 1, each model = 0.06 weighting
Stage1 = Satge1/public_count

#HILL CLIMB
#Takes the matrix of models anked by prediction and the targets => an array of applyed weights for each model
#blends best model per round and returns best model weighting ratio
def HillClimb(matrix, target, names):
    model_count = matrix.shape[1]

    #begin with single-best model
    best_start=0
    #ensure auc will fir to first model
    best_start_auc = -1.0

    for i in range(model_count):
        auc_step = roc_auc_score(target, matrix[:, i])
        if auc_step > best_start_auc:
            best_start_auc = auc_step
            best_start = i

    # Define Climbing point
    pick_counts = numpy.zeros(model_count, dtype=int)
    pick_counts[best_start] = 1
    running_sum = matrix[:, best_start].copy()
    current_auc = best_start_auc
    print(f"start: {names[best_start]} -> {current_auc:.7f}")

    for step in range(MAX_STEPS):
        total_picks = pick_counts.sum()
        best_mdel = -1
        best_model_auc = current_auc

        #try adding each model avalibe (17), then evaluate
        for i in range(model_count):
            avalibe_model_belnd = (running_sum + matrix[:, i]) / (total_picks + 1)
            candidate_auc = roc_auc_score(target, avalibe_model_belnd)
            if candidate_auc > best_model_auc:
                best_model_auc = candidate_auc
                best_mdel = i

        gain = (best_model_auc - current_auc)
        if  best_mdel == -1 or gain < MIN_GAIN:
            print(f"Stopping after {step}, beast ain < {MIN_GAIN}")
            break

        #write the winning blend to store
        #ensures model setup that is best gets weightd prperlt with the amount of occurences
        pick_counts[best_mdel] = pick_counts[best_mdel] + 1 #adjust weights 
        running_sum += matrix[:, best_mdel]
        current_auc = best_model_auc
        print(f"step {step + 1}: + {names[best_mdel]} -> {current_auc:.7f} ({gain:+.7f})")

    #ensures model setup that is best gets weightd prperlt with the amount of occurences = 1
    weights = pick_counts / pick_counts.sum()
    return weights

#nested check => leans weights on 10 folds - f, score on fold f

nested_rows = []
for fold_number in tqdm(fold_nums, desc="Nested folds", unit="fold"):
    learn_rows = (fold != fold_number)
    test_rows = (fold == fold_number)

    fold_weights = HillClimb(oof_RegTable[learn_rows], predictions[learn_rows], model_names)

    #take the 18 predictions, multipy by weughts, everage, then return
    hillPrediction = oof_RegTable[test_rows] @ fold_weights
    equalPrediction = oof_RegTable[test_rows] @ Stage1
    HillAUC = roc_auc_score(predictions[test_rows], hillPrediction)
    equalAUC = roc_auc_score(predictions[test_rows], equalPrediction)

    nested_rows.append({"fold": fold_number, "equal_auc": equalAUC, "hill_auc": HillAUC, "change": HillAUC - equalAUC})

nested = pandas.DataFrame(nested_rows)
folds_won = int((nested["change"] > 0).sum())
mean_change = nested["change"].mean()
print("fold |   equal   |   hill    | change")
for _, row in nested.iterrows():
    print(f"  {int(row['fold'])}  | {row['equal_auc']:.6f} | {row['hill_auc']:.6f} | {row['change']:+.7f}")
print(f"Hill climbing won {folds_won}/10 folds, mean change {mean_change:+.7f}")

#take decided weights and run them once on each row so we can use for the test set 

print(f"final hillclimb for all models: ")
final_weights = HillClimb(oof_RegTable, predictions, model_names)

print(f"final weights")
weightsFIN = []
for name, weight in zip(model_names, final_weights):
    weightsFIN.append({"model": name, "weight": weight})
weightTable = pandas.DataFrame(weightsFIN).sort_values("weight", ascending=False)
for _, row in weightTable.iterrows():
    if row["weight"] > 0:
        print(f"{row['weight']:.3f}  {row['model']}")


equal_oof_auc = roc_auc_score(predictions, oof_RegTable @ Stage1)
hill_oof_auc = roc_auc_score(predictions, oof_RegTable @ final_weights)
print(f"\nOOF AUC on all rows: equal {equal_oof_auc:.7f} | hill {hill_oof_auc:.7f}")
print("(this hill number is optimistic; trust the nested result above)")


#build submission and pick one
hill_test = test_RegTable @ final_weights   # each test row: weighted sum of the models' ranks

paths.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
pandas.DataFrame({ID_COL: library["test_ids"], TARGET_COL: hill_test}).to_csv(paths.OUTPUTS_DIR / "submission_hill.csv", index=False)
weightTable.to_csv(paths.OUTPUTS_DIR / "stage3_weights.csv", index=False)
nested.to_csv(paths.OUTPUTS_DIR / "stage3_nested.csv", index=False)

if folds_won >= FOLDS_TO_WIN and mean_change > 0:
    print(f"\nVERDICT: hill climbing passed ({folds_won}/10 folds). Submit outputs/submission_hill.csv")
else:
    print(f"\nVERDICT: hill climbing did NOT pass ({folds_won}/10 folds). Keep the Stage 1 equal blend.")
