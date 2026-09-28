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
            avalibe_model_belnd = (running_sum + matrix[:, i] / (total_picks +1))
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