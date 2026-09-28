#train catboost model on the same 9/10 fold logic as the previous models
import numpy
import pandas
from catboost import CatBoostClassifier, Pool
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from src import paths
from src.regularise import ID_COL, TARGET_COL, FOLD_COL, loadLib

#model sonstants

MODEL_NAME = "catboost"

#keep false for now
QICK_CHK = True

USE_GPU = False
SEED=42

TRAIN_SET = pandas.read_csv(paths.RAW_DIR / "train.csv")
TEST_SET = pandas .read_csv(paths.RAW_DIR / "test.csv")

CLASSIFICATION = (TRAIN_SET[TARGET_COL] == "Yes").astype(int).to_numpy()

#Match 6-veiw folds => catboost fold, ensureing 6v == catboost
fold_sorce = pandas.read_csv(paths.SIX_VIEWS_DIR / "oof_six_views.csv", usecols=[ID_COL, FOLD_COL])
TRAIN_SET = TRAIN_SET.merge(fold_sorce, on=ID_COL, how="left", validate="one_to_one")
if TRAIN_SET[FOLD_COL].isna().any():
    raise ValueError("col mismatch")
fold = TRAIN_SET[FOLD_COL].to_numpy(dtype=int)
fold_numbers = sorted(set(fold)) #[0]->[9]


#build coherent table to train model
#first sort data into int and string

text_cols=[]
int_cols=[]
for col in TRAIN_SET.columns:
    #exclude vital/used components
    if col == ID_COL or col == TARGET_COL or col == FOLD_COL:
        continue
    if pandas.api.types.is_numeric_dtype(TRAIN_SET[col]):
        int_cols.append(col)
    else:
        text_cols.append(col)

features = text_cols + int_cols
X = TRAIN_SET[features].copy()
X_test = TEST_SET[features].copy()
cat_cols = list(text_cols)


#add int=> float for DATA uniformity

for col in int_cols:
    new_col = col + "_exact"
    X[new_col] = TRAIN_SET[col].astype(str)
    X_test[new_col] = TEST_SET[col].astype(str)
    cat_cols.append(new_col)


#setup folds
oof_pred = numpy.zeros(len(TRAIN_SET))
test_pred = numpy.zeros(len(TEST_SET))
#bundle features with bin in cotegoracle
test_pool = Pool(X_test, cat_features=cat_cols)

#TRAIN MODEL => train on 9, test on one fold

if QICK_CHK:
    fold_to_run = [0]
else:
    fold_to_run = fold_numbers

foldAUC=[]
for fold_num in fold_to_run:
    is_valid = (fold == fold_num)
    is_train = ~is_valid

    train_pool = Pool(X[is_train], CLASSIFICATION[is_train], cat_features=cat_cols)
    valid_pool = Pool(X[is_valid], CLASSIFICATION[is_valid], cat_features=cat_cols)

    if USE_GPU:
        ask_type = "GPU"
    else:
        ask_type="CPU"

    CATBOOST = CatBoostClassifier(
        iterations=4000,            
        learning_rate=0.08,        
        depth=6,                    
        l2_leaf_reg=5,              
        eval_metric="AUC",          
        early_stopping_rounds=200,  
        random_seed=SEED,
        task_type=ask_type,
        thread_count=-1,            
        verbose=250,                
    )

    CATBOOST.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    #define prediction to be the Will_Buy_EV
    oof_pred[is_valid] = CATBOOST.predict_proba(valid_pool)[:,1]
    test_pred = test_pred + CATBOOST.predict_proba(test_pool)[:, 1]

    auc = roc_auc_score(CLASSIFICATION[is_valid], oof_pred[is_valid])
    foldAUC.append(auc)

    print(f"FOLD {fold_num}: AUC {auc:.6f}")  
#prediction are given on the full set, ,ust average them:
test_pred = test_pred / len(fold_to_run)
if QICK_CHK:
    print(f"Quick check done: fold 0 AUC {foldAUC[0]:.6f}")
    raise SystemExit

#save OOF and test predictions
paths.CATBOOST_MODED_DIR.mkdir(parents=True, exist_ok=True)
oof_frame = pandas.DataFrame({ID_COL: TRAIN_SET[ID_COL], MODEL_NAME: oof_pred})
test_frame = pandas.DataFrame({ID_COL: TEST_SET[ID_COL], MODEL_NAME: test_pred})
oof_frame.to_csv(paths.CATBOOST_MODED_DIR / f"{MODEL_NAME}_oof.csv", index=False)
test_frame.to_csv(paths.CATBOOST_MODED_DIR / f"{MODEL_NAME}_test.csv", index=False)

overallAUC = roc_auc_score(CLASSIFICATION, oof_pred)
print(f"Results saved to {paths.CATBOOST_MODED_DIR}")
print(f"{MODEL_NAME} OOF AUC: {overallAUC:.6f}") 
#compare new results to old results
library = loadLib()

oof_table = library["oof_table"]

#ensure that the rows will allighn
if not (library["train_ids"] == TRAIN_SET[ID_COL].to_numpy()).all():
    raise ValueError("Row order differs between train.csv and the OOF library.")

#define prevous 17 

public_names=[]
for name in oof_table.columns:
    if name != MODEL_NAME:
        public_names.append(name)

publicsum = numpy.zeros(len(TRAIN_SET))
for name in public_names:
    publicsum += oof_table[name].to_numpy()

#define 17 and 18 blends
blend_17 = publicsum / len(public_names)

cat_ranks = rankdata(oof_pred) / len(oof_pred)

BlendEightTeen = (publicsum + cat_ranks) / (len(public_names) + 1)

correation = numpy.corrcoef(cat_ranks, blend_17)[0, 1]

folds_won = 0
print(f"Correlation with the 17-model blend: {correation:.5f}")
print("fold | blend of 17 | blend of 18 | change")
for fold_num in fold_numbers:
    in_fold = (fold == fold_num)
    auc_17 = roc_auc_score(CLASSIFICATION[in_fold], blend_17[in_fold])
    auc_18 = roc_auc_score(CLASSIFICATION[in_fold], BlendEightTeen[in_fold])
    if auc_18 > auc_17:
        folds_won = folds_won + 1
    print(f"  {fold_num}  |  {auc_17:.6f}   |  {auc_18:.6f}   | {auc_18 - auc_17:+.7f}")

total_17 = roc_auc_score(CLASSIFICATION, blend_17)
total_18 = roc_auc_score(CLASSIFICATION, BlendEightTeen)
print(f"Overall: {total_17:.7f} -> {total_18:.7f} ({total_18 - total_17:+.7f}), won {folds_won}/10 folds")