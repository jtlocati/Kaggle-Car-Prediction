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
QICK_CHK = False

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
    if type(TRAIN_SET[col]) is int or type(TRAIN_SET[col]) is float:
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