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

