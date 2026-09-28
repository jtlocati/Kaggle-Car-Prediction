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


