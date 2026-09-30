#Current OOF files are misalighned, me must linerise them so our models real OOF outputs are real
import numpy
import pandas
from scipy.stats import rankdata

from src import paths

ID_COL = "id"
TARGET_COL = "Will_Buy_EV"
FOLD_COL = "fold"

#Najiama is in two parts: train & test: must combine
NAJIAMA_PAIRS = {
    "01_blend": ("01_blend_oof.csv", "01_submission.csv"),
    "lgbm_v1": ("Pure LGBM_V1_oof.csv", "Pure LGBM_V1_test.csv"),
    "lgbm_v3": ("Pure LGBM_V3_oof.csv", "Pure LGBM_V3_test.csv"),
    "lgbm_v5": ("Pure LGBM_V5_oof.csv", "Pure LGBM_V5_test.csv"),
    "lgbm_v6": ("Pure LGBM_V6_oof.csv", "Pure LGBM_V6_test.csv"),
    "sergey_lgbm": ("Sergey_LGBM_oof.csv", "Sergey_LGBM_submission.csv"),
    "xgb_te_10f": ("XGBoost_Triple_TE_10folds_oof.csv", "XGBoost_Triple_TE_10folds_test.csv"),
    "xgb_te_5f": ("XGBoost_Triple_TE_5folds_oof.csv", "XGBoost_Triple_TE_5folds_test.csv"),
}

#unpack the megayak oof
SIX_VIEWS_TABLES = {
    ("oof_six_views.csv", "test_six_views.csv"): [
        "A_lgbm_triple_te_digits_3seed",
        "B_xgb_on_A_features",
        "C_no_digits_windows_lift_sm2_30_300",
        "D_no_exact_key_ladder_windows",
        "E_ladder25_250_2500_lift_sm5_50_500",
        "F_exact_rate_as_init_score",
        "ensemble",
    ],
    ("oof_realmlp_g.csv", "test_realmlp_g.csv"): [
        "G_realmlp_10fold",
        "G_realmlp_3seed",
    ],
}

COUS_MODELS = [ "catboost" ]

#stage 4: two public models that see the data differently from the other 17
#name -> (oof file, oof col, test file, test col), paths inside data/external
EXTRA_MODELS = {
    "blamerx": ("blamerx/oof.csv", "pred", "blamerx/submission.csv", "Will_Buy_EV"),
    "heuljax": ("heuljax/oof.csv", "oof_pred", "heuljax/test.csv", "test_pred"),
}


#regularise each models prediction to be the rank / num_rows, this allows us to properly evaluate then rank our models
def rankTo01(values):
    #establish the two vars then return
    ranks = rankdata(values, method="average")
    row_count = len(values)
    return (ranks/row_count)


def allignIDS(frame, wanted_ID, col, source_name):
    #frame = one CSV reading
    #ID_CAT = Targets we want (test vs train)
    #col = what prediction col we pull from

    #reorder so all passengers are alligned
    lookup = pandas.DataFrame({ID_COL: wanted_ID})

    merged = lookup.merge(frame[[ID_COL, col]], on=ID_COL, how="left", validate="one_to_one")

    properValues = merged[col].to_numpy(dtype=float)

    #catch a missing value and convert to NaN, rather than let the progam AS
    if numpy.isnan(properValues).any():
        missing_count = int(numpy.isnan(properValues).sum())
        raise ValueError(f"{source_name} has {missing_count} missing IDs for col {col}")
    return properValues

#takes a table with id and prediction => the col name 
def SlimToPredictions(frame, source):
    other_cols = []
    for i in frame.columns:
        if i != ID_COL:
            other_cols.append(i)

    if len(other_cols) != 1:
        raise ValueError(f"{source}:  expected id + 1 column, got {list(frame.columns)}")
    return other_cols[0]

#return a coherent table with all the info that we need
def loadLib():
    #extract folds from megayak
    sixViews = pandas.read_csv(paths.SIX_VIEWS_DIR / "oof_six_views.csv")
    train_ids = sixViews[ID_COL].to_numpy()
    predictions = sixViews[TARGET_COL].to_numpy(dtype=int)
    fold = sixViews[FOLD_COL].to_numpy(dtype=int)

    #draw from kaggles sample submission so we can format properly
    sample_sub = pandas.read_csv(paths.RAW_DIR / "sample_submission.csv")
    test_ids = sample_sub[ID_COL].to_numpy()

    #cread empty dictionarys to hold the procution of the two drawn librarys
    #{model_name: array of rnaked }
    oof_by_model = {}
    test_by_model = {}

    #regularise the najiagma models
    for model, file in  NAJIAMA_PAIRS.items():
        oof_file = file[0]
        test_file = file[1]

        oof_table = pandas.read_csv(paths.NAJIAMA_DIR /oof_file)
        test_table = pandas.read_csv(paths.NAJIAMA_DIR / test_file)

        oof_cols = SlimToPredictions(oof_table, oof_file)
        test_cols = SlimToPredictions(test_table, test_file)

        raw_oof = allignIDS(oof_table, train_ids, oof_cols, oof_file)
        raw_test = allignIDS(test_table, test_ids, test_cols, test_file)

        #store ranked and regularised information
        oof_by_model[model] = rankTo01(raw_oof)
        test_by_model[model] = rankTo01(raw_test)

    #regularise six view tables
    for file, col_name in SIX_VIEWS_TABLES.items():
        oof_file = file[0]
        test_file = file[1]

        oof_frame = pandas.read_csv(paths.SIX_VIEWS_DIR / oof_file)
        test_frame = pandas.read_csv(paths.SIX_VIEWS_DIR / test_file)

        for col in col_name:
            raw_oof = allignIDS(oof_frame, train_ids, col, oof_file)
            raw_test = allignIDS(test_frame, test_ids, col, test_file)
            oof_by_model[col] = rankTo01(raw_oof)
            test_by_model[col] = rankTo01(raw_test)
            print(f"loaded col: {col}")
    #refularise file format + scoring metrics for catboost

    for model in COUS_MODELS:
        oof_path = paths.CATBOOST_MODED_DIR / f"{model}_oof.csv"
        test_path = paths.CATBOOST_MODED_DIR / f"{model}_test.csv"

        if oof_path.exists() == False or test_path.exists() == False:
            print(f"SKIPPED: {model}: NOT trained")
            continue

        oof_frame = pandas.read_csv(oof_path)
        test_frame = pandas.read_csv(test_path)

        preds_oof = SlimToPredictions(oof_frame, oof_path.name)
        pred_test = SlimToPredictions(test_frame, test_path.name)

        raw_oof = allignIDS(oof_frame, train_ids, preds_oof, oof_path.name)
        raw_test = allignIDS(test_frame, test_ids, pred_test, test_path.name)

        oof_by_model[model] = rankTo01(raw_oof)
        test_by_model[model] = rankTo01(raw_test)
        print(f"loaded {model}")
    

    #stage 4: load the extra public models, same allign + rank as every other model
    for model, files in EXTRA_MODELS.items():
        oof_file = files[0]   #train row predictions
        oof_col = files[1]    #col that holds them
        test_file = files[2]  #test row predictions
        test_col = files[3]   #col that holds them

        oof_frame = pandas.read_csv(paths.EXTERNAL_DIR / oof_file)
        test_frame = pandas.read_csv(paths.EXTERNAL_DIR / test_file)

        raw_oof = allignIDS(oof_frame, train_ids, oof_col, oof_file)
        raw_test = allignIDS(test_frame, test_ids, test_col, test_file)

        oof_by_model[model] = rankTo01(raw_oof)
        test_by_model[model] = rankTo01(raw_test)
        print(f"loaded extra: {model}")

    oof_table = pandas.DataFrame(oof_by_model)
    test_table = pandas.DataFrame(test_by_model)

    #return coherent list
    return {"y": predictions, "fold": fold, "train_ids": train_ids, "test_ids": test_ids, "oof_table": oof_table, "test_table": test_table}