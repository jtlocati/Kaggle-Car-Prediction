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
    lookup = pandas.DataFrame({ID_COL, wanted_ID})

    merged = lookup.merge(frame[[ID_COL, col]], on=ID_COL, how="left", validate="one_to_one")

    properValues = merged[col].to_numpy(dtype=float)

    #catch a missing value and convert to NaN, rather than let the progam AS
    if numpy.isnan(properValues).any():
        missing_count = int(numpy.isnan(properValues).sum())
        raise ValueError(f"{source_name} has {missing_count} missing IDs for col {col}")
    return properValues