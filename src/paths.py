#file to find all csv nessicitys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#Competition CSVs
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

#outside OOF libs
EXTERNAL_DIR = Path(__file__).resolve().parent.parent / "data" / "external"
NAJIAMA_DIR = EXTERNAL_DIR / "s6e9-oof"
SIX_VIEWS_DIR = EXTERNAL_DIR / "six-views"

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"

#extend path to coustom model(s)
CATBOOST_MODED_DIR = EXTERNAL_DIR / "catboost"


