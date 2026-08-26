"""Exit 1 if model.json was trained on a different feature set than FEATURES.

The gameday path deliberately skips training — nothing it reads changes on a
Sunday morning — but that makes it silently dependent on a model file built
from the same feature list. When the two drift, XGBoost raises a feature_names
mismatch deep inside export_web.py and the run dies after the fetches have
already happened. Checking first turns that into a one-line decision to retrain.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features import FEATURES  # noqa: E402


def main() -> None:
    path = "model.json"
    if not os.path.exists(path):
        print("model.json missing")
        sys.exit(1)
    try:
        from xgboost import XGBClassifier

        m = XGBClassifier()
        m.load_model(path)
        saved = list(m.get_booster().feature_names or [])
    except Exception as e:
        print(f"could not read {path}: {e}")
        sys.exit(1)
    if saved != list(FEATURES):
        missing = [f for f in FEATURES if f not in saved]
        extra = [f for f in saved if f not in FEATURES]
        print(f"feature drift — missing from model: {missing or 'none'}; "
              f"no longer in FEATURES: {extra or 'none'}")
        sys.exit(1)
    print(f"model.json matches FEATURES ({len(saved)} features)")


if __name__ == "__main__":
    main()
