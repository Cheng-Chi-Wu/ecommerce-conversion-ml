"""Score precomputed features with the locally generated model bundle."""
import argparse

import joblib
import numpy as np
import pandas as pd

from .common import PROCESSED


def score_features(bundle, frame):
    features = bundle['features']
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise ValueError(f'Missing features: {missing}')
    values = frame[features]
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in values.dtypes):
        raise ValueError('Features must be numeric')
    if not np.isfinite(values.to_numpy()).all() or (values < 0).any().any():
        raise ValueError('Features must be finite and nonnegative')
    result = frame[[column for column in ['visitorid', 'observation_time'] if column in frame]].copy()
    result['score'] = bundle['model'].predict_proba(values)[:, 1] if len(frame) else np.array([])
    result['prediction'] = result.score.ge(bundle['threshold']).astype('int64')
    result['model_version'] = bundle.get('metadata', {}).get('model_version', 'unversioned')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='CSV with the ten documented features')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    bundle = joblib.load(PROCESSED/'selected_model.joblib')
    score_features(bundle, pd.read_csv(args.input)).to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
