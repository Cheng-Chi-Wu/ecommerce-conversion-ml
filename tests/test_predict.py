import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src.predict import score_features


def bundle():
    model = DummyClassifier(strategy='prior').fit(pd.DataFrame({'count': [0, 1]}), [0, 1])
    return {'features': ['count'], 'model': model, 'threshold': .5,
            'metadata': {'model_version': 'test-v1'}}


def test_batch_contract_and_empty_input():
    result = score_features(bundle(), pd.DataFrame({'visitorid': [3, 4], 'count': [0, 2]}))
    assert list(result.visitorid) == [3, 4]
    assert list(result.score) == [.5, .5]
    assert list(result.model_version) == ['test-v1', 'test-v1']
    assert score_features(bundle(), pd.DataFrame({'count': pd.Series(dtype=float)})).empty


@pytest.mark.parametrize('frame', [pd.DataFrame({'other': [1]}),
                                  pd.DataFrame({'count': [np.nan]}),
                                  pd.DataFrame({'count': [-1]}),
                                  pd.DataFrame({'count': ['invalid']})])
def test_batch_contract_rejects_invalid_features(frame):
    with pytest.raises(ValueError):
        score_features(bundle(), frame)
