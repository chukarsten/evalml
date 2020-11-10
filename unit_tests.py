import pytest
from evalml.pipelines.components import (Imputer, LogisticRegressionClassifier,
                                         OneHotEncoder)
from sklearn import datasets

from pipeline import DummyPipeline


@pytest.fixture
def X_y_binary():
    X, y = datasets.make_classification(n_samples=100, n_features=20,
                                        n_informative=2, n_redundant=2,
                                        random_state=0)
    return X, y


def test_pipeline_creation(X_y_binary):
    X, y = X_y_binary

    parameters = {
        'Imputer': {
            "categorical_impute_strategy": "most_frequent",
            "numeric_impute_strategy": "mean",
        },
        'One Hot Encoder': {
            'top_n': 10,
            'handle_missing': 'error'
        },
        'Logistic Regression Classifier': {
            'penalty': 'l2',
            'C': 1.0,
        }
    }
    components = [Imputer, OneHotEncoder, LogisticRegressionClassifier]  # noqa: F841
    pipeline = DummyPipeline(parameters=parameters,  # noqa: F841
                             components=components,
                             random_state=42)
    import pdb; pdb.set_trace()
    pipeline.fit(X,y)
