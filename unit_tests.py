import numpy as np
import pytest
from evalml.objectives import F1, Recall, Precision, AUC, AccuracyBinary, AccuracyMulticlass, R2, MSE
from evalml.pipelines.components import Imputer, LogisticRegressionClassifier, OneHotEncoder
from evalml.pipelines.components.estimators import Estimator
from evalml.pipelines.components.transformers import Transformer
from evalml.preprocessing import split_data
from sklearn import datasets

from pipeline import KeystoneXL


@pytest.fixture
def X_y_binary():
    X, y = datasets.make_classification(n_samples=100, n_features=20,
                                        n_informative=2, n_redundant=2,
                                        random_state=0)
    return X, y


@pytest.fixture
def get_test_params():
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
    random_state = 42
    components = [Imputer, OneHotEncoder, LogisticRegressionClassifier]  # noqa: F841
    metrics = [F1, AUC, Recall, Precision, AUC, AccuracyBinary, AccuracyMulticlass, R2, MSE]  # noqa: F841
    return parameters, components, metrics, random_state


def test_pipeline_creation(X_y_binary, get_test_params):
    parameters, components, metrics, random_state = get_test_params
    pipeline = KeystoneXL(parameters=parameters,  # noqa: F841
                          components=components,
                          random_state=random_state)

    # Make sure all components are initialized and with proper parameters
    for idx, p_comp in enumerate(pipeline.components):
        assert isinstance(p_comp,
                          components[idx]), "Pipeline: components are not initialized in same order as presented."
        if isinstance(p_comp, Transformer):
            assert p_comp.parameters.items() >= parameters[
                p_comp.name].items(), "Pipeline: component parameters not set correctly."
        elif isinstance(p_comp, Estimator):
            correct_params = parameters[p_comp.name]
            set_params = p_comp.parameters
            set_params.pop('random_state', None)
            assert set_params.items() >= correct_params.items(), "Pipeline: component parameters not set correctly."

    # Make sure the random_state made it into the estimator
    assert pipeline.random_state == random_state
    assert pipeline.components[-1]._component_obj.get_params()["random_state"] == random_state


def test_pipeline_fit(X_y_binary, get_test_params):
    parameters, components, metrics, random_state = get_test_params
    X, y = X_y_binary
    X_train, X_test, y_train, y_test = split_data(X, y, random_state=random_state)
    pipeline = KeystoneXL(parameters=parameters,  # noqa: F841
                          components=components,
                          random_state=random_state)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    np.testing.assert_allclose(y_pred.tolist(), [1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1])


def test_pipeline_predict(X_y_binary, get_test_params):
    X, y = X_y_binary
    X_train, X_test, y_train, y_test = split_data(X, y)
    parameters, components, metrics, random_state = get_test_params
    pipeline = KeystoneXL(parameters=parameters,  # noqa: F841
                          components=components,
                          random_state=random_state)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(sum(y_pred.to_numpy() == y_test.to_numpy()) / len(y_pred))


def test_pipeline_metrics(X_y_binary, get_test_params):
    X, y = X_y_binary
    X_train, X_test, y_train, y_test = split_data(X, y)
    parameters, components, metrics, random_state = get_test_params
    pipeline = KeystoneXL(parameters=parameters,  # noqa: F841
                          components=components,
                          random_state=random_state)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(sum(y_pred.to_numpy() == y_test.to_numpy()) / len(y_pred))

# split_data() docs should have :return: param more explicit about the order of the
# returns
