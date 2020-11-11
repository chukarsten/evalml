import copy
import functools

from evalml.pipelines.components.estimators import Estimator
from evalml.pipelines.components.transformers import Transformer


def trackcalls(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.has_been_called = True
        return func(*args, **kwargs)

    wrapper.has_been_called = False
    return wrapper


class KeystoneXL:
    """ML Pipeline Interview Class"""

    def __init__(self, components, parameters, random_state):
        """Custom ML pipeline with components defined by the user.

        Arguments:
            components (list of evalml.components): array of the
            evalml components consisting of a series of Transformers
            followed by an Estimator
        Raises:
            ValueError: if the components list does not end in an instance
            of an Estimator
        Returns:
            None

        """
        self.parameters = copy.deepcopy(parameters)
        self.random_state = random_state
        self.components = []

        for component in components:
            if isinstance(component(), Estimator):
                self.parameters[component.name]["random_state"] = self.random_state
            self.components.append(component(**self.parameters[component.name]))

        if not isinstance(self.components[-1], Estimator):
            raise ValueError("KeystoneXL:init() - Pipeline components must end in Estimator.")

    @trackcalls
    def fit(self, X_train, y_train):
        """Function that fits all components in the pipeline using the training
        data.

        Arguments:
            X_train (np.ndarray): array of the training input data to fit
            y_train (np.ndarray): array of the test output data to score against
        Returns:
            None

        """
        data = X_train
        for component in self.components:
            component.fit(data, y_train)
            if isinstance(component, Transformer):
                data = component.transform(data)

    def predict(self, X_test):
        """Function to run X_test through pipeline and evaluate predictions
        against requrested metrics.

        Arguments:
            X_test (np.ndarray): array of the test input data to predict on
            y_test (np.ndarray): array of the test output data to score against
        Returns:
            y_pred (np.ndarray):

        """

        data = X_test
        for component in self.components:
            if isinstance(component, Transformer):
                data = component.transform(data)
            if isinstance(component, Estimator):
                data = component.predict(data)
        return data

    def metrics(self, y_predicted, y_true, metrics):
        """Function to run requested metrics on outputs.

        Arguments:
            y_predicted (np.ndarray): array of the predicted labels
            y_true (np.ndarray): array of the truth labels
            metrics (list of evalml.objectives): list of metrics to evaluate against
        Returns:
            results (dict): dictionary of requested metrics scores

        """
        results = {}
        for metric in metrics:
            metric_str = metric.__name__
            metric_score = metric().score(y_predicted, y_true)
            print("%s: %f" % (metric_str, metric_score))
            results[metric_str] = metric_score
        return results
