import copy
from evalml.pipelines import DecisionTreeClassifier
from evalml.pipelines.components.transformers import Transformer
from evalml.pipelines.components.estimators import Estimator
from evalml.objectives import F1, Recall, Precision, AUC, AccuracyBinary, AccuracyMulticlass
from evalml.model_understanding import confusion_matrix

class KeystoneXL:
    """Your code here!"""

    def __init__(self, components, parameters, random_state):
        self.parameters = copy.deepcopy(parameters)
        self.random_state = random_state
        self.components = []

        for component in components:
            if isinstance(component(), Estimator):
                self.parameters[component.name]["random_state"] = self.random_state
            self.components.append(component(**self.parameters[component.name]))

    def fit(self, X_train, y_train):
        out = X_train
        for component in self.components:
            component.fit(out, y_train)
            if isinstance(component, Transformer):
                out = component.transform(out)
            if isinstance(component, Estimator):
                return component.predict(out)

    def predict(self, X_test):
        """Function to run X_test through pipeline and evaluate predictions
        against requrested metrics.

        Arguments:
            X_test (np.ndarray): array of the test input data to predict on
            y_test (np.ndarray): array of the test output data to score against
        Returns:
            y_pred (np.ndarray):

        """

        out = X_test
        for component in self.components:
            if isinstance(component, Transformer):
                out = component.transform(out)
            if isinstance(component, Estimator):
                out = component.predict(out)
        return out

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







