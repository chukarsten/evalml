from evalml.pipelines.components import (Imputer, LogisticRegressionClassifier,
                                         OneHotEncoder)

class DummyPipeline:
    """Your code here!"""

    def __init__(self, components, parameters, random_state):
        self.components = components
        self.parameters = parameters
        self.random_state = random_state

        for component in components:
            if isinstance(component(), Imputer):
                print("IMP:", component)
                self.imputer = component(**self.parameters["Imputer"])
            if isinstance(component(), OneHotEncoder):
                print("OHE:", component)
                self.one_hot_encoder = component(**self.parameters["One Hot Encoder"])
            if isinstance(component(), LogisticRegressionClassifier):
                print("LRC:", component)
                self.log_reg_clf = component(**self.parameters["Logistic Regression Classifier"])

    def fit(self, X_train, y_train):
        if self.imputer:
            imputer = self.imputer
            imputer.fit(X_train, y_train)
            imputer_out = imputer.transform(X_train)
        if self.one_hot_encoder:
            ohe = self.one_hot_encoder
            ohe.fit(imputer_out, y_train)
            ohe_out = ohe.transform(imputer_out, y_train)
        if self.log_reg_clf:
            lrc = self.log_reg_clf
            lrc.fit(ohe_out, y_train)
            return lrc.predict(ohe_out)


    def predict(self):
        pass


