"""Machine-learning model training and evaluation for FraudShield.

Functions
---------
train_logistic_regression : Fit a Logistic Regression classifier.
train_decision_tree       : Fit a Decision Tree classifier.
train_mlp                 : Fit a Multi-Layer Perceptron classifier.
train_random_forest       : Fit a Random Forest classifier.
train_ensemble            : Fit a soft-voting Ensemble classifier.
evaluate_model            : Print a classification report and return accuracy.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


def train_logistic_regression(
    X_train: pd.DataFrame,
    Y_train: pd.Series,
    random_state: int = 42,
) -> LogisticRegression:
    """Fit and return a Logistic Regression model.

    Parameters
    ----------
    X_train : array-like
        Training feature matrix.
    Y_train : array-like
        Training labels.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    LogisticRegression
        Fitted model.
    """
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train, Y_train)
    return model


def train_decision_tree(
    X_train: pd.DataFrame,
    Y_train: pd.Series,
    random_state: int = 42,
) -> DecisionTreeClassifier:
    """Fit and return a Decision Tree classifier.

    Parameters
    ----------
    X_train : array-like
        Training feature matrix.
    Y_train : array-like
        Training labels.
    random_state : int
        Random seed.

    Returns
    -------
    DecisionTreeClassifier
        Fitted model.
    """
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, Y_train)
    return model


def train_mlp(
    X_train: pd.DataFrame,
    Y_train: pd.Series,
    hidden_layer_sizes: tuple = (64, 64, 64),
    activation: str = "tanh",
    max_iter: int = 200,
    random_state: int = 42,
) -> MLPClassifier:
    """Fit and return an MLP classifier.

    Parameters
    ----------
    X_train : array-like
        Training feature matrix.
    Y_train : array-like
        Training labels.
    hidden_layer_sizes : tuple
        Number of neurons per hidden layer.
    activation : str
        Activation function (``'tanh'``, ``'relu'``, …).
    max_iter : int
        Maximum number of training epochs.
    random_state : int
        Random seed.

    Returns
    -------
    MLPClassifier
        Fitted model.
    """
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver="adam",
        max_iter=max_iter,
        random_state=random_state,
    )
    model.fit(X_train, Y_train)
    return model


def train_random_forest(
    X_train: pd.DataFrame,
    Y_train: pd.Series,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Fit and return a Random Forest classifier.

    Parameters
    ----------
    X_train : array-like
        Training feature matrix.
    Y_train : array-like
        Training labels.
    random_state : int
        Random seed.

    Returns
    -------
    RandomForestClassifier
        Fitted model.
    """
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, Y_train)
    return model


def train_ensemble(
    X_train: pd.DataFrame,
    Y_train: pd.Series,
    decision_tree: DecisionTreeClassifier,
    mlp: MLPClassifier,
    random_forest: RandomForestClassifier,
) -> VotingClassifier:
    """Fit and return a soft-voting Ensemble of DT, MLP, and RF.

    Parameters
    ----------
    X_train : array-like
        Training feature matrix.
    Y_train : array-like
        Training labels.
    decision_tree : DecisionTreeClassifier
        Pre-fitted Decision Tree model.
    mlp : MLPClassifier
        Pre-fitted MLP model.
    random_forest : RandomForestClassifier
        Pre-fitted Random Forest model.

    Returns
    -------
    VotingClassifier
        Fitted ensemble model.
    """
    ensemble = VotingClassifier(
        estimators=[
            ("DT", decision_tree),
            ("MLP", mlp),
            ("RF", random_forest),
        ],
        voting="soft",
    )
    ensemble.fit(X_train, Y_train)
    return ensemble


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    Y_test: pd.Series,
    model_name: str = "Model",
) -> float:
    """Evaluate *model* on the test set and print a classification report.

    Parameters
    ----------
    model : sklearn estimator
        Any fitted classifier with a ``predict`` method.
    X_test : array-like
        Test feature matrix.
    Y_test : array-like
        True labels.
    model_name : str
        Label shown in the printed output.

    Returns
    -------
    float
        Test-set accuracy.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)
    print(f"=== {model_name} ===")
    print(classification_report(Y_test, y_pred))
    print(f"Test accuracy: {accuracy:.4f}\n")
    return accuracy
