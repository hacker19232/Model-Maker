import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression,
    PassiveAggressiveClassifier, PassiveAggressiveRegressor, SGDClassifier, SGDRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor,
    BaggingClassifier, BaggingRegressor, AdaBoostClassifier, AdaBoostRegressor, IsolationForest,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.decomposition import PCA, KernelPCA
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import (
    RANSACRegressor, TheilSenRegressor, HuberRegressor, ARDRegression, BayesianRidge, Lars, LassoLars, OrthogonalMatchingPursuit,
)
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV
from sklearn.dummy import DummyClassifier, DummyRegressor

class Model():
    def __init__(self, dataset):
        classification_models = {
    'LogisticRegression': LogisticRegression(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'SVC': SVC(),
    'AdaBoostClassifier': AdaBoostClassifier(),
    'GaussianNB': GaussianNB(),
    'MultinomialNB': MultinomialNB(),
    'BernoulliNB': BernoulliNB(),
    'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis(),
    'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
    'RidgeClassifier': RidgeClassifier(),
    'RidgeClassifierCV': RidgeClassifierCV(),
    'DummyClassifier': DummyClassifier()
}
        regression_models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'ElasticNet': ElasticNet(),
    'DecisionTreeRegressor': DecisionTreeRegressor(),
    'RandomForestRegressor': RandomForestRegressor(),
    'GradientBoostingRegressor': GradientBoostingRegressor(),
    'KNeighborsRegressor': KNeighborsRegressor(),
    'SVR': SVR(),
    'AdaBoostRegressor': AdaBoostRegressor(),
    'MLPRegressor': MLPRegressor(),
    'RANSACRegressor': RANSACRegressor(),
    'TheilSenRegressor': TheilSenRegressor(),
    'HuberRegressor': HuberRegressor(),
    'ARDRegression': ARDRegression(),
    'BayesianRidge': BayesianRidge(),
    'Lars': Lars(),
    'LassoLars': LassoLars(),
    'OrthogonalMatchingPursuit': OrthogonalMatchingPursuit(),
    'DummyRegressor': DummyRegressor()
}

        self.dataset = dataset
        self.df = None
        self.X = None
        self.y = None
    
    def print9(self):
        if self.df is not None:
            return self.df
        
        else:
            print("Dataset is not created. Call create_df() first.")

    def create_df(self):
        try:
            self.df = pd.read_csv(self.dataset)
        except FileNotFoundError:
            print("Dataset file not found. Please check the file path.")
            return

    def create_xy(self, y_column):
        if self.df is not None:
            self.y = self.df[y_column]
            self.X = self.df.drop(y_column, axis=1)
        else:
            print("Dataset is not loaded. Call create_df() first.")

    def split_data(self, testing_size=0.33):
        if self.X is not None and self.y is not None:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=testing_size)
        else:
            print("Features and labels are not created. Call create_xy() first.")
    


    def load_all_classification(self):
        if self.X_train is not None and self.y_train is not None:
            best_model_name = None
            best_accuracy = 0.0
            best_confusion_matrix = None
            model_metrics = {}  # Dictionary to store model metrics

            for model_name, model_instance in self.classification_models.items():
                print(f"Training {model_name}...")
                model_instance.fit(self.X_train, self.y_train)

                # Test the model
                self.y_pred = model_instance.predict(self.X_test)
                accuracy = metrics.accuracy_score(self.y_test, self.y_pred)
                confusion_matrix = metrics.confusion_matrix(self.y_test, self.y_pred, labels=np.arange(1, 8))

                # Store metrics in the dictionary
                model_metrics[model_name] = {
                    'accuracy': accuracy,
                    'confusion_matrix': confusion_matrix
                }

                print(f"{model_name} trained successfully. Accuracy: {accuracy:.2f}")

                # Update the best model if necessary
                if accuracy > best_accuracy:
                    best_model_name = model_name
                    best_accuracy = accuracy
                    best_confusion_matrix = confusion_matrix

            print(f"Best Classification Model: {best_model_name}")
            print(f"Best Classification Accuracy: {best_accuracy:.2f}")
            print(f"Confusion Matrix of the Best Classification Model:")
            print(best_confusion_matrix)

            return best_model_name, best_accuracy, best_confusion_matrix
        else:
            print("Training data is not split. Call split_data() first.")

    def load_all_regression(self):
        if self.X_train is not None and self.y_train is not None:
            best_model_name = None
            best_mse = float('inf')  # Initialize to positive infinity
            model_metrics = {}  # Dictionary to store model metrics

            for model_name, model_instance in self.regression_models.items():
                print(f"Training {model_name}...")
                model_instance.fit(self.X_train, self.y_train)

                # Test the model
                self.y_pred = model_instance.predict(self.X_test)
                mse = metrics.mean_squared_error(self.y_test, self.y_pred)

                # Store metrics in the dictionary
                model_metrics[model_name] = {
                    'mean_squared_error': mse
                }

                print(f"{model_name} trained successfully. Mean Squared Error: {mse:.2f}")

                # Update the best model if necessary
                if mse < best_mse:
                    best_model_name = model_name
                    best_mse = mse

            print(f"Best Regression Model: {best_model_name}")
            print(f"Best Mean Squared Error: {best_mse:.2f}")

            return best_model_name, best_mse
        else:
            print("Training data is not split. Call split_data() first.")











# Example usage:
# model = Model('your_dataset.csv')
# model.create_df()
# model.create_xy('target_column')
# model.split_data()
# model.train('RandomForestClassifier')
# model.test()
# print(model.accuracy_score())
# print(model.confusion_matrix())
