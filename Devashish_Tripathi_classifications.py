"""
Author: Devashish Tripathi

Description:Code which contains classes to perform classification in respective stages

"""


from time import time
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler

from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, SequentialFeatureSelector
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report, make_scorer, ConfusionMatrixDisplay

"""Class to evaluate the various classifier using GridSearch without any other modifications"""


class classification_methods:

    def __init__(self, classifier, features, labels):
        self.classifier = classifier
        self.features = features
        self.labels = labels

    """Function to store models and params to consider"""

    def create_classifier(self):

        model = None
        params = None

        # Decision Tree
        if self.classifier == "DecTree":
            print("Decision Tree Classification")
            model = DecisionTreeClassifier(
                random_state=1, class_weight='balanced')
            params = {'model__criterion': ['gini', 'entropy', 'log_loss'],
                      'model__splitter': ['best', 'random'],
                      'model__ccp_alpha': [0, 0.1, 0.05, 1, 0.001],
                      'model__max_features': ['auto', 'sqrt', 'log2']}

        # Logisitic Regression
        elif self.classifier == "LogRes":
            print("Logisitic Regression Classification")
            model = LogisticRegression(random_state=1, class_weight='balanced')
            params = {'model__solver': [
                'lbfgs', 'liblinear', 'saga', 'newton-cg']}

        # Random Forest
        elif self.classifier == "RanFor":
            print("Random Forest Classification")
            model = RandomForestClassifier(random_state=1,)
            params = {'model__n_estimators': [10, 100, 200],
                      'model__criterion': ['gini', 'entropy', 'log_loss'],
                      'model__max_features': ['sqrt', 'log2', None],
                      'model__class_weight': ['balanced', 'balanced_subsample']
                      }

        # SVC
        elif self.classifier == "SVC":
            print("Support Vector Machine Classification")
            model = SVC(random_state=1, class_weight='balanced')
            params = {"model__C": [1, 0.5, 0.01, 5],
                      "model__kernel": ['rbf', 'poly', 'sigmoid'],
                      }
        elif self.classifier == "LSVC":
            print("Linear SVM Classification")
            model = LinearSVC(random_state=1, class_weight='balanced')
            params = {"model__C": [1, 0.5, 0.01, 5],
                      }

        # Gaussian Naive Bayes
        elif self.classifier == "GNBay":
            print("Gaussian Naive Bayes Classification")
            model = GaussianNB()
            params = {}

        # Multinomial Naive Bayes
        elif self.classifier == "MNBay":
            print("Multinomial Naive Bayes Classification")
            model = MultinomialNB()
            params = {'model__alpha': [1, 0.5, 0.05, 0]}

        elif self.classifier == "KNN":
            print("K Nearest Neighbors Classification")
            model = KNeighborsClassifier()
            params = {"model__n_neighbors": [5, 300, 500],
                      "model__weights": ['uniform', 'distance'],
                      "model__p": [1, 2, 3]}

        # Adaboost Classifier
        elif self.classifier == "AdaB":
            print("Applying Adaboost classifier")
            model = AdaBoostClassifier(random_state=1)
            be_dt = DecisionTreeClassifier(ccp_alpha=0.001, criterion='gini',
                                           max_features='sqrt', splitter='random', random_state=1)
            be_lr = LogisticRegression(solver='newton-cg', random_state=1)
            be_rf = RandomForestClassifier(class_weight='balanced_subsample', criterion='entropy',
                                           max_features=None, n_estimators=200, random_state=1)
            be_svc = SVC(C=5, kernel='poly', random_state=1)
            be_mnb = MultinomialNB(alpha=1)
            be_gnb = GaussianNB()
            be_knn = KNeighborsClassifier(
                n_neighbors=1, p=1, weights='uniform')

            params = {'model__base_estimator': [be_dt, be_gnb, be_knn, be_lr, be_svc, be_mnb, be_rf],
                      'model__n_estimators': [50, 100, 200]}

        # GradBoost Classifier
        elif self.classifier == "GradB":
            print("Applying Gradboost classifier")
            model = GradientBoostingClassifier(random_state=1)
            params = {'model__loss': ['log_loss', 'exponential'],
                      'model__criterion': ['friedman_mse', 'squared_error'],
                      'model__ccp_alpha': [0, 1, 0.05, 5],
                      'model__n_estimators': [100, 200, 500]}

        else:
            print("Unknown Classifier...")
            exit(0)

        return model, params

    """Function to evaluate model"""

    def eval_model(self):
        X = self.features
        y = self.labels
        n_splits = 5

        print("-"*70)
        print(f'Running {self.classifier} for {n_splits} splits.')
        print("-"*70)
        run = 1

        SKF = StratifiedKFold(n_splits=n_splits, random_state=1, shuffle=True)
        for train_idx, test_idx in SKF.split(X, y):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]
            print()
            print("Run: ", run)
            print()

            model, params = self.create_classifier()

            pipe = Pipeline([('model', model)])
            scorer = make_scorer(f1_score, pos_label=1)
            grid = GridSearchCV(pipe, params, cv=5, scoring=scorer)

            start = time()
            grid.fit(X_train, y_train)
            print("Best params for this run:", grid.best_params_)
            end = time()

            y_train_pred = grid.predict(X_train)
            y_test_pred = grid.predict(X_test)

            print("Train data evaluation::")
            cm = confusion_matrix(y_train, y_train_pred)
            print(cm)
            d1 = ConfusionMatrixDisplay(cm)
            d1.plot()
            plt.show()
            print(classification_report(y_train, y_train_pred))
            print("Accuracy:", accuracy_score(y_train, y_train_pred))

            print("Test data evaluation::")
            cm = confusion_matrix(y_test, y_test_pred)
            print(cm)
            d2 = ConfusionMatrixDisplay(cm)
            d2.plot()
            plt.show()
            print(classification_report(y_test, y_test_pred))
            print(cm)
            print("Accuracy:", accuracy_score(y_test, y_test_pred))

            tot_time = end-start
            print("Time taken in the run:", tot_time)
            run += 1


"""Class to select models based on weights, scalings etc."""


class classification_selection:
    def __init__(self, classifier, features, labels):
        self.classifier = classifier
        self.features = features
        self.labels = labels

    def create_classifier(self):
        model = None

        # Decision Tree
        if self.classifier == "DecTree":
            print("Decision Tree Classification")
            model = DecisionTreeClassifier(random_state=1, class_weight='balanced',
                                           ccp_alpha=0.001, criterion='gini',
                                           max_features='sqrt', splitter='random')

        # Logisitic Regression
        elif self.classifier == "LogRes":
            print("Logisitic Regression Classification")
            model = LogisticRegression(
                random_state=1, class_weight='balanced', solver='newton-cg')

        elif self.classifier == "MNBay":
            print("Multinomial Naive Bayes Classification")
            model = MultinomialNB()
            params = {'model__alpha': [1, 0.5, 0.05, 0]}

        # Adaboost Classifier
        elif self.classifier == "AdaB":
            print("Applying Adaboost classifier")
            be_lr = LogisticRegression(
                random_state=1, class_weight='balanced', solver='newton-cg')
            model = AdaBoostClassifier(
                random_state=1, base_estimator=be_lr, n_estimators=100)

        # GradBoost Classifier
        elif self.classifier == "GradB":
            print("Applying Gradboost classifier")
            model = GradientBoostingClassifier(random_state=1, ccp_alpha=0,
                                               criterion='friedman_mse', loss='log_loss', n_estimators=500)

        else:
            print("Unknown Classifier...")
            exit(0)

        return model

    """Function to evaluate classifier"""

    def eval_classifier(self):

        X = self.features
        y = self.labels
        n_splits = 5

        print("-"*70)
        print(f'Running {self.classifier} for {n_splits} splits')
        print("-"*70)
        run = 1

        SKF = StratifiedKFold(n_splits=n_splits, random_state=1, shuffle=True)
        for train_idx, test_idx in SKF.split(X, y):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]
            print()
            print("Run: ", run)
            print()

            model = self.create_classifier()

            pipe = Pipeline(
                [('scaler', None), ('feature_sel', None), ('model', model)])

            params = [
                {'scaler': [None, MaxAbsScaler(), MinMaxScaler(), StandardScaler()],
                 'feature_sel': [None]},

                {'scaler': [None, MaxAbsScaler(), MinMaxScaler(), StandardScaler()],
                 'feature_sel': [SelectKBest(mutual_info_classif)],
                 'feature_sel__k': [5, 10, 20, 30]},

                {'scaler': [None, MaxAbsScaler(), MinMaxScaler(), StandardScaler()],
                 'feature_sel': [SelectKBest(chi2)],
                 'feature_sel__k': [5]},

                {'scaler': [None, MaxAbsScaler(), MinMaxScaler(), StandardScaler()],
                 'feature_sel': [SequentialFeatureSelector(estimator=model)],
                 'feature_sel__direction': ['forward', 'backward'],
                 'feature_sel__n_features_to_select': ['auto']},

                {'scaler': [None, MaxAbsScaler(), MinMaxScaler(), StandardScaler()],
                 'feature_sel': [PCA(random_state=1)],
                 'feature_sel__n_components': [5, 10, 20, 30]}

            ]

            scorer = make_scorer(f1_score, pos_label=1)
            grid = GridSearchCV(pipe, params, cv=5, scoring=scorer)

            start = time()
            grid.fit(X_train, y_train)
            print("Best params for this run:", grid.best_params_)
            end = time()

            y_train_pred = grid.best_estimator_.predict(X_train)
            y_test_pred = grid.best_estimator_.predict(X_test)

            print("Train data evaluation::")
            cm = confusion_matrix(y_train, y_train_pred)
            print(cm)
            # d1=ConfusionMatrixDisplay(cm)
            # d1.plot()
            # plt.show()
            print(classification_report(y_train, y_train_pred))
            print("Accuracy:", accuracy_score(y_train, y_train_pred))

            print("Test data evaluation::")
            cm = confusion_matrix(y_test, y_test_pred)
            print(cm)
            # d2=ConfusionMatrixDisplay(cm)
            # d2.plot()
            # plt.show()
            print(classification_report(y_test, y_test_pred))
            print("Accuracy:", accuracy_score(y_test, y_test_pred))

            tot_time = end-start
            print("Time taken in the run:", tot_time)
            run += 1


"""Class to create the model based on the final classifier and get labels for the test data"""


class final_classifier:
    def __init__(self, features, labels, test_data=None):
        self.features = features
        self.labels = labels
        self.test_data = test_data
        self.model = None

    """Creating the final model"""

    def final_eval(self):
        X = self.features
        y = self.labels
        be = LogisticRegression(
            solver='newton-cg', class_weight='balanced', random_state=1)
        ada_model = AdaBoostClassifier(
            base_estimator=be, n_estimators=200, random_state=1)

        n_splits = 5
        run = 1

        SKF = StratifiedKFold(n_splits=n_splits, random_state=1, shuffle=True)
        for train_idx, test_idx in SKF.split(X, y):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]
            print("-"*70)
            print("Run: ", run)
            print("-"*70)

            start = time()
            ada_model.fit(X_train, y_train)
            end = time()

            y_train_pred = ada_model.predict(X_train)
            y_test_pred = ada_model.predict(X_test)

            print("Train data evaluation::")
            cm = confusion_matrix(y_train, y_train_pred)
            d1 = ConfusionMatrixDisplay(cm)
            d1.plot()
            # plt.show()
            print(cm)
            print(classification_report(y_train, y_train_pred))
            print("Accuracy:", accuracy_score(y_train, y_train_pred))

            print("Test data evaluation::")
            cm = confusion_matrix(y_test, y_test_pred)
            d2 = ConfusionMatrixDisplay(cm)
            d2.plot()
            # plt.show()
            print(cm)
            print(classification_report(y_test, y_test_pred))
            print("Accuracy:", accuracy_score(y_test, y_test_pred))

            tot_time = end-start
            print("Time taken for model fitting in the run: %.3f seconds" % tot_time)
            run += 1

        self.model = ada_model

    """Function for evaluating and Storing the final results on the test data"""

    def eval_test(self, df_test):
        ada_model = self.model
        y = ada_model.predict(df_test)
        print(len(y))
        print(y)
        with open('Devashish_Tripathi_testlabels.txt', 'w') as file:
            for out in y:
                file.write(f'{out}\n')
