"""
Author: Devashish Tripathi

Description:Main code for classification. Loads and Displays the datasets. 

"""


from sklearn.impute import KNNImputer, SimpleImputer

from Devashish_Tripathi_classifications import classification_selection
from Devashish_Tripathi_classifications import classification_methods
from Devashish_Tripathi_classifications import final_classifier

import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

""" Function to display the datasets"""


def disp_df(df, state):
    print("*"*80)
    print(f"Currently showing info of {state} data")
    print(df.info())
    print("*"*80)
    if state == 'label':
        plt.figure()
        ax = df.value_counts().plot(kind='bar',)
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.title('Label info')
        ax.bar_label(ax.containers[-1], label_type='edge')
        plt.show()

    else:

        if state == 'train' or state == 'test':
            disp_df = df.drop([' Income ', 'Dt_Customer'], axis=1)
            for col in disp_df.columns:
                counts = disp_df[col].value_counts()
                vis_df = pd.DataFrame()
                vis_df[col] = counts
                print(vis_df)

        else:
            disp_df = df.drop([' Income '], axis=1)
            for col in disp_df.columns:
                counts = disp_df[col].value_counts()
                vis_df = pd.DataFrame()
                vis_df[col] = counts
                print(vis_df)

        print("Duplicated information")
        df_dups = df[df.duplicated()]
        print("No. of duplicates:", df_dups.shape[0])


""" Function to clean the datasets"""


def clean_df(df, state):
    df[' Income '] = df[' Income '].str.replace('$', '')
    df[' Income '] = df[' Income '].str.replace(',', '')
    df[' Income '] = df[' Income '].astype('float64')

    k = 5
    strat = 'mean'
    # strat = 'median'

    # kimp = KNNImputer(missing_values=np.nan, n_neighbors=k)
    # kimp.set_output(transform='pandas')
    simp=SimpleImputer(missing_values=np.nan,strategy=strat,)
    simp.set_output(transform='pandas')

    Dt_Cst = df['Dt_Customer']
    df = df.drop(['Dt_Customer'], axis=1)
    df = pd.get_dummies(df)
    Dt_Cst = pd.to_datetime(Dt_Cst)
    df['Dt_Cst_year'] = Dt_Cst.dt.year
    df['Dt_Cst_month'] = Dt_Cst.dt.month
    df['Dt_Cst_day'] = Dt_Cst.dt.day

    print("Using mean for impute")
    # print("Using median for impute")
    # print("Using KNN-Imputer with k=", k)

    # df = kimp.fit_transform(df)
    df = simp.fit_transform(df)
    if state == 'train':
        df = df.drop(['Marital_Status_Absurd', 'Marital_Status_YOLO'], axis=1)

    return df


""" Loading, Cleaning and Displaying the data"""

trn_data = pd.read_csv('marketing_trn_data.csv')
trn_labels = pd.read_csv(
    'marketing_trn_class_labels.csv', names=['idx', 'label'])
tst_data = pd.read_csv('marketing_tst_data.csv')
df = trn_data.copy()
df_test = tst_data.copy()

disp_df(df, 'train')
# disp_df(df_test, 'test')

X = clean_df(df, 'train')
y = trn_labels['label']

# disp_df(X, 'cleaned train')
disp_df(y, 'label')


"""Running all classifiers on all features and no scaling"""

# classifiers=["DecTree","LogRes","GNBay","MNBay","KNN","LSVC","RanFor","SVC"]
# for classifier in classifiers:
#     classify=classification_methods(classifier,X,y)
#     classify.eval_model()

"""Running all classifiers with class weights balanced on all features and no scaling"""

# weight_classifiers=["DecTree","LogRes","LSVC","RanFor","SVC"]
# for classifier in weight_classifiers:
#     classify=classification_methods(classifier,X,y)
#     classify.eval_model()

"""Running Gradient Boost and AdaBoosts"""

# #Gradient Boosting
# classify=classification_methods('GradB',X,y)
# classify.eval_model()

# AdaBoost
# classify=classification_methods('AdaB',X,y)
# classify.eval_model()

"""Choosing best possible model out of LogRes, DecTree, MNBay, Adaboost and GradBoost by doing feature selection etc."""

# sel_classifiers=["DecTree","LogRes","MNBay","AdaB","GradB"]
# for classifier in sel_classifiers:
#     classify=classification_selection(classifier,X,y)
#     classify.eval_classifier()


""" After evaluation, the best model turned out to be Adaboost with Logistic Regression base estimator 
    With no scaling and no feature selection"""


df_test = clean_df(df_test, 'test')
# disp_df(df_test, 'clean test')

"""Evaluating results on the final classifier and finding labels of the test data"""

f_clf = final_classifier(X, y)
f_clf.final_eval()
f_clf.eval_test(df_test)
