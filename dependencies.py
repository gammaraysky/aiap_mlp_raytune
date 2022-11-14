##### INIT AND HELPER METHODS, CLASSES
import logging
import sqlite3
import urllib.request
from typing import Optional, List, Dict
from joblib import dump

import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, \
                            f1_score, fbeta_score
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier,\
                             HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

from pprint import pprint

import ray
from ray import tune
from ray.tune.sklearn import TuneSearchCV
from ray.tune.search.hyperopt import HyperOptSearch
from ray.air import session, RunConfig
#? set up logging to print INFO level information
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt

pd.options.display.float_format = '{:,.4f}'.format

class Col_transformers():

    def __init__(self, nominal, ordinal, numeric):

        # col-specific ordinal encoding
        self.qualification = [['Diploma', 'Bachelor', 'Master', 'Ph.D',]]
        self.membership = [['Normal',  'Bronze', 'Silver', 'Gold',]]

        # column transformer for dataset
        self.transformer = ColumnTransformer(transformers=[
                ('categorical',     OneHotEncoder(), nominal),
                ('qualification',   OrdinalEncoder(categories=self.qualification), ['Qualification']),
                ('membership',      OrdinalEncoder(categories=self.membership), ['Membership']),
                ('gender',          OrdinalEncoder(), ['Gender']),
                ('numeric',         RobustScaler(), [c for c in numeric if c!='Gender'])
        ])

        # column transformer specific for HistGBM
        self.transformer_ord = ColumnTransformer(transformers=[
                ('categorical',     OrdinalEncoder(), nominal),
                ('qualification',   OrdinalEncoder(categories=self.qualification), ['Qualification']),
                ('membership',      OrdinalEncoder(categories=self.membership), ['Membership']),
                ('gender',          OrdinalEncoder(), ['Gender']),
                ('numeric',         RobustScaler(), [c for c in numeric if c!='Gender'])
        ])

        # categorical feature mask specific for HistGBM
        self.cat_feature_mask = [True]*5 + [False]*8

        return


class Dataset():
    def __init__(self, test_size, random_state):
        ### LOAD DATA: X, y,
        # as well as define lists of nominal, ordinal, and numeric columns (lists to be passed into column transformers)
        self.X, self.y, self.nominal, self.ordinal, self.numeric = self.load_clean_data()

        self.test_size = test_size
        self.random_state = random_state

        ### LOAD COLUMN TRANSFORMER
        self.tfm = Col_transformers(self.nominal, self.ordinal, self.numeric)


        self.oversample = Pipeline(steps=[('ros', RandomOverSampler(random_state=self.random_state)) ])



    def get_ordinal_imbalanced(self):
        return self.tfm.transformer_ord.fit_transform(self.X, self.y), self.y

    def get_onehot_imbalanced(self):
        return self.tfm.transformer.fit_transform(self.X, self.y), self.y

    def get_transformed_ordinal(self):
        # TRANSFORM X
        self.X_ord = self.tfm.transformer_ord.fit_transform(self.X, self.y)

        # TRAIN TEST SPLIT
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_ord, self.y, test_size=self.test_size, random_state=self.random_state, shuffle=True, stratify=self.y)

        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_transformed_ordinal_oversample(self):
        # TRANSFORM X
        self.X_ord = self.tfm.transformer_ord.fit_transform(self.X, self.y)

        # TRAIN TEST SPLIT
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_ord, self.y, test_size=self.test_size, random_state=self.random_state, shuffle=True, stratify=self.y)

        # OVERSAMPLE THE SPLITS
        self.X_train_os, self.y_train_os = self.oversample.fit_resample(self.X_train, self.y_train)
        self.X_test_os, self.y_test_os = self.oversample.fit_resample(self.X_test, self.y_test)

        return self.X_train, self.X_test, self.y_train, self.y_test, self.X_train_os, self.X_test_os, self.y_train_os, self.y_test_os


    def get_transformed_onehot(self):
        # TRANSFORM X
        self.X_ohe = self.tfm.transformer.fit_transform(self.X, self.y)

        # TRAIN TEST SPLIT
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_ohe, self.y, test_size=self.test_size, random_state=self.random_state, shuffle=True, stratify=self.y)

        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_transformed_onehot_oversample(self):
        # TRANSFORM X
        self.X_ohe = self.tfm.transformer.fit_transform(self.X, self.y)

        # TRAIN TEST SPLIT
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_ohe, self.y, test_size=self.test_size, random_state=self.random_state, shuffle=True, stratify=self.y)

        # OVERSAMPLE THE SPLITS
        self.X_train_os, self.y_train_os = self.oversample.fit_resample(self.X_train, self.y_train)
        self.X_test_os, self.y_test_os = self.oversample.fit_resample(self.X_test, self.y_test)

        return self.X_train, self.X_test, self.y_train, self.y_test, self.X_train_os, self.X_test_os, self.y_train_os, self.y_test_os

    def load_clean_data(self):

        #? load dataset from file
        conn = sqlite3.connect(r'D:\code\aiap12-mohammad-hanafi-bin-md-haffidz-380D\v2\data\attrition.db') #('./data/attrition.db')
        df = pd.read_sql_query("SELECT * FROM attrition", conn)

        #? run clean steps
        ### DROP MEMBER UNIQUE ID COL
        df.drop('Member Unique ID', axis=1, inplace=True)

        ### CLEAN TYPOS IN TRAVEL TIME, QUALIFICATIONS COLS
        df['Travel Time'] = df['Travel Time'].apply(self.convert_traveltime)
        df['Qualification'] = df['Qualification'].apply(self.convert_qualifications)

        ### ENCODE GENDER COL
        df['Gender'] = df['Gender'].map({'Male': 1,'Female': 0})
        df['Gender'] = df['Gender'].astype('int64')


        ### RESTRIPE -1 AS NP.NAN IN AGE AND BIRTH YEAR COLUMNS
        for col in ['Birth Year','Age']:
            df[col] = df[col].astype('int64')
            df[col] = [np.nan if i==-1 else i for i in df[col]]

        ### FILL MISSING ROWS OF BIRTH YEAR WITH 2022 - AGE
        df.loc[df['Birth Year'].isna(), ['Birth Year']] = 2022 - df['Age']

        ### DROP AGE AS IT IS REDUNDANT NOW
        df.drop('Age', axis=1, inplace=True)

        df['Months_log'] = [ np.log(i)  if i>0 else 0 for i in df['Months']]
        df['UsageTime_log'] = [ np.log(i)  if i>0 else 0 for i in df['Usage Time']]

        ### SPLIT COLS INTO NUMERIC AND CATEGORICALS
        numeric   = [col for col in df.columns if df[col].dtype == 'float64' or df[col].dtype == 'int64' and col!='Attrition']
        nominal   = [col for col in df.columns if df[col].dtype == 'object' or df[col].dtype == 'category']

        for col in nominal:
            df[col] = df[col].astype('category')

        ordinal   = ['Membership', 'Qualification']
        nominal   = [c for c in nominal if c not in ordinal]

        ### DROP ROWS WITH NEGATIVE INCOME
        df = df[df['Monthly Income']>=0]

        df = df.reset_index(drop=True)


        y = df.pop('Attrition')
        X = df

        return X, np.array(y), nominal, ordinal, numeric


    ### CLEAN TRAVEL TIME COL
    def convert_traveltime(self, row:Optional[str]=None) -> str:
        """Cleans up Travel Time column entries '0.2 hours' or '20 mins' into hours in float."""
        if row.endswith('hours'):
            row = row.split(' hours')[0]
            row = float(row)
        elif row.endswith('mins'):
            row = row.split(' mins')[0]
            row = float(row)/60

        return row

    ### CLEAN QUALIFICATIONS COL
    def convert_qualifications(self, row:Optional[str]=None) -> str:
        """Cleans up Qualification column entries misspellings e.g. "Master's" and "Master" """
        if row == "Master's":
            row = 'Master'
        elif row == "Bachelor's":
            row = 'Bachelor'
        elif row == 'Doctor of Philosophy':
            row = 'Ph.D'

        return row





def trial_results(results) -> Dict:
    #
    # prints dataframe of trial results, prints best params, and also returns bestconfig we can feed to rebuild a model from
    #
    df = results.get_dataframe()
    cols = [c for c in df.columns if 'auc' in c or 'config' in c]
    topn = df[cols].sort_values(['mean_test_imb_auc', 'mean_test_os_auc', 'mean_val_os_auc',  ], ascending=False).loc[:10]
    # topn = df[cols].sort_values(['mean_val_os_auc', 'mean_test_os_auc', 'mean_test_imb_auc',    ], ascending=False).loc[:10]
    display(topn)

    ### SHOW BEST PARAMS
    best_result = results.get_best_result(metric='mean_test_imb_auc', mode='max')  # Get best result object
    best_config = best_result.config # save best config for refit later

    print('BEST METRICS, BEST PARAMS\n-------------------------\n')
    [print(f'{k:>20} : {best_result.metrics[k]:.4f}') for k in best_result.metrics if 'auc' in k]
    best_result.metrics['config']

    return best_config

def refit_model(clf, ordinal=False):
    dataset = Dataset(TEST_SIZE, RANDOM_STATE)
    if (ordinal):
        X_train, X_test, y_train, y_test, X_train_os, X_test_os, y_train_os, y_test_os = dataset.get_transformed_ordinal_oversample()
    else:
        X_train, X_test, y_train, y_test, X_train_os, X_test_os, y_train_os, y_test_os = dataset.get_transformed_onehot_oversample()

    clf.fit(X_train_os, y_train_os)

    y_trainpred_os = clf.predict(X_train_os)
    y_testpred_os  = clf.predict(X_test_os)
    y_testpred_imb = clf.predict(X_test)

    train_os_auc = roc_auc_score(y_train_os, y_score=y_trainpred_os, average="macro")
    test_os_auc  = roc_auc_score(y_test_os,  y_score=y_testpred_os,  average="macro")
    test_imb_auc = roc_auc_score(y_test,     y_score=y_testpred_imb, average="macro")

    print("train_os_auc  test_os_auc  test_imb_auc ")
    print(f"{train_os_auc:>12.4f}  {test_os_auc:>11.4f}  {test_imb_auc:>12.4f}")

    return

def tune_for_recall(clf, X,y, threshline):

    y_pred_proba = clf.predict_proba(X)[:,1] # predict proba gives matrix [[0.1, 0.9], [0.7, 0.3], ...] for class 0, class 1. so just take class 1.

    thresholds = np.arange(0.01, 1, 0.01)
    predictions = []
    accs,f1s,f2s,pres,recs = [],[],[],[],[]
    # Find scores for each threshold
    for threshold in thresholds:

        prediction = [ 1 if i>threshold else 0 for i in y_pred_proba ]

        predictions.append(prediction)

        acc = accuracy_score(y, prediction)
        f1 = f1_score(y, prediction)
        f2 = fbeta_score(y, prediction, beta=2)
        pre = precision_score(y, prediction, zero_division=0)
        rec = recall_score(y, prediction)

        accs.append(acc)
        f1s.append(f1)
        f2s.append(f2)
        pres.append(pre)
        recs.append(rec)

    fig, ax = plt.subplots(figsize=(10,5), constrained_layout=True)

    ax.plot(thresholds, accs, lw=2.5, alpha=0.8, label='acc')
    ax.plot(thresholds, f1s,  lw=2.5, alpha=0.8, label='f1')
    ax.plot(thresholds, f2s,  lw=2.5, alpha=0.8, label='f2')
    ax.plot(thresholds, pres, lw=2.5, alpha=0.8, label='prec')
    ax.plot(thresholds, recs, lw=2.5, alpha=0.8, label='rec')
    line = ax.axvline(x=threshline, alpha=0.7)
    ax.text(x=threshline-0.02, y=0.95, s=f"{threshline}", fontsize=12, c=line.get_color())
    ax.set_xlabel('Probability thresholds')
    ax.set_title('Accuracy, F1, F2, Precision, Recall at varying proba thresholds')
    ax.legend()
    plt.show()

    return pd.DataFrame({
        'threshold' : thresholds,
        'accuracy' : accs,
        'f1' : f1s,
        'f2' : f2s,
        'precision' : pres,
        'recall' : recs})


