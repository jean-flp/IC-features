#%%
from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import RFE, SelectKBest, f_classif, VarianceThreshold, SelectFromModel

from hyperparameter_search.strategy_search import SearchContext
from hyperparameter_search.search_spaces import SEARCH_SPACES

from pathlib import Path
import os
from tqdm import tqdm

import json

import mlflow
from  dotenv import load_dotenv
import sys
from runner.pipeline_runner import PipelineRunner

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../")
)
print(project_root)

sys.path.append(project_root)
from calculo_feature.synthesizeUtils import train_test_between_files
from mlflow_config.mlflow_server import start_server
from mlflow_config.mlflow_config import configure_mlflow

__X_TRAIN__, __X_TEST__, __Y_TRAIN__, __Y_TEST__ = train_test_between_files()

sys.path.remove(project_root)


load_dotenv()
__SEED__ = int(os.getenv("SEED"))
__TEST_SIZE__ = float(os.getenv("TESTE_SIZE"))
ML_FLOW_NPCA_VALUE = os.getenv("ML_FLOW_NPCA_VALUE")
ML_FLOW_MODEL_VALUE = os.getenv("ML_FLOW_MODEL_VALUE")

#%%
## MLFLOW 

start_server()
configure_mlflow()

BASE_DIR = Path(__file__).resolve().parent

PATH_SYNTHESIZED_FILES = BASE_DIR.parent / "data" / "processed" / "synthesized"

print(PATH_SYNTHESIZED_FILES)

#%%
df = pd.DataFrame()
referencia = None

excluded_files = {'10090','6183'}

for root, dirs, files in os.walk(PATH_SYNTHESIZED_FILES):

    for file in files:

        if file.split('.')[0] in excluded_files:
            continue

        df_temp = pd.read_csv(os.path.join(root, file))
        df = pd.concat([df,df_temp],ignore_index=True)

df.info()

df_mansoni = pd.read_csv(os.path.join(root,f"6183.synthesized_features.csv"))
df_musculus = pd.read_csv(os.path.join(root,f"10090.synthesized_features.csv"))

#%%

""" Início ML """
# Separação em conjuntos de treino e teste

X_train_recorte, X_test_recorte, y_train_recorte, y_test_recorte = __X_TRAIN__, __X_TEST__, __Y_TRAIN__, __Y_TEST__ 



X_train = df[df["protein_id"].isin(X_train_recorte["protein_id"])].drop(["Locus","essential","Sequence","protein_id"],axis=1)
X_test = df[df["protein_id"].isin(X_test_recorte["protein_id"])].drop(["Locus","essential","Sequence","protein_id"],axis=1)
y_train = df[df["protein_id"].isin(y_train_recorte["protein_id"])]["essential"]
y_test = df[df["protein_id"].isin(y_test_recorte["protein_id"])]["essential"]


X_mansoni = df_mansoni.drop(["Locus","essential","Sequence","protein_id"],axis=1)
X_musculus = df_musculus.drop(["Locus","essential","Sequence","protein_id"],axis=1)
y_musculus = df_musculus['essential']

#%%
dict_model_value =  ML_FLOW_MODEL_VALUE
dict_npca_value = ML_FLOW_NPCA_VALUE

print(dict_model_value)
print(dict_npca_value)

def filter_columns(df, sequence_model=None, npca=None):
    cols = []

    for col in df.columns:
        # Mantém todas as colunas que não são embeddings

        if not col.startswith(("proteinbert_", "prot_", "esm_", "node2vec_")):
            cols.append(col)
            continue

        # Mantém apenas colunas não-embedding
        if sequence_model == "no_embedding":
            continue
        # Mantém sempre o Node2Vec correspondente ao PCA escolhido
        if npca is not None and col.startswith(f"node2vec_n{npca}_"):
            cols.append(col)
            continue

        # Mantém apenas o embedding de sequência escolhido
        if sequence_model == "proteinbert" and col.startswith(f"proteinbert_n{npca}_"):
            cols.append(col)

        elif sequence_model == "prot" and col.startswith(f"prot_n{npca}_"):
            cols.append(col)

        elif sequence_model == "esm" and col.startswith(f"esm_n{npca}_"):
            cols.append(col)

    return df[cols]

#%%
X_train = filter_columns(df=X_train,sequence_model=dict_model_value,npca=dict_npca_value)
X_test = filter_columns(df=X_test,sequence_model=dict_model_value,npca=dict_npca_value)
X_mansoni = filter_columns(df=X_mansoni,sequence_model=dict_model_value,npca=dict_npca_value)
X_musculus = filter_columns(df=X_musculus,sequence_model=dict_model_value,npca=dict_npca_value)

print("Train:", X_train.shape)
print("Test:", X_test.shape)
print("Mansoni:", X_mansoni.shape)
print("Musculus:", X_musculus.shape)

print(set(X_train.columns) - set(X_musculus.columns))
print(set(X_musculus.columns) - set(X_train.columns))

#%%
pipelines_rf = {
    'rf-base': ImbPipeline([
        #('selector', SelectKBest(score_func=f_classif)),
        ('classifier', RandomForestClassifier(class_weight='balanced', random_state=__SEED__))
    ]),

    'rf-undersample': ImbPipeline([
        ('sampler', RandomUnderSampler(random_state=__SEED__)),
        #('selector', SelectKBest(score_func=f_classif)),
        ('classifier', RandomForestClassifier(random_state=__SEED__))
    ]),

    'rf-oversample': ImbPipeline([
        ('sampler', RandomOverSampler(random_state=__SEED__)),
        #('selector', SelectKBest(score_func=f_classif)),
        ('classifier', RandomForestClassifier(random_state=__SEED__))
    ]),
    
    'rf-smote': ImbPipeline([
        ('sampler', SMOTE(random_state=__SEED__)),
        #('selector', SelectKBest(score_func=f_classif)),
        ('classifier', RandomForestClassifier(random_state=__SEED__))
    ]),
    
    'rf-smoteenn': ImbPipeline([
        ('sampler', SMOTEENN(random_state=__SEED__)),
        #('selector', SelectKBest(score_func=f_classif)),
        ('classifier', RandomForestClassifier(random_state=__SEED__))
    ])
}

pipelines_gb = {
    'gb_baseline': ImbPipeline([
        #('selector', SelectKBest(score_func=f_classif)),
        ('classifier', GradientBoostingClassifier(random_state=__SEED__))
    ]),

    'gb_oversample': ImbPipeline([
        ('sampler', RandomOverSampler(random_state=__SEED__)),
        #('selector', SelectKBest(score_func=f_classif)),
        ('classifier', GradientBoostingClassifier(random_state=__SEED__))
    ]),

    'gb_undersample': ImbPipeline([
        ('sampler', RandomUnderSampler(random_state=__SEED__)),
        #('selector', SelectKBest(score_func=f_classif)),
        ('classifier', GradientBoostingClassifier(random_state=__SEED__))
    ]),

    'gb_smote': ImbPipeline([
        ('sampler', SMOTE(random_state=__SEED__)),
        #('selector', SelectKBest(score_func=f_classif)),
        ('classifier', GradientBoostingClassifier(random_state=__SEED__))
    ]),
    
    'gb_smoteenn': ImbPipeline([
        ('sampler', SMOTEENN(random_state=__SEED__)),
        #('selector', SelectKBest(score_func=f_classif)),
        ('classifier', GradientBoostingClassifier(random_state=__SEED__))
    ])
}


neg = (y_train == 0).sum()
pos = (y_train == 1).sum()

ratio = neg / pos

pipelines_xgb = {
    'xgb-scale-pos-weight': ImbPipeline([
        ('classifier', XGBClassifier(booster='gbtree', scale_pos_weight=ratio, verbosity=0, random_state=__SEED__))
    ]),

    'xgb_undersample': ImbPipeline([
        ('sampler', RandomUnderSampler(random_state=__SEED__)),
        #('selector', SelectKBest(score_func=f_classif)),
        ('classifier', XGBClassifier(booster='gbtree', verbosity=0, random_state=__SEED__))
    ]),

    'xgb_oversample': ImbPipeline([
        ('sampler', RandomOverSampler(random_state=__SEED__)),
        #('selector', SelectKBest(score_func=f_classif)),
        ('classifier', XGBClassifier(booster='gbtree', verbosity=0, random_state=__SEED__))
    ]),
    
    'xgb_smote': ImbPipeline([
        ('sampler', SMOTE(random_state=__SEED__)),
        #('selector', SelectKBest(score_func=f_classif)),
        ('classifier', XGBClassifier(booster='gbtree', verbosity=0, random_state=__SEED__))
    ]),
    
    'xgb_smoteenn': ImbPipeline([
        ('sampler', SMOTEENN(random_state=__SEED__)),
        #('selector', SelectKBest(score_func=f_classif)),
        ('classifier', XGBClassifier(booster='gbtree', verbosity=0, random_state=__SEED__))
    ])
}



# pipeline com GridSearch
# search_strategies: grid, optuna(bayes) e randomized
runner = PipelineRunner(
    cv=5,
    scoring="roc_auc",
    search_strategy="optuna",
    seed=__SEED__

)
#%%
runners_results_random_forest = runner.run(
    model_name="random_forest",
    pipelines=pipelines_rf,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    X_external=X_musculus,
    y_external=y_musculus
)
# #%%
# with open("random_forest_results_output.json", "w", encoding="utf-8") as file:
#     json.dump(runners_results_random_forest, file, indent=4, sort_keys=True)
#%%
runners_results_gradient_boosting = runner.run(
    model_name="gradient_boosting",
    pipelines=pipelines_gb,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    X_external=X_musculus,
    y_external=y_musculus
)

# with open("gradient_boosting_results_output.json", "w", encoding="utf-8") as file:
#     json.dump(runners_results_gradient_boosting, file, indent=4, sort_keys=True)

#%%
runners_results_xgboost = runner.run(
    model_name="xgboost",
    pipelines=pipelines_xgb,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    X_external=X_musculus,
    y_external=y_musculus
)

# with open("xgboost_results_output.json", "w", encoding="utf-8") as file:
#     json.dump(runners_results_xgboost, file, indent=4, sort_keys=True)


#%%
""" Carregando os melhores modelos dos tipos de classificadores """
# Os melhores modelos de cada tipo de classificador foram salvos e agora serão utilizados para predizer o mus musculus, então com esses resultados será feitas a interseção e por fim as métricas dos resultados
# Por fim o mesmo será feito no organismo alvo, mansoni
