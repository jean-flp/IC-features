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
sequence_embedding_model = 'protein_bert'

if sequence_embedding_model == 'esm':
    X_train = X_train.drop(columns=[col for col in X_train.columns if "proteinbert_pca" in col or "prot_pca" in col])
    X_test = X_test.drop(columns=[col for col in X_test.columns if "proteinbert_pca" in col or "prot_pca" in col])

    X_mansoni = X_mansoni.drop(columns=[col for col in X_mansoni.columns if "proteinbert_pca" in col or "prot_pca" in col])
    X_musculus = X_musculus.drop(columns=[col for col in X_musculus.columns if "proteinbert_pca" in col or "prot_pca" in col])

elif sequence_embedding_model == 'protein_bert':
    X_train = X_train.drop(columns=[col for col in X_train.columns if "esm_pca" in col or "prot_pca" in col])
    X_test = X_test.drop(columns=[col for col in X_test.columns if "esm_pca" in col or "prot_pca" in col])

    X_mansoni = X_mansoni.drop(columns=[col for col in X_mansoni.columns if "esm_pca" in col or "prot_pca" in col])
    X_musculus = X_musculus.drop(columns=[col for col in X_musculus.columns if "esm_pca" in col or "prot_pca" in col])

elif sequence_embedding_model == 'prot':
    X_train = X_train.drop(columns=[col for col in X_train.columns if "proteinbert_pca" in col or "esm_pca" in col])
    X_test = X_test.drop(columns=[col for col in X_test.columns if "proteinbert_pca" in col or "esm_pca" in col])

    X_mansoni = X_mansoni.drop(columns=[col for col in X_mansoni.columns if "proteinbert_pca" in col or "esm_pca" in col])
    X_musculus = X_musculus.drop(columns=[col for col in X_musculus.columns if "proteinbert_pca" in col or "esm_pca" in col])


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
runner = PipelineRunner(
    cv=5,
    scoring="roc_auc",
    search_strategy="grid",
    seed=__SEED__

)
runner.run(
    model_name="random_forest",
    pipelines=pipelines_rf,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    X_external=X_musculus,
    y_external=y_musculus
)

runner.run(
    model_name="gradient_boosting",
    pipelines=pipelines_gb,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    X_external=X_musculus,
    y_external=y_musculus
)

runner.run(
    model_name="xgboost",
    pipelines=pipelines_xgb,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    X_external=X_musculus,
    y_external=y_musculus
)

#%%
""" Carregando os melhores modelos dos tipos de classificadores """
# Os melhores modelos de cada tipo de classificador foram salvos e agora serão utilizados para predizer o mus musculus, então com esses resultados será feitas a interseção e por fim as métricas dos resultados
# Por fim o mesmo será feito no organismo alvo, mansoni



print("\n" + "="*70)
print("PREDIÇÕES NO MUS MUSCULUS")
print("="*70)

# Predições de cada modelo
predictions_musculus = {}

for model_type, info in best_models.items():
    model = info['model']
    model_name = info['name']
    
    # Fazer predições
    y_pred = model.predict(X_musculus)
    y_proba = model.predict_proba(X_musculus)[:, 1]
    
    predictions_musculus[model_type] = {
        'predictions': y_pred,
        'probabilities': y_proba,
        'model_name': model_name
    }
    
    # Avaliar
    print(f"\n{model_type} ({model_name}):")
    print(f"  Predições positivas: {y_pred.sum()}/{len(y_pred)}")
    print("\n" + classification_report(y_musculus, y_pred, target_names=['Não-Essencial', 'Essencial']))
