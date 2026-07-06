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


# from skopt import BayesSearchCV
# from skopt.space import Real, Integer, Categorical

from pathlib import Path
import os
from tqdm import tqdm

import mlflow
from  dotenv import load_dotenv
import sys

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
def pipeline(param_grid, pipelines, X_train, y_train, X_test, y_test, X_musculus=None, y_musculus=None):

    results = {}

    for name, pipeline in tqdm(pipelines.items()):

        with mlflow.start_run(run_name=name):

            print(f"\n{'='*70}")
            print(f"Testando: {name}")
            print('='*70)
            
            # search = BayesSearchCV(
            #     estimator=pipeline,
            #     search_spaces=param_grid,
            #     scoring='roc-auc',
            #     cv=5,
            #     n_jobs=-1,
            #     n_iter=30,
            #     random_state=seed
            # )

            search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring='roc_auc',
                cv=5,
                n_jobs=-1,
                random_state=__SEED__
            )

            search.fit(X_train, y_train)

            best_model = search.best_estimator_

            # y_proba_val = best_model.predict_proba(X_test)[:, 1]

            # thresholds = np.arange(0.1, 0.95, 0.01)

            # best_threshold = 0.5
            # best_f1 = 0

            # for t in thresholds:

            #     y_pred_temp = (y_proba_val >= t).astype(int)

            #     f1 = f1_score(y_test, y_pred_temp)

            #     # evitar threshold que prevê tudo negativo
            #     if y_pred_temp.sum() > 0:

            #         if f1 > best_f1:
            #             best_f1 = f1
            #             best_threshold = t

            # print(f"Best threshold: {best_threshold:.2f}")
            # print(f"Best f1: {best_f1:.4f}")

            # selector = best_model.named_steps.get('selector')

            # mlflow.log_param(
            #     "feature_selection",
            #     type(selector).__name__ if selector else "None"
            # )

            # if hasattr(selector, 'k'):
            #     mlflow.log_param("n_features_selected", selector.k)

            # y_proba_test = best_model.predict_proba(X_test)[:, 1]

            # y_pred_test = (y_proba_test >= best_threshold).astype(int)
            y_pred_test = best_model.predict(X_test)
            test_f1 = f1_score(y_test, y_pred_test)

            report_test = classification_report(y_test, y_pred_test, output_dict=True)

            # métricas por classe
            # Classe 0
            mlflow.log_metric("test_precision_class_0", report_test['0']['precision'])
            mlflow.log_metric("test_recall_class_0", report_test['0']['recall'])
            mlflow.log_metric("test_f1_class_0", report_test['0']['f1-score'])

            # Classe 1
            mlflow.log_metric("test_precision_class_1", report_test['1']['precision'])
            mlflow.log_metric("test_recall_class_1", report_test['1']['recall'])
            mlflow.log_metric("test_f1_class_1", report_test['1']['f1-score'])
            

            mlflow.log_param("pipeline_name", name)
            mlflow.log_params(search.best_params_)

            sampler = best_model.named_steps.get('sampler')
            mlflow.log_param("sampler", type(sampler).__name__ if sampler else "None")

            classifier = best_model.named_steps.get('classifier')
            mlflow.log_param("model", type(classifier).__name__)


            mlflow.log_metric("cv_f1", search.best_score_)
            mlflow.log_metric("test_f1", test_f1)

            if X_musculus is not None:

                y_pred_mus = best_model.predict(X_musculus)

                report = classification_report(y_musculus, y_pred_mus, output_dict=True)

                # métricas por classe
                # Classe 0
                mlflow.log_metric("mus_precision_class_0", report['0']['precision'])
                mlflow.log_metric("mus_recall_class_0", report['0']['recall'])
                mlflow.log_metric("mus_f1_class_0", report['0']['f1-score'])

                # Classe 1
                mlflow.log_metric("mus_precision_class_1", report['1']['precision'])
                mlflow.log_metric("mus_recall_class_1", report['1']['recall'])
                mlflow.log_metric("mus_f1_class_1", report['1']['f1-score'])

            # ===== SALVAR MODELO =====
            mlflow.sklearn.log_model(best_model, "model")

            results[name] = {
                'best_params': search.best_params_,
                'best_score_cv': search.best_score_,
                'test_f1': test_f1
            }

    return results

#%%
" ================= Pipeline Random Forest ================== "

pipelines = {
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

# Grid de parâmetros
rfc_bayes = {
    'classifier__max_depth': Integer(5, 10),
    'classifier__bootstrap': Categorical([True, False]),
    'classifier__criterion': Categorical(["gini", "entropy"]),
    'classifier__n_estimators': Integer(100, 300),
    #'selector__k': Integer(10, 40)
}

rfc_grid = {
    'classifier__max_depth': [5,6,7,8,9,10],
    'classifier__bootstrap': [True, False],
    'classifier__criterion': ["gini", "entropy"],
    'classifier__n_estimators': [100, 200, 300],
    #'selector__k': Integer(10, 40)
}

rfc_results = pipeline(rfc_grid, pipelines, X_train, y_train, X_test, y_test, X_musculus, y_musculus)
#rfc_model, rfc_metrics = salvar_metricas_csv(rfc_results, X_musculus, y_musculus)


#%%
"============ Pipeline XGBoost =============="""

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()

ratio = neg / pos

pipelines = {
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

# Grid de parâmetros
xgb_bayes = {
    'classifier__learning_rate': Real(0.01, 0.5, prior='log-uniform'),
    'classifier__max_depth': Integer(3, 10),
    'classifier__n_estimators': Integer(100, 500),
    #'selector__k': Integer(10, 40)
}

xgb_grid = {
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
    'classifier__max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
    'classifier__n_estimators': [100, 200, 300, 400, 500],
    #'selector__k': Integer(10, 40)
}

xgb_results = pipeline(xgb_grid, pipelines, X_train, y_train, X_test, y_test, X_musculus, y_musculus)
#%%
"============ Pipeline Gradient Boosting =============="""

pipelines = {
    'gb_baseline': ImbPipeline([
        #('selector', SelectKBest(score_func=f_classif)),
        ('classifier', GradientBoostingClassifier(random_state=seed))
    ]),

    'gb_oversample': ImbPipeline([
        ('sampler', RandomOverSampler(random_state=seed)),
        #('selector', SelectKBest(score_func=f_classif)),
        ('classifier', GradientBoostingClassifier(random_state=seed))
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

# Grid de parâmetros
gb_bayes= {
    'classifier__learning_rate': Real(0.1, 1.0, prior='log-uniform'),
    'classifier__max_depth': Integer(3, 10),
    'classifier__n_estimators': Integer(100, 500),

    #'selector__k': Integer(10, 40)
}

gb_grid= {
    'classifier__learning_rate': [0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    'classifier__max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
    'classifier__n_estimators': [100, 200, 300, 400, 500],

    #'selector__k': Integer(10, 40)
}

gb_results = pipeline(gb_grid, pipelines, X_train, y_train, X_test, y_test, X_musculus, y_musculus)
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
