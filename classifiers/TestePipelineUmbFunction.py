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

from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

import os
from tqdm import tqdm

import mlflow

#%%
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment(experiment_id= 7)

#%%
pasta_atual = os.getcwd()
print(pasta_atual)

df_cel = pd.read_csv(os.path.join(pasta_atual, 'Todas_Features/Features_cel.csv'))
df_sce = pd.read_csv(os.path.join(pasta_atual, 'Todas_Features/Features_sce.csv'))
df_dme = pd.read_csv(os.path.join(pasta_atual, 'Todas_Features/Features_dme.csv'))

df_mus = pd.read_csv(os.path.join(pasta_atual, 'Todas_Features/Features_mus.csv'))

df_man = pd.read_csv(os.path.join(pasta_atual, 'Todas_Features/Features_man.csv'))

df = pd.concat([df_cel, df_sce, df_dme], ignore_index=True)

#%%

""" Início ML """
# Separação em conjuntos de treino e teste
X = df.drop(['Locus','IsEssential', 'Sequence'], axis=1)

y = df['IsEssential']
test_size = 0.2
seed = 0
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=test_size, 
                                                    random_state=seed, 
                                                    stratify=y)
# Dados mansoni
df_mansoni = df_man.set_index('Locus')
X_mansoni = df_mansoni.drop(['Sequence'], axis=1)

# Dados musculus
df_musculus = df_mus.set_index('Locus')
X_musculus = df_musculus.drop(['Sequence', 'IsEssential'], axis=1)
y_musculus = df_mus['IsEssential']

feature_order = [
        'Sequence_Length', 'Aromaticity', 'Sec_Struct_Helix', 'Sec_Struct_Turn', 'Sec_Struct_Sheet',
        'Percent_A', 'Percent_C', 'Percent_D', 'Percent_E', 'Percent_F', 'Percent_G', 'Percent_H',
        'Percent_I', 'Percent_K', 'Percent_L', 'Percent_M', 'Percent_N', 'Percent_P', 'Percent_Q',
        'Percent_R', 'Percent_S', 'Percent_T', 'Percent_V', 'Percent_W', 'Percent_Y', 'IsoelectricPoint',
        'Tiny_Number', 'Small_Number', 'Aliphatic_Number', 'Aromatic_Number', 'Non-polar_Number',
        'Polar_Number', 'Charged_Number', 'Basic_Number', 'Acidic_Number', 'Local Average Connectivity',
        'Density of Maximum neighborhood Component', 'Topology Potential', 'Edge Clustering Coefficient',
        'DegreeCentrality', 'EigenvectorCentrality', 'BetweennessCentrality', 'ClosenessCentrality', 'Clustering'
]

emboss = [
        'Tiny_Number', 'Small_Number', 'Aliphatic_Number', 'Aromatic_Number', 'Non-polar_Number',
        'Polar_Number', 'Charged_Number', 'Basic_Number', 'Acidic_Number'
]

X_musculus = X_musculus[feature_order]

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
        ('classifier', RandomForestClassifier(class_weight='balanced', random_state=seed))
    ]),

    'rf-undersample': ImbPipeline([
        ('sampler', RandomUnderSampler(random_state=seed)),
        #('selector', SelectKBest(score_func=f_classif)),
        ('classifier', RandomForestClassifier(random_state=seed))
    ]),

    'rf-oversample': ImbPipeline([
        ('sampler', RandomOverSampler(random_state=seed)),
        #('selector', SelectKBest(score_func=f_classif)),
        ('classifier', RandomForestClassifier(random_state=seed))
    ]),
    
    'rf-smote': ImbPipeline([
        ('sampler', SMOTE(random_state=seed)),
        #('selector', SelectKBest(score_func=f_classif)),
        ('classifier', RandomForestClassifier(random_state=seed))
    ]),
    
    'rf-smoteenn': ImbPipeline([
        ('sampler', SMOTEENN(random_state=seed)),
        #('selector', SelectKBest(score_func=f_classif)),
        ('classifier', RandomForestClassifier(random_state=seed))
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
        ('classifier', XGBClassifier(booster='gbtree', scale_pos_weight=ratio, verbosity=0, random_state=seed))
    ]),

    'xgb_undersample': ImbPipeline([
        ('sampler', RandomUnderSampler(random_state=seed)),
        #('selector', SelectKBest(score_func=f_classif)),
        ('classifier', XGBClassifier(booster='gbtree', verbosity=0, random_state=seed))
    ]),

    'xgb_oversample': ImbPipeline([
        ('sampler', RandomOverSampler(random_state=seed)),
        #('selector', SelectKBest(score_func=f_classif)),
        ('classifier', XGBClassifier(booster='gbtree', verbosity=0, random_state=seed))
    ]),
    
    'xgb_smote': ImbPipeline([
        ('sampler', SMOTE(random_state=seed)),
        #('selector', SelectKBest(score_func=f_classif)),
        ('classifier', XGBClassifier(booster='gbtree', verbosity=0, random_state=seed))
    ]),
    
    'xgb_smoteenn': ImbPipeline([
        ('sampler', SMOTEENN(random_state=seed)),
        #('selector', SelectKBest(score_func=f_classif)),
        ('classifier', XGBClassifier(booster='gbtree', verbosity=0, random_state=seed))
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
        ('sampler', RandomUnderSampler(random_state=seed)),
        #('selector', SelectKBest(score_func=f_classif)),
        ('classifier', GradientBoostingClassifier(random_state=seed))
    ]),

    'gb_smote': ImbPipeline([
        ('sampler', SMOTE(random_state=seed)),
        #('selector', SelectKBest(score_func=f_classif)),
        ('classifier', GradientBoostingClassifier(random_state=seed))
    ]),
    
    'gb_smoteenn': ImbPipeline([
        ('sampler', SMOTEENN(random_state=seed)),
        #('selector', SelectKBest(score_func=f_classif)),
        ('classifier', GradientBoostingClassifier(random_state=seed))
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
