from optuna.trial import Trial

def random_forest_optuna(trial: Trial) -> dict:

    return {

        "classifier__max_depth":
            trial.suggest_int(
                "classifier__max_depth",
                5,
                10
            ),

        "classifier__bootstrap":
            trial.suggest_categorical(
                "classifier__bootstrap",
                [True, False]
            ),

        "classifier__criterion":
            trial.suggest_categorical(
                "classifier__criterion",
                [
                    "gini",
                    "entropy"
                ]
            ),

        "classifier__n_estimators":
            trial.suggest_int(
                "classifier__n_estimators",
                100,
                300
            )
    }


def xgboost_optuna(trial: Trial) -> dict:

    return {

        "classifier__learning_rate":
            trial.suggest_float(
                "classifier__learning_rate",
                0.01,
                0.5,
                log=True
            ),

        "classifier__max_depth":
            trial.suggest_int(
                "classifier__max_depth",
                3,
                10
            ),

        "classifier__n_estimators":
            trial.suggest_int(
                "classifier__n_estimators",
                100,
                500
            )
    }



def gradient_boosting_optuna(trial: Trial) -> dict:

    return {

        "classifier__learning_rate":
            trial.suggest_float(
                "classifier__learning_rate",
                0.1,
                1.0,
                log=True
            ),

        "classifier__max_depth":
            trial.suggest_int(
                "classifier__max_depth",
                3,
                10
            ),

        "classifier__n_estimators":
            trial.suggest_int(
                "classifier__n_estimators",
                100,
                500
            )
    }



SEARCH_SPACES = {

    "random_forest": {

        "grid": {

            "classifier__max_depth": [5, 6, 7, 8, 9, 10],

            "classifier__bootstrap": [True, False],

            "classifier__criterion": [
                "gini",
                "entropy"
            ],

            "classifier__n_estimators": [
                100,
                200,
                300
            ]
        },

        "optuna": random_forest_optuna
    },

    "xgboost": {

        "grid": {

            "classifier__learning_rate": [
                0.01,
                0.05,
                0.1,
                0.2,
                0.3
            ],

            "classifier__max_depth": [
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10
            ],

            "classifier__n_estimators": [
                100,
                200,
                300,
                400,
                500
            ]
        },

        "optuna": xgboost_optuna
    },

    "gradient_boosting": {

        "grid": {

            "classifier__learning_rate": [
                0.1,
                0.2,
                0.3,
                0.5,
                0.7,
                1.0
            ],

            "classifier__max_depth": [
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10
            ],

            "classifier__n_estimators": [
                100,
                200,
                300,
                400,
                500
            ]
        },

        "optuna": gradient_boosting_optuna
    }
}