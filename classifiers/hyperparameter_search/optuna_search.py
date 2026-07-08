import optuna
from sklearn.model_selection import cross_val_score
from hyperparameter_search.BaseSearch import BaseSearch


class OptunaSearch:

    def __init__(self):
        self.estimator = None
        self.search_space = None
        self.cv = None
        self.scoring = None
        self.n_trials = 50
        self.random_state = 42
        self.n_jobs = -1

        self.study = None
        self.best_params_ = None
        self.best_score_ = None
        self.best_estimator_ = None

    def create_search(
        self,
        estimator,
        param_space,
        cv,
        scoring,
        seed,
    ):
        self.estimator = estimator
        self.search_space = param_space
        self.cv = cv
        self.scoring = scoring
        self.random_state = seed

        return self

    def _objective(self, trial):

        params = self.search_space(trial)

        self.estimator.set_params(**params)

        score = cross_val_score(
            self.estimator,
            self.X,
            self.y,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
        ).mean()

        return score

    def fit(self, X, y):

        self.X = X
        self.y = y

        sampler = optuna.samplers.TPESampler(
            seed=self.random_state
        )

        self.study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
        )

        self.study.optimize(
            self._objective,
            n_trials=self.n_trials,
        )

        self.best_params_ = self.study.best_params
        self.best_score_ = self.study.best_value

        self.best_estimator_ = self.estimator.set_params(
            **self.best_params_
        )

        self.best_estimator_.fit(X, y)

        return self