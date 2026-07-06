from hyperparameter_search.optuna_search import OptunaSearch
from hyperparameter_search.BaseSearch import BaseSearch

class Bayes(BaseSearch):

    def create_search(
        self,
        estimator,
        param_space,
        cv,
        scoring,
        seed
    ):

        return OptunaSearch(

            estimator=estimator,

            search_space=param_space,

            cv=cv,

            scoring=scoring,

            random_state=seed

        )