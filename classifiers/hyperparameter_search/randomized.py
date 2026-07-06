from sklearn.model_selection import RandomizedSearchCV
from hyperparameter_search.BaseSearch import BaseSearch

class Randomized(BaseSearch):
    def create_search(
        self,
        estimator,
        param_space,
        cv,
        scoring,
        seed
    ):

        return RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_space,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            random_state=seed
        )