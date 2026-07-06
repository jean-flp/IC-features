from sklearn.model_selection import GridSearchCV
from hyperparameter_search.BaseSearch import BaseSearch

class Grid(BaseSearch):


    def create_search(
        self,
        estimator,
        param_space,
        cv,
        scoring,
        seed
    ):

        return GridSearchCV(
            estimator=estimator,
            param_grid=param_space,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )