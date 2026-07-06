from hyperparameter_search.grid import Grid
from hyperparameter_search.randomized import Randomized
#Optuna é o bayes
from hyperparameter_search.bayes import Bayes
from hyperparameter_search.BaseSearch import BaseSearch


search_dict = {
    "grid":Grid,
    "randomized_search": Randomized,
    "optuna": Bayes
}

class SearchContext:
    def __init__(self):
        self.strategy: BaseSearch = None

    def setStrategy(self, search_name: str):
        strategy_class = search_dict.get(search_name)

        if strategy_class is None:
            raise ValueError(f"Modelo não suportado: {search_name}")

        self.strategy_class = strategy_class
        self.search_name = search_name

        return self
    def create_search(
            self,
            estimator,
            param_space,
            cv,
            scoring,
            seed
        ):
        strategy = self.strategy_class()

        return strategy.create_search(
            estimator=estimator,
            param_space=param_space,
            cv=cv,
            scoring=scoring,
            seed=seed
        )