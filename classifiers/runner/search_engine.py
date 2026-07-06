from hyperparameter_search.strategy_search import SearchContext
from hyperparameter_search.search_spaces import SEARCH_SPACES

class SearchEngine:

    def create(

        self,

        model_name,

        strategy,

        estimator,

        cv,

        scoring,

        seed

    ):

        context = SearchContext().setStrategy(strategy)

        if model_name not in SEARCH_SPACES:

            raise ValueError(f"Erro no valor de modelo para {model_name}")
        
        if strategy not in SEARCH_SPACES[model_name]:

            raise ValueError(f"Erro no valor de strategy para {strategy}")

        return context.create_search(

            estimator=estimator,

            param_space=SEARCH_SPACES[model_name][strategy],

            cv=cv,

            scoring=scoring,

            seed=seed

        )