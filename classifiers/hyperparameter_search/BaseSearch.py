from abc import ABC, abstractmethod

class BaseSearch(ABC):

    @abstractmethod
    def create_search(
        self,
        estimator,
        param_space,
        cv,
        scoring,
        seed
    ):
        pass