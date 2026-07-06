from runner.search_engine import SearchEngine
from runner.metrics_evaluator import MetricsEvaluator
from runner.mlflow_logger import MLFlowLogger
from tqdm import tqdm


class PipelineRunner:

    def __init__(
        self,
        search_strategy: str,
        cv: int,
        scoring: str,
        seed: int,
    ):

        self.search_engine = SearchEngine()
        self.metrics = MetricsEvaluator()
        self.logger = MLFlowLogger()

        self.search_strategy = search_strategy
        self.cv = cv
        self.scoring = scoring
        self.seed = seed

    def run(
        self,
        model_name: str,
        pipelines: dict,
        X_train,
        y_train,
        X_test,
        y_test,
        X_external=None,
        y_external=None,
    ):
        results = {}

        for name, estimator in tqdm(pipelines.items()):
            
            print(f"ESTOU NO PASSO DO SEARCH PARA: {name} e {estimator}")
            search = self.search_engine.create(

                model_name=model_name,

                strategy=self.search_strategy,

                estimator=estimator,

                cv=self.cv,

                scoring=self.scoring,

                seed=self.seed

            )
            print(f"ESTOU NO PASSO DO fit PARA: {name} e {estimator}")
            search.fit(
                X_train,
                y_train
            )
            print(f"ESTOU NO PASSO DO best_model PARA: {name} e {estimator}")
            best_model = search.best_estimator_
            print(f"ESTOU NO PASSO DO metrics PARA: {name} e {estimator}")
            metrics = self.metrics.evaluate(

                model=best_model,

                X_test=X_test,

                y_test=y_test,

                X_external=X_external,

                y_external=y_external

            )
            print(f"ESTOU NO PASSO DO mlflow logger PARA: {name} e {estimator}")
            self.logger.log(

                run_name=name,

                search=search,

                model=best_model,

                metrics=metrics

            )
            print(f"ESTOU NO PASSO Da atribuição results PARA: {name} e {estimator}")
            results[name] = {

                "best_model": best_model,

                "best_params": search.best_params_,

                "best_score": search.best_score_,

                "metrics": metrics

            }
        return results