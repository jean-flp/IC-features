import mlflow
import mlflow.sklearn


class MLFlowLogger:

    def log(

        self,

        run_name,

        search,

        model,

        metrics,

        fit_time

    ):

        

        #
        # parâmetros
        #

        mlflow.log_param(
            "pipeline_name",
            run_name
        )

        mlflow.log_params(
            search.best_params_
        )
        
        mlflow.log_metric(
            "fit_time",
            fit_time
        )

        #
        # sampler
        #

        sampler = model.named_steps.get("sampler")

        mlflow.log_param(

            "sampler",

            type(sampler).__name__

            if sampler

            else "None"

        )

        #
        # classificador
        #

        classifier = model.named_steps.get(

            "classifier"

        )

        mlflow.log_param(

            "classifier",

            type(classifier).__name__

        )

        #
        # CV
        #

        mlflow.log_metric(

            "cv_score",

            search.best_score_

        )

        #
        # Teste
        #

        for metric_name, value in metrics["test"].items():

            mlflow.log_metric(

                metric_name,

                value

            )

        #
        # Base externa
        #

        if "external" in metrics:

            for metric_name, value in metrics["external"].items():

                mlflow.log_metric(

                    metric_name,

                    value

                )

        #
        # Modelo
        #

        mlflow.sklearn.log_model(

            model,

            artifact_path="model"

        )