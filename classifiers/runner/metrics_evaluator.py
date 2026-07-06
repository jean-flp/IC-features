from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)


class MetricsEvaluator:

    def _evaluate_dataset(
        self,
        model,
        X,
        y,
        prefix: str
    ):

        y_pred = model.predict(X)

        metrics = {

            f"{prefix}_accuracy":
                accuracy_score(y, y_pred),

            f"{prefix}_precision":
                precision_score(
                    y,
                    y_pred,
                    zero_division=0
                ),

            f"{prefix}_recall":
                recall_score(
                    y,
                    y_pred,
                    zero_division=0
                ),

            f"{prefix}_f1":
                f1_score(
                    y,
                    y_pred,
                    zero_division=0
                ),
        }

        #
        # ROC AUC (caso exista predict_proba)
        #

        if hasattr(model, "predict_proba"):

            y_prob = model.predict_proba(X)[:, 1]

            metrics[f"{prefix}_roc_auc"] = roc_auc_score(
                y,
                y_prob
            )

        #
        # Métricas por classe
        #

        report = classification_report(
            y,
            y_pred,
            output_dict=True,
            zero_division=0
        )

        for label in ["0", "1"]:

            metrics[f"{prefix}_precision_class_{label}"] = report[label]["precision"]

            metrics[f"{prefix}_recall_class_{label}"] = report[label]["recall"]

            metrics[f"{prefix}_f1_class_{label}"] = report[label]["f1-score"]

        return metrics

    def evaluate(

        self,

        model,

        X_test,

        y_test,

        X_external=None,

        y_external=None

    ):

        metrics = {

            "test":

                self._evaluate_dataset(

                    model,

                    X_test,

                    y_test,

                    "test"

                )

        }

        if (

            X_external is not None

            and

            y_external is not None

        ):

            metrics["external"] = self._evaluate_dataset(

                model,

                X_external,

                y_external,

                "external"

            )

        return metrics