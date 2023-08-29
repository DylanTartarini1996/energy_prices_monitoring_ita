import mlflow
import mlflow.xgboost

from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

def register_model(metric: str, threshold: float, experiment_name: str, model_name: str)-> None:
    """
        Register a model from the last experiment run if its metric is below a certain threshold
    """
    try:
        client = MlflowClient()
        # Fetch the latest experiment
        experiment = client.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        # Get the latest run
        latest_run = client.search_runs(experiment_id, order_by=["start_time desc"], max_results=1)[0]
        # Get the MAPE metric
        metric_val = latest_run.data.metrics[metric]
        print(f"Latest run: run_id={latest_run.info.run_id}, {metric}={metric_val}")
        if metric_val < threshold:
            # Register the model
            model_uri = f"runs:/{latest_run.info.run_id}/XGBoost"
            registered_model = client.create_registered_model(model_name)
            client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=latest_run.info.run_id
            )
            print(f"Model registered: name={model_name}")
        else:
            print(f"MAPE is above the threshold ({threshold}), model not registered.")
    except MlflowException as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    experiment_name = "xgboost_predictor"
    model_name = "XGBoost2"
    metric="MAPE"
    threshold = 0.2
    register_model(metric=metric, 
                   threshold=threshold, 
                   experiment_name=experiment_name, 
                   model_name=model_name)