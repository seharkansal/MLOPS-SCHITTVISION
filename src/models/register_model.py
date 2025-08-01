import mlflow
from mlflow.tracking import MlflowClient
import dagshub
import warnings
import os

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "seharkansal"
repo_name = "MLOPS-SCHITTVISION"

mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)

experiment_name = "SchittVision-GPT-Inference"  # Your inference pipeline experiment name
emotion_model_name = "emotion_model"
gpt_model_name = "gpt_model"

def get_latest_run_id(experiment_name):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise Exception(f"No experiment found with name: {experiment_name}")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    if not runs:
        raise Exception("No runs found in the experiment.")

    return runs[0].info.run_id

def register_model(model_name, model_uri):
    client = MlflowClient()
    try:
        model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        print(f"✅ Registered {model_name} version {model_version.version} to Staging.")
    except Exception as e:
        print(f"❌ Error registering {model_name}: {e}")

def main():
    run_id = get_latest_run_id(experiment_name)

    # Both models saved under these artifact paths in the run
    emotion_model_uri = f"runs:/{run_id}/emotion_model"
    gpt_model_uri = f"runs:/{run_id}/gpt_model"

    register_model(emotion_model_name, emotion_model_uri)
    register_model(gpt_model_name, gpt_model_uri)

if __name__ == "__main__":
    main()
