import mlflow
import torch
import os
import argparse

def download_model_as_pt(run_id, output_path, model_name, state_only_dict=True):
    """
    Downloads a PyTorch model from MLflow using run_id
    and saves it as a .pt file.

    Args:
        run_id (str): MLflow run ID.
        output_path (str): Directory where .pt will be saved.
        model_name (str): Name of output model file.
        state_dict_only (bool): Save only state_dict or full model.
    """
    model_uri = f"runs:/{run_id}/best_model"

    artifact_path = "model"   # Change if your registered artifact folder is different
    print(f"Downloading model artifact from MLflow run: {run_id}")

    # Load model from MLflow
    # model = mlflow.pytorch.load_model(f"runs:/{run_id}/{artifact_path}")
    model = mlflow.pytorch.load_model(model_uri)

    # Construct final file path
    final_path = os.path.join(output_path, f"{model_name}.pt")

    # Make sure directory exists
    os.makedirs(output_path, exist_ok=True)

    # Save either full model or only the state dict
    if state_only_dict:
        torch.save(model.state_dict(), final_path)
        print(f"Saved model state_dict to: {final_path}")
    else:
        torch.save(model, final_path)
        print(f"Saved full model to: {final_path}")

    return final_path



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True,
                        help="MLflow run ID to load the trained model")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name for saved .pt model")
    args = parser.parse_args()

    output_dir = "/home/smartan5070/Downloads/SlowfastTrainer-main/downloaded_models"

    download_model_as_pt(
        run_id=str(args.run_id),
        output_path=str(output_dir),
        model_name=args.model_name
    )
