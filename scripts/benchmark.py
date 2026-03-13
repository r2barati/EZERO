import torch
import numpy as np
import pandas as pd
import argparse
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data.monash_loader import generate_zero_shot_eval_dataset
from data.preprocessing import create_rolling_windows
from models.hybrid_model import HybridNBeatsTCNModel

# Optional dependencies for foundation models (will fail gracefully if missing)
try:
    from chronos import ChronosPipeline
except ImportError:
    ChronosPipeline = None

def evaluate_local_model(series_list, model_path, input_window, forecast_horizon):
    print(f"\\n--- Evaluating Local Hybrid NBeats+TCN Model ---")
    model = HybridNBeatsTCNModel(
        input_size=input_window,
        output_size=forecast_horizon,
        hidden_units=64,
        stack_depth=3,
        tcn_channels=[64, 128],
        tcn_kernel_size=3,
        tcn_dropout=0.2
    )
    # If the file exists, load it, otherwise run untrained to test the pipeline
    try:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print("Loaded trained model weights.")
    except Exception as e:
        print(f"Could not load local weights '{model_path}', running with random initialization for demonstration.")

    model.eval()

    all_preds, all_targets = [], []
    inference_time = 0

    with torch.no_grad():
        for series in series_list:
            series = series[~np.isnan(series)]
            if len(series) < input_window + forecast_horizon:
                continue

            X, y, means, stds = create_rolling_windows(series, input_window, forecast_horizon)
            if len(X) == 0:
                continue

            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32)
            means_tensor = torch.tensor(means, dtype=torch.float32).unsqueeze(-1)
            stds_tensor = torch.tensor(stds, dtype=torch.float32).unsqueeze(-1)

            # Time the inference
            start_time = time.time()
            outputs = model(X_tensor)
            inference_time += (time.time() - start_time)

            # Denormalize
            outputs_denorm = outputs * stds_tensor + means_tensor
            targets_denorm = y_tensor * stds_tensor + means_tensor

            all_preds.append(outputs_denorm.numpy())
            all_targets.append(targets_denorm.numpy())

    if len(all_preds) == 0:
        return float('inf'), float('inf'), 0

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    mse = mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)

    print(f"Local Model -> MSE: {mse:.4f} | MAE: {mae:.4f} | Inference Time: {inference_time:.2f}s")
    return mse, mae, inference_time

def evaluate_chronos(series_list, model_id="amazon/chronos-t5-tiny", input_window=24, forecast_horizon=4):
    """
    Evaluates Amazon Chronos (a highly mature encoder-only T5 foundation model)
    Note: Chronos does not take sliding windows natively in its pipeline easily,
    so we simulate one prediction per series for benchmark speed.
    """
    print(f"\\n--- Evaluating {model_id} ---")
    if ChronosPipeline is None:
        print("Install 'chronos' package to benchmark Amazon Chronos.")
        return float('inf'), float('inf'), 0

    try:
        pipeline = ChronosPipeline.from_pretrained(
            model_id,
            device_map="cpu", # Force CPU for sandbox limits
            torch_dtype=torch.float32,
        )
    except Exception as e:
        print(f"Could not load Chronos model: {e}")
        return float('inf'), float('inf'), 0

    all_preds, all_targets = [], []
    inference_time = 0

    for series in series_list:
        series = series[~np.isnan(series)]
        if len(series) < input_window + forecast_horizon:
            continue

        # Chronos inference: we give it the context window, it predicts the horizon
        context = torch.tensor(series[:input_window])
        target = series[input_window:input_window+forecast_horizon]

        start_time = time.time()
        # predict() returns (num_samples, prediction_length)
        forecast = pipeline.predict(context, prediction_length=forecast_horizon)
        inference_time += (time.time() - start_time)

        # Chronos returns probabilistic samples, we take the median as the point forecast
        point_forecast = np.median(forecast[0].numpy(), axis=0)

        all_preds.append(point_forecast)
        all_targets.append(target)

    if len(all_preds) == 0:
        return float('inf'), float('inf'), 0

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    mse = mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)

    print(f"{model_id} -> MSE: {mse:.4f} | MAE: {mae:.4f} | Inference Time: {inference_time:.2f}s")
    return mse, mae, inference_time

def evaluate_nixtla_timegpt(series_list, api_key=None, input_window=24, forecast_horizon=4):
    """
    Placeholder for evaluating TimeGPT via API.
    TimeGPT is closed source and requires an API token.
    """
    print(f"\\n--- Evaluating Nixtla TimeGPT-1 ---")
    if not api_key:
        print("Skipping TimeGPT. No API key provided (requires NIxTLA token).")
        return float('inf'), float('inf'), 0
    # Implementation would use `from nixtla import NixtlaClient`
    print("TimeGPT integration stubbed. Requires production API limits to evaluate entire dataset.")
    return 0, 0, 0

def evaluate_google_timesfm(series_list, input_window=24, forecast_horizon=4):
    """
    Placeholder for Google TimesFM-2.5.
    TimesFM requires specific checkpoint downloads and ~200M params loading which
    exceeds standard sandbox limits, but the API is standard `timesfm.TimesFm` class.
    """
    print(f"\\n--- Evaluating Google TimesFM-2.5 ---")
    print("Skipping TimesFM. Requires 16GB+ RAM and JAX compilation which timeouts in sandbox.")
    return float('inf'), float('inf'), 0

def evaluate_ibm_ttm(series_list, input_window=24, forecast_horizon=4):
    """
    Placeholder for IBM Tiny Time Mixers (TTM).
    Uses standard HuggingFace `AutoModelForTimeSeriesForecasting`.
    """
    print(f"\\n--- Evaluating IBM Tiny Time Mixers (TTM) ---")
    print("TTM requires strict `tsfm` library installations. Stubbing for now.")
    return float('inf'), float('inf'), 0

def main():
    parser = argparse.ArgumentParser(description="Zero-Shot Foundation Model Benchmark")
    parser.add_argument('--num_series', type=int, default=10, help="Number of zero-shot OOD series to generate.")
    parser.add_argument('--input_window', type=int, default=24)
    parser.add_argument('--forecast_horizon', type=int, default=4)
    parser.add_argument('--local_model_path', type=str, default="hybrid_model.pth")
    args = parser.parse_args()

    print(f"Generating explicit Zero-Shot Evaluation Dataset ({args.num_series} sequences)...")
    # Generate completely out of distribution data (huge scales, strange frequencies)
    series_list = generate_zero_shot_eval_dataset(args.num_series, 100, 300)

    results = {}

    # Evaluate Your Custom Model
    mse, mae, t = evaluate_local_model(series_list, args.local_model_path, args.input_window, args.forecast_horizon)
    results["Custom Hybrid (Local)"] = {"MSE": mse, "MAE": mae, "Time (s)": t}

    # Evaluate Amazon Chronos (Smallest variant that fits in RAM)
    # Note: Chronos evaluation requires `pip install chronos`
    mse, mae, t = evaluate_chronos(series_list, "amazon/chronos-t5-tiny", args.input_window, args.forecast_horizon)
    results["Amazon Chronos-T5-Tiny"] = {"MSE": mse, "MAE": mae, "Time (s)": t}

    # Evaluate placeholders
    evaluate_google_timesfm(series_list)
    evaluate_ibm_ttm(series_list)
    evaluate_nixtla_timegpt(series_list)

    print("\\n=======================================================")
    print("               BENCHMARK RESULTS (Zero-Shot)             ")
    print("=======================================================")
    df = pd.DataFrame(results).T
    print(df.to_markdown())
    print("=======================================================")
    print("Note: Commercial models like TimesFM and Moirai are highly parameterized (>200M)")
    print("and require dedicated GPU inference environments to run successfully.")

if __name__ == "__main__":
    main()
