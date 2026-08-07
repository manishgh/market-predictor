import warnings
from pathlib import Path

from market_predictor.edge_rebuild.intraday_training import load_intraday_training_config, train_intraday_edge_candidate

if __name__ == "__main__":
    dataset_dir = Path("data/features/edge_rebuild_intraday_dataset_causal_20260804_v3")
    output_dir = Path("data/models/edge_rebuild_intraday_candidate_v3")
    config = load_intraday_training_config(Path("configs/edge_rebuild_intraday_training.toml"))
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print("Starting Intraday V3 candidate training...")
        result = train_intraday_edge_candidate(
            dataset_authority_directory=dataset_dir,
            output_directory=output_dir,
            config=config
        )
        print(f"Intraday V3 Training Result: {result.status}")
