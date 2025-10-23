import os
import json
from datetime import datetime
from collector import collect_data
from predictor import Predictor
from analyser import run_analyser

# --- Configuration Loading ---
CONFIG_FILE = 'config.json'

try:
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    print(f"ERROR: Configuration file '{CONFIG_FILE}' not found. Please ensure it exists.")
    exit()

CV_SCHEME = config['MODEL']['DEFAULT_CV']
BORUTA_N_SPLITS = config['MODEL']['BORUTA_N_SPLITS']
RUN_LABEL = f"TW_{CV_SCHEME}_BorutaNest{BORUTA_N_SPLITS}_{datetime.now().strftime('%Y%m%d')}"

# --- Main Workflow ---
def main_pipeline():
    """Executes the full data collection, prediction, and analysis workflow."""
    print("=" * 70)
    print("DBS Therapeutic Window Prediction Pipeline (DBS_Prediction_TW)")
    print(f"Run Label: {RUN_LABEL}")
    print("=" * 70)

    # --- PHASE 1: DATA COLLECTION ---
    print("\n[PHASE 1] Starting Data Collection...")
    try:
        # Data structure: (X, y, S, feat_labels_plot)
        data = collect_data(config)
        
        if data is None:
            print("Collection failed or returned no data. Exiting.")
            return
    except Exception as e:
        print(f"An unexpected error occurred during data collection: {e}")
        return

    # --- PHASE 2: PREDICTION ---
    print("\n[PHASE 2] Starting Prediction and Cross-Validation...")
    
    predictor = Predictor(config)
    try:
        _ = predictor.run_prediction(data, RUN_LABEL)
    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}")
        return

    # --- PHASE 3: ANALYSIS ---
    print("\n[PHASE 3] Starting Analysis and Plotting...")
    run_analyser(config, RUN_LABEL, n_perm=config['MODEL']['PERM'])

    print("\n" + "=" * 70)
    print(f"Pipeline Finished Successfully. Results in: {os.path.join(config['PATHS']['PREDICTOR_OUTPUT'], RUN_LABEL)}")
    print("=" * 70)

# %%
if __name__ == "__main__":
    # Ensure root directories exist
    os.makedirs(config['PATHS']['RESULTS_DIR'], exist_ok=True)
    
    # Run the main process
    main_pipeline()