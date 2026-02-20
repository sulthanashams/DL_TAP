import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress INFO and WARNING messages
from single.single_helpers_network_test import * 
from single.single_transformer import *
from single.single_params_network_test import *
from single.single_dataset_network_test import *
from plotting import *
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras import Input, Model 
from tensorflow.keras.losses import MeanSquaredError
from time import time
import concurrent.futures
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os


import tensorflow as tf


# List all physical GPUs
gpus = tf.config.list_physical_devices('GPU')
print("Available GPUs:", gpus)

# Assign this session to GPU 1 (the second GPU)
if len(gpus) > 1:
    tf.config.set_visible_devices(gpus[1], 'GPU')
    print("✅ Using GPU 1 for this session")
else:
    print("⚠️ Only one GPU detected, using default GPU 0")

# Optional: allow memory growth to prevent OOM errors
tf.config.experimental.set_memory_growth(gpus[1], True)

unique_set_train = read_file(UNIQUE_PATH_DICT_train)


def load_data(files, unique_set):
    X, Y, scalers = [], [], []

    for file_name in tqdm(files):
        x, y, scaler = generate_xy(file_name, unique_set)
        X.append(x)
        Y.append(y)
        scalers.append(scaler)

    # Use the first scaler as training scaler (they should be consistent)
    train_scaler = scalers[0]

    X = tf.stack(X, axis=0)
    Y = tf.stack(Y, axis=0)
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.shuffle(buffer_size=len(X)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return dataset, train_scaler
    
    # --- helper ---
def predict_and_plot(model, TEST_FILES, unique_set, train_scaler, link_miss=0):
    """Predict test cases and return full aggregated metrics."""
    print("Preparing test data...")
    test_files = TEST_FILES
    print(test_files)
    test_data_loader, scalers = get_test_set(test_files, unique_set)

    print("Start predicting...")
    pred_tensor = predict_withScaler(model, test_data_loader, scalers, device)

    # Aggregate all metrics
    result, Link_flow, Path_flow, pred_mean_cost, UE_mean_path_cost, p, s = \
        aggregate_result_single(pred_tensor, test_files)

    # --- print summary ---
    print(result)
    print("Avg path cost:", round(np.mean(pred_mean_cost), 4), "mins")
    print("Prediction average delay:", round(p, 4), "mins =",
          round(p / np.mean(pred_mean_cost) * 100, 2), "%")
    print("Solution average delay:", round(s, 2), "mins =",
          round(s / np.mean(UE_mean_path_cost) * 100, 2), "%")
    print(f"Difference: {round(p - s, 4)} mins")

    return result, Link_flow, Path_flow, pred_mean_cost, UE_mean_path_cost, p, s




def evaluate_multiple_variants(model, base_variant_dir, train_scaler):
    """
    Evaluate multiple topology variants (each with its own unique_set and OD test files),
    compute metrics, and visualize per-variant mean prediction delay (box plot).
    """
    all_mae_link, all_rmse_link, all_mape_link = [], [], []
    all_mae_path, all_rmse_path, all_mape_path = [], [], []
    all_pred_delay, all_sol_delay = [], []
    all_pred_delay_pct, all_sol_delay_pct = [], []
    variant_labels = []  # to label each topology variant on the plot

    for i in range(1, 21):  # assuming up to 20 variants
        variant_folder = os.path.join(base_variant_dir, f"Variant_{i}")
        if not os.path.exists(variant_folder):
            print(f" Skipping missing folder: {variant_folder}")
            continue

        print(f"\n🔹 Evaluating SiouxFalls Topology Variant #{i} ...")

        unique_path_file = os.path.join(variant_folder, "unique_paths.pkl")
        if not os.path.exists(unique_path_file):
            print(f" Missing unique_set file for variant {i}: {unique_path_file}")
            continue

        unique_set_variant = read_file(unique_path_file)
        test_files = load_files_from_folders(variant_folder, max_files=50)

        # --- Run model prediction ---
        result, _, _, pred_mean_cost, UE_mean_path_cost, p, s = predict_and_plot(
            model, test_files, unique_set_variant, train_scaler, link_miss=0
        )

        # --- Record errors 
        all_mae_link.append(result.loc[result['Indicator'] == 'MAE', 'Link flow'].values[0])
        all_rmse_link.append(result.loc[result['Indicator'] == 'RMSE', 'Link flow'].values[0])
        all_mape_link.append(result.loc[result['Indicator'] == 'MAPE', 'Link flow'].values[0])
        all_mae_path.append(result.loc[result['Indicator'] == 'MAE', 'Path flow'].values[0])
        all_rmse_path.append(result.loc[result['Indicator'] == 'RMSE', 'Path flow'].values[0])
        all_mape_path.append(result.loc[result['Indicator'] == 'MAPE', 'Path flow'].values[0])

        all_pred_delay.append(p)
        all_sol_delay.append(s)
   
        # 🆕 store percentages for each variant
        pred_pct = (p / np.mean(pred_mean_cost)) * 100
        sol_pct = (s / np.mean(UE_mean_path_cost)) * 100
        all_pred_delay_pct.append(pred_pct)
        all_sol_delay_pct.append(sol_pct)
   
    # --- aggregate across all variants ---
    rows = ['MAE', 'RMSE', 'MAPE']
    result_mean = pd.DataFrame({
        'Indicator': rows,
        'Link flow': [np.mean(all_mae_link), np.mean(all_rmse_link), np.mean(all_mape_link)],
        'Path flow': [np.mean(all_mae_path), np.mean(all_rmse_path), np.mean(all_mape_path)]
    })
   
    print("\n📊 === Final Mean Results Across Topological Variants ===")
    print(result_mean)
    print("Mean Prediction Delay:", round(np.mean(all_pred_delay), 3), "mins")
    print("Mean Solution Delay:", round(np.mean(all_sol_delay), 3), "mins")
    print("Mean Difference:", round(np.mean(np.array(all_pred_delay) - np.array(all_sol_delay)), 3), "mins")

    # 🆕 percentage delays
    print("Mean Prediction Delay (%):", round(np.mean(all_pred_delay_pct), 3), "%")
    print("Mean Solution Delay (%):", round(np.mean(all_sol_delay_pct), 3), "%")
    print("Mean Difference (%):", round(np.mean(np.array(all_pred_delay_pct) - np.array(all_sol_delay_pct)), 3), "%")

    # --- 🆕 Box plot of mean prediction delay only ---
    plt.figure(figsize=(8, 5))
    sns.boxplot(y=all_pred_delay, color='cornflowerblue', width=0.4)
    sns.stripplot(y=all_pred_delay, color='darkblue', size=6, jitter=True, alpha=0.7)
    plt.title("Distribution of Mean Prediction Delay Across Topology Variants", fontsize=12)
    plt.ylabel("Prediction Delay (minutes)", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return result_mean
    
MODEL_DIR = r"E:\sshams\DL_TAP\Path_Flow_Prediction-main\Model\transformer_full_model_size"

def main():
    # ----------------------------------------------------------
    # LOAD DATA
    # ----------------------------------------------------------
    files = load_files_from_folders(FOLDERS, max_files=DATA_SIZE)
    train_files, val_files, test_files = split_dataset(files, TRAIN_RATE, VAL_RATE)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_train = executor.submit(load_data, train_files, unique_set_train)
        future_val = executor.submit(load_data, val_files, unique_set_train)
        train_data_loader, train_scaler = future_train.result()
        val_data_loader, _ = future_val.result()
    
    # ----------------------------------------------------------
    # MODEL: Load existing full model or train a new one
    # ----------------------------------------------------------
    if os.path.exists(MODEL_DIR):
        print(f"⚙️ Found existing model in '{MODEL_DIR}', loading it...")
        model = tf.keras.models.load_model(MODEL_DIR, compile=False)
    else:
        print("🚀 No saved model found — training a new Transformer model...")
        model = Transformer(input_dim=input_dim, output_dim=output_dim,
                            d_model=d_model, E_layer=E_layer, D_layer=D_layer,
                            heads=heads, dropout=dropout, l2_reg=l2_reg)
        loss_fn = MeanSquaredError()
        optimizer = Adam(learning_rate=learning_rate, clipvalue=1.0, decay=l2_reg)

        start = time()
        model, train_loss, val_loss = model.fit(train_data_loader, val_data_loader,
                                                optimizer, loss_fn, epochs, device)
        end = time()
        print("✅ Finish training in:", round((end-start)/3600, 2), "hours")

        # Plot and save training history
        plot_loss(train_loss, val_loss, epochs, TRAIN_HISTORY_TITLE)

      
       # # # ✅ Save full model
       # # ----------------------------------------------------------
       # # ✅ Save or Rebuild Model 
       # # ----------------------------------------------------------


       # # 1️⃣ Ensure model folder exists
       #  if not os.path.exists(MODEL_DIR):
       #     print(f"⚠️ Model folder not found at {MODEL_DIR}. Saving current model...")
       #     os.makedirs(os.path.dirname(MODEL_DIR), exist_ok=True)
       #     model.save(MODEL_DIR, save_format="tf")
       #     print(f"✅ Full Transformer model saved to '{MODEL_DIR}'")
       #  else:
       #     print(f"⚙️ Found existing model at '{MODEL_DIR}', loading it...")
       #     model = tf.keras.models.load_model(MODEL_DIR, compile=False)

       # # 2️⃣ Rebuild flexible-input wrapper
       #  print("🔧 Rebuilding model with flexible input shapes...")
       #  try:
       #     x_input = Input(shape=(None, input_dim), name="x")
       #     y_input = Input(shape=(None, output_dim), name="y")

       #     # Use functional call
       #     output = model(x_input, y_input)

       #     keras_model = Model(inputs=[x_input, y_input], outputs=output)

       #     # 3️⃣ Save new flexible version
       #     flex_path = MODEL_DIR + "_flex"
       #     keras_model.save(flex_path, include_optimizer=False)
       #     print(f"✅ Saved flexible-input model to '{flex_path}'")

       #  except Exception as e:
       #     print(f"❌ Error rebuilding flexible model: {e}")
       #     print("ℹ️ Proceeding with loaded model as-is (fixed input shape).")
      

          # --- inside your main() after model is loaded and trained ---
    base_variant_dir = r"E:\sshams\DL_TAP\Path_Flow_Prediction-main\Generate_data\SiouxFalls\Size_Variants_25_test/UE_Output"
       
    print("\n🚀 Evaluating model on 10 SiouxFalls topology variants...")
    final_result = evaluate_multiple_variants(model, base_variant_dir, train_scaler)


if __name__ == "__main__":
    main()
