import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress INFO and WARNING messages
from single.single_helpers import * 
from single.single_transformer import *
from single.single_params import *
from single.single_dataset import *
from plotting import *
from tensorflow.keras.optimizers import Adam  
from tensorflow.keras.losses import MeanSquaredError
from time import time
import concurrent.futures

import tensorflow as tf

# print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
# print(tf.config.list_logical_devices('GPU'))
# tf.debugging.set_log_device_placement(True)

# List all physical GPUs
gpus = tf.config.list_physical_devices('GPU')
print("Available GPUs:", gpus)

# Assign this session to GPU 1 (the second GPU)
if len(gpus) > 1:
    tf.config.set_visible_devices(gpus[1], 'GPU')
    print(" Using GPU 1 for this session")
else:
    print(" Only one GPU detected, using default GPU 0")

# Optional: allow memory growth to prevent OOM errors
tf.config.experimental.set_memory_growth(gpus[1], True)

# files = load_files_from_folders(FOLDERS, max_files=100)
# path_set_dict = path_encoder(files)
#unique_set = read_file(UNIQUE_PATH_DICT)

unique_set_raw = read_file(UNIQUE_PATH_DICT)

# If the loaded object is a set, convert it to a dictionary mapping each path → index
if isinstance(unique_set_raw, set):
    unique_set = {path: idx for idx, path in enumerate(unique_set_raw)}
else:
    unique_set = unique_set_raw

def load_data(files, unique_set):
    dataset = Dataset(files, unique_set)
    data_loader = dataset.to_tf_dataset(BATCH_SIZE)
    return data_loader

def predict_and_plot(model, TEST_FILES, unique_set, link_miss=0):
    #test_files = load_files_from_folders(TEST_FILES, max_files=int(DATA_SIZE*TEST_RATE))
    
    test_files=TEST_FILES           #if test files is already loaded 
    test_data_loader, scalers = get_test_set(test_files, unique_set)

    # PREDICTING 
    print("Start predicting...")
    pred_tensor = predict_withScaler(model, test_data_loader, scalers, device)

    # Calculate error
    print_result_single(pred_tensor, test_files, NODE_POSITION, f"{ERROR_TITLE} - Missing {link_miss} links")


MODEL_DIR = r"E:\sshams\DL_TAP\Path_Flow_Prediction-main\Model\transformer_full_model_3"
def main():
    # LOAD DATA
    files = load_files_from_folders(FOLDERS, max_files=DATA_SIZE)
    train_files, val_files, test_files = split_dataset(files, TRAIN_RATE, VAL_RATE)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_train = executor.submit(load_data, train_files, unique_set)
        future_val = executor.submit(load_data, val_files, unique_set)
        
        # Get the results
        train_data_loader = future_train.result()
        val_data_loader = future_val.result()
    
    # MODEL: Load existing full model or train a new one
    # ----------------------------------------------------------
    if os.path.exists(MODEL_DIR):
        print(f" Found existing model in '{MODEL_DIR}', loading it...")
        model = tf.keras.models.load_model(MODEL_DIR, compile=False)
    else:
        print(" No saved model found — training a new Transformer model...")
        model = Transformer(input_dim=input_dim, output_dim=output_dim,
                            d_model=d_model, E_layer=E_layer, D_layer=D_layer,
                            heads=heads, dropout=dropout, l2_reg=l2_reg)
        loss_fn = MeanSquaredError()
        optimizer = Adam(learning_rate=learning_rate, clipvalue=1.0, decay=l2_reg)

        start = time()
        model, train_loss, val_loss = model.fit(train_data_loader, val_data_loader,
                                                optimizer, loss_fn, epochs, device)
        end = time()
        print(" Finish training in:", round((end-start)/3600, 2), "hours")

        # Plot and save training history
        plot_loss(train_loss, val_loss, epochs, TRAIN_HISTORY_TITLE)
  

    # PLOTTING LOSS
    #plot_loss(train_loss, val_loss, epochs, TRAIN_HISTORY_TITLE)
    predict_and_plot(model, test_files, unique_set, 0)
    #predict_and_plot(model, FOLDERS, unique_set, 0)

    #predict_and_plot(model, TEST_FILES_2, unique_set, 2)
    #predict_and_plot(model, TEST_FILES_3, unique_set, 3)

    plt.show()

if __name__ == "__main__":
    main()
