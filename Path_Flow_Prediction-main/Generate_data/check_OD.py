import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# ---------------- Configuration ----------------

net_file = r"E:\sshams\DL_TAP\Path_Flow_Prediction-main\Generate_data\SiouxFalls\SiouxFalls_net.tntp"
od_dir = r"E:\sshams\DL_TAP\Path_Flow_Prediction-main\Generate_data\SiouxFalls\OD_Matrices"

# # ---------------- Helper function ----------------
def normalize(arr):
    scaler = MinMaxScaler()
    return scaler.fit_transform(np.array(arr).reshape(-1, 1))

# ---------------- Load network ----------------
net = pd.read_csv(net_file, delimiter='\t', skiprows=8)
capacity = net['capacity'].values
length = net['length'].values
fft = net['free_flow_time'].values

# ---------------- Print ranges ----------------
print("Network link attributes ranges:")
print(f"Link length: min={length.min()}, max={length.max()}, mean={length.mean():.2f}, median={np.median(length)}")
print(f"Free flow time: min={fft.min()}, max={fft.max()}, mean={fft.mean():.2f}, median={np.median(fft)}")
print(f"Capacity: min={capacity.min()}, max={capacity.max()}, mean={capacity.mean():.2f}, median={np.median(capacity)}")


# ---------------- Load all OD matrices ----------------
all_demands = []

if os.path.isdir(od_dir):
    print(f" Loading OD matrices from folder: {od_dir}")
    od_files = sorted([f for f in os.listdir(od_dir) if f.endswith(".pkl")])
    for file in tqdm(od_files, desc="Loading OD matrices"):
        path = os.path.join(od_dir, file)
        with open(path, "rb") as f:
            od_data = pickle.load(f)
        if isinstance(od_data, dict):
            all_demands.extend(od_data.values())
        elif isinstance(od_data, list):
            for d in od_data:
                if isinstance(d, dict):
                    all_demands.extend(d.values())
        else:
            print(f"{file}: unsupported type {type(od_data)} — skipped.")

else:
    raise FileNotFoundError(f" Invalid path: {od_dir}")

# Convert to NumPy array
all_demands = np.array(all_demands, dtype=float)
print(f"\n Loaded {len(all_demands)} total OD demand values.")
print(f"   Min: {np.min(all_demands):.2f}, Max: {np.max(all_demands):.2f}, Mean: {np.mean(all_demands):.2f}")


def check_array(name, arr):
    arr = np.array(arr, dtype=float)  # ensure it's numeric
    num_inf = np.isinf(arr).sum()
    num_nan = np.isnan(arr).sum()
    num_neg = (arr < 0).sum() if arr.dtype != bool else 0
    num_zero = (arr == 0).sum()
    min_val = np.nanmin(arr)
    max_val = np.nanmax(arr)
    mean_val = np.nanmean(arr)
    print(f"\n{name}:")
    print(f"  Shape: {arr.shape}")
    print(f"  Inf count: {num_inf}")
    print(f"  NaN count: {num_nan}")
    print(f"  Negative count: {num_neg}")
    print(f"  Zero count: {num_zero}")
    print(f"  Min: {min_val}, Max: {max_val}, Mean: {mean_val}")

# Check all your plotting arrays
check_array("length", length)
check_array("fft", fft)
check_array("capacity", capacity)
check_array("all_demands", all_demands)

# ---------------- Plot distributions ----------------
plt.figure(figsize=(12, 10))

plt.subplot(2,2,1)
sns.histplot(np.log1p(length), bins=20, kde=True, color='blue')
plt.xlabel('Log(Link Length)')
plt.ylabel('Frequency')

plt.subplot(2,2,2)
sns.histplot(np.log1p(fft), bins=20, kde=True, color='orange')
plt.xlabel('Log(Free Flow Time)')
plt.ylabel('Frequency')

plt.subplot(2,2,3)
sns.histplot(normalize(capacity).flatten(), bins=20, kde=True, color='green')
plt.xlabel('Normalized Capacity')
plt.ylabel('Frequency')

plt.subplot(2,2,4)
sns.histplot(all_demands, bins=20, kde=True, color='red')
plt.xlabel('OD Demand')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


print("\nOD demand across all matrices:")
print(f"Min demand: {all_demands.min()}")
print(f"Max demand: {all_demands.max()}")
print(f"Mean demand: {all_demands.mean():.2f}")
print(f"Median demand: {np.median(all_demands)}")
