import os
import pickle
import glob
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import generate_OD_demand, readNet, add_link_ids_to_tntp  

# ---------------- Configuration ----------------
net_file = r"SiouxFalls/SiouxFalls_net.tntp"
net_link_file= r"SiouxFalls/SiouxFalls_net_link.tntp"
save_dir = r"E:\sshams\DL_TAP\Path_Flow_Prediction-main\Generate_data\SiouxFalls\OD_Matrices"

os.makedirs(save_dir, exist_ok=True)

#add_link_ids_to_tntp(net_file, net_link_file)
Network, Nodes, links, cap, fft, alpha, beta, lengths = readNet(net_file)


num_nodes = len(Nodes)

min_demand = 50
max_demand = 500
num_pairs = int((num_nodes ** 2) * 0.7)
num_matrix = 4000
batch_size = 50  # smaller batch to avoid too many handles on Windows
max_workers = min(10, os.cpu_count() - 1)  # safe number of processes

total_cpus = os.cpu_count()
print(f" Total logical CPUs available: {total_cpus}")
print(f" Number of worker processes that will be used: {max_workers}\n")

print(f"\n Network Info")
print(f"Network: {os.path.basename(net_file).replace('.tntp','')}")
print(f"  Nodes: {num_nodes}")
print(f"  Links: {links}")
print(f"Generating OD matrices: total {num_matrix}, {num_pairs} pairs each\n")

# ---------------- Windows-safe multiprocessing ----------------
def generate_and_save(i):
    """Generate a single OD matrix and save it to disk."""
    try:
        file_path = os.path.join(save_dir, f"od_matrix_{i}.pkl")
        if os.path.exists(file_path):
            return file_path
        od_demand = generate_OD_demand(num_nodes, min_demand, max_demand, num_pairs)
        with open(file_path, 'wb') as f:
            pickle.dump(od_demand, f)
        del od_demand
        return file_path
    except Exception as e:
        print(f" Error generating/saving matrix {i}: {e}")
        raise e

def main():
    # ---------------- Check existing ----------------
    existing_files = [f for f in os.listdir(save_dir) if f.endswith('.pkl')]
    existing_indices = {int(f.split("_")[-1].split(".")[0]) for f in existing_files}
    remaining_indices = [i for i in range(num_matrix) if i not in existing_indices]

    print(f" {len(existing_indices)} matrices already exist, generating remaining {len(remaining_indices)}...")

    if remaining_indices:
        start_time = time.time()

        for start in range(0, len(remaining_indices), batch_size):
            batch = remaining_indices[start:start + batch_size]
            print(f"\n⏳ Generating batch {start // batch_size + 1} of {((len(remaining_indices) - 1) // batch_size) + 1} ...")
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(generate_and_save, i): i for i in batch}
                for future in as_completed(futures):
                    try:
                        file_path = future.result()
                        print(f" Saved: {file_path}")
                    except Exception as e:
                        print(f" Failed matrix {futures[future]}: {e}")

        elapsed_minutes = (time.time() - start_time) / 60
        print(f"\n Finished generating remaining OD matrices. Total time: {elapsed_minutes:.2f} minutes")
    else:
        print(f" All {num_matrix} OD matrices already exist — skipping generation.")

    # ---------------- Verification ----------------
    pkl_files = sorted(glob.glob(os.path.join(save_dir, "od_matrix_*.pkl")))
    print(f"\nTotal .pkl files found: {len(pkl_files)} in {save_dir}")

    if len(pkl_files) > 0:
        sample_files = random.sample(pkl_files, min(2, len(pkl_files)))
        for file in sample_files:
            with open(file, 'rb') as f:
                od_demand = pickle.load(f)
            print(f"\nLoaded file: {os.path.basename(file)}")
            print(f"Total OD pairs: {len(od_demand)}")
            print("Example OD pairs:")
            for (o, d), val in random.sample(list(od_demand.items()), min(5, len(od_demand))):
                print(f"  ({o} → {d}) : {val}")
            print("-"*50)

# -------------- Windows safe entry point --------------
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # needed for Windows
    main()
