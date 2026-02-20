# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 15:06:09 2025

@author: shams
"""

import os
import pickle
import time
from math import ceil
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import solve_UE, get_full_paths_from_folder_filtered_timed_find_paths  # existing functions
import random
import sys
import os, re, sys, time
from math import ceil

os.environ["GUROBI_LICENSE_FILE"] = r"E:\sshams\DL_TAP\gurobi.lic"

# ---------------- Configuration ----------------


net_file = r"E:\sshams\DL_TAP\Path_Flow_Prediction-main\Generate_data\SiouxFalls\SiouxFalls_net.tntp"
demand_dir = r"E:\sshams\DL_TAP\Path_Flow_Prediction-main\Generate_data\SiouxFalls\OD_Matrices"
output_prefix = '../Solution/SiouxFalls/Output_time/SiouxFalls_UE_'
path_file = 'SiouxFalls/unique_paths.pkl'
pair_file = 'SiouxFalls/pair_path.pkl'


def load_od_files(demand_dir):
    """
    Loads OD matrices from directory.
    If a single file contains a list of OD matrices (e.g., od_matrix_4000.pkl),
    split them into separate files (od_matrix_1.pkl, od_matrix_2.pkl, ...).
    Returns a sorted list of all OD matrix file names.
    """

    all_files = sorted(
        [f for f in os.listdir(demand_dir) if f.endswith('.pkl')],
        key=lambda x: int(''.join(filter(str.isdigit, x)) or 0)    
    )

    print(f"Found {len(all_files)} OD matrices in {demand_dir}")
    return all_files[0:10]

# ---------------- Function to solve UE for a single OD ----------------
def solve_single_od(args):
    net_file, demand_dir, file_name, pair_path, output_prefix = args
    od_path = os.path.join(demand_dir, file_name)

    # Extract number from OD matrix filename
    base_number = os.path.splitext(file_name)[0].split('_')[-1]

    # Build output file path
    output_file = os.path.join(
        os.path.dirname(output_prefix),
        f"{os.path.basename(output_prefix)}{base_number}"
    )

    # Solve UE and save output
    dataa = solve_UE(net_file, od_path, pair_path, output_file, base_number, to_solve=1)

    return file_name, output_file

# ---------------- Main function ----------------
def main():
    batch_size = 20
    path_num = 3
    limit = 20
    num_processes = min(60, max(1, os.cpu_count() - 1))

    # ---------------- Compute or load paths ----------------
    if os.path.exists(path_file) and os.path.exists(pair_file):
        print(f"Loading precomputed paths from {path_file} and {pair_file}...")
        with open(path_file, 'rb') as f:
            path_set_dict = pickle.load(f)
        with open(pair_file, 'rb') as f:
            pair_path = pickle.load(f)
    else:
        print("Computing paths from OD matrices...")

        path_set_dict, pair_path = get_full_paths_from_folder_filtered_timed_find_paths(
            demand_dir=demand_dir,
            net_file=net_file,
            path_num=3,            # number of paths per OD
            limit=limit,              # random sample of 10 OD matrices
            num_processes=20,      # adjust for CPU cores
            path_file=path_file,
            pair_file=pair_file,
            #stop_when_full=False
        )
        with open(path_file, 'wb') as f:
            pickle.dump(path_set_dict, f)
        with open(pair_file, 'wb') as f:
            pickle.dump(pair_path, f)
        print(f"Paths computed and saved to {path_file} and {pair_file}")

    # ---------------- Verification of generated paths ----------------
    print("\n Verifying computed paths:")
    num_pairs = len(pair_path)
    print(f"Total OD pairs with paths: {num_pairs}")
    if num_pairs > 0:
        sample_pairs = random.sample(list(pair_path.items()), min(3, num_pairs))
        for (od_pair, paths) in sample_pairs:
            print(f"\nOD pair {od_pair}: {len(paths)} paths")
            for i, path in enumerate(paths[:3]):
                print(f"  Path {i+1}: {path}")
    num_paths = len(path_set_dict)
    print(f"\nTotal unique paths: {num_paths}")
    # if num_paths > 0:
    #     sample_path_ids = random.sample(list(path_set_dict.items()), min(3, num_paths))
    #     print("\nSample path → ID mappings:")
    #     for path, pid in sample_path_ids:
    #         print(f"  ID {pid}: {path}")
    print("\n Path verification complete.\n")

        # ---------------- Prepare OD matrices ----------------
    output_dir = os.path.dirname(output_prefix)
    os.makedirs(output_dir, exist_ok=True)
    
    # Count how many UE solution files already exist (no .pkl ending)
    existing_outputs = [
        f for f in os.listdir(output_dir)
        if f.startswith(os.path.basename(output_prefix))
    ]
    
    # Extract numeric suffixes (e.g. EMA_UE_532 → 532)
    existing_indices = sorted([
        int(re.search(r'(\d+)$', f).group(1))
        for f in existing_outputs if re.search(r'(\d+)$', f)
    ])
    
    if existing_indices:
        last_index = existing_indices[-1]
        print(f" Last completed UE index: {last_index}")
    else:
        last_index = -1
        print(" No existing UE files found.")
    
    # Get all OD matrix files in demand_dir
    all_files = load_od_files(demand_dir)  # your function from earlier
    
    # Calculate remaining OD matrices to solve
    remaining_files = all_files[last_index + 1:]
    
    print(f" Total OD matrices: {len(all_files)}")
    print(f" Already solved: {last_index + 1}")
    print(f" Remaining to compute: {len(remaining_files)}")
    
    if not remaining_files:
        print(" All UE solutions already computed. Nothing to do!")
        sys.exit(0)
    
    # ---------------- Batch processing ----------------
    start_time = time.time()
    total_batches = ceil(len(remaining_files) / batch_size)
    
    for batch_id, batch_start in enumerate(range(0, len(remaining_files), batch_size), start=1):
        batch = remaining_files[batch_start:batch_start + batch_size]
        print(f"\n⚙️ Processing batch {batch_id} of {total_batches} ...")
        
        args_list = [(net_file, demand_dir, f, pair_path, output_prefix) for f in batch]
    
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = {executor.submit(solve_single_od, args): args for args in args_list}
    
            for future in as_completed(futures):
                args = futures[future]
                file_name = args[2]  # file name from args
                try:
                    file_name, output_file = future.result()
    
                    # Check if output saved correctly (no .pkl, just filename)
                    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
                        print(f" Failed to save output for {file_name}. Exiting program!")
                        sys.exit(1)
                    else:
                        print(f" Completed UE for {file_name} → {output_file}")
    
                except Exception as e:
                    print(f" Failed {file_name}: {e}")
                    sys.exit(1)
    
    elapsed_minutes = (time.time() - start_time) / 60
    print(f"\n Finished all remaining UE iterations. Total time: {elapsed_minutes:.2f} minutes")
    # ---------------- Safe entry point for Windows ----------------
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
