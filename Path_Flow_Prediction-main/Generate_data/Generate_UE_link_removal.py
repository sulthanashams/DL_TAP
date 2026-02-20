# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 04:54:17 2026

@author: Cosmos
This script performs robustness testing by:
1️ Randomly removing a percentage of links from the TNTP network.
2️ Cleaning OD pair paths that use those removed links.
3️ Solving UE for all OD matrices using the modified network and filtered paths.
"""

import os
import pickle
import time
import random
import sys
from math import ceil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import *


# ================================
# --- CONFIGURATION PARAMETERS ---
# ================================

# Base input network and OD data
NET_FILE  = r"E:\sshams\DL_TAP\Path_Flow_Prediction-main\Generate_data\SiouxFalls\SiouxFalls_net.tntp"
DEMAND_DIR = r"E:\sshams\DL_TAP\Path_Flow_Prediction-main\Generate_data\SiouxFalls\OD_Matrices"
OUTPUT_PREFIX = r"E:\sshams\DL_TAP\Path_Flow_Prediction-main\Solution\SiouxFalls\Output_70link\SiouxFall_UE_"
PAIR_FILE = r"E:\sshams\DL_TAP\Path_Flow_Prediction-main\Generate_data\SiouxFalls\pair_path.pkl"
PATH_FILE = r"E:\sshams\DL_TAP\Path_Flow_Prediction-main\Generate_data\SiouxFalls\unique_paths.pkl"

# Percentage of links to remove
REMOVE_PERCENT = 5


# ========================================
# --- NETWORK LINK REMOVAL UTILITIES ---
# ========================================

def count_links_in_tntp(file_path):
    """Read TNTP network file and return list of link IDs."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    link_ids = [int(line.split()[0]) for line in lines if line.split() and line.split()[0].isdigit()]
    return link_ids


def remove_links_from_tntp(input_file, output_file, remove_ids):
    """
    Replace the removed links' data with zeros to disable them
    (keeps file structure intact for model compatibility).
    """
    with open(input_file, 'r') as file:
        lines = file.readlines()
    remove_ids = set(remove_ids)

    modified_lines = []
    for line in lines:
        parts = line.split()
        if parts and parts[0].isdigit() and int(parts[0]) in remove_ids:
            parts[1:] = ['0'] * (len(parts) - 2)
            modified_line = '\t' + '\t'.join(parts) + '\t;\n'
            modified_lines.append(modified_line)
        else:
            modified_lines.append(line)

    with open(output_file, 'w') as file:
        file.writelines(modified_lines)
    



def remove_links_from_path(pair_path, remove_ids):
    """
    Remove ODs with no feasible path and OD paths that use removed links, and report stats.
    """
    remove_ids = set(remove_ids)
    new_dict = defaultdict(list)
    removed_paths_count = 0
    od_pairs_lost = 0

    for key, paths in pair_path.items():
        kept_paths = [p for p in paths if not any(link in p for link in remove_ids)]
        removed_paths_count += len(paths) - len(kept_paths)
        if len(kept_paths) == 0:
            od_pairs_lost += 1
        if kept_paths:
            new_dict[key] = kept_paths


    total_pairs = len(pair_path)
    print(f" Removed {removed_paths_count} infeasible paths from pair_path.")
    print(f" OD pairs with **no feasible paths left**: {od_pairs_lost} / {total_pairs} "
          f"({od_pairs_lost/total_pairs*100:.2f}%)")

    return dict(new_dict)


# ========================================
# --- UE SOLVER WRAPPER FUNCTION ---
# ========================================

def solve_single_od(args):
    net_file, demand_dir, file_name, pair_path, output_prefix = args
    od_path = os.path.join(demand_dir, file_name)
    base_number = os.path.splitext(file_name)[0].split('_')[-1]

    print(f"inside func solve_single_od | base_number={base_number}, demand_file={od_path}")

    output_file = os.path.join(
        os.path.dirname(output_prefix),
        f"{os.path.basename(output_prefix)}{base_number}"
    )

    dataa = solve_UE(net_file, od_path, pair_path, output_file, base_number, to_solve=1)
    return file_name, output_file


# ========================================
# --- MAIN PIPELINE ---
# ========================================

def main():
    batch_size = 20
    num_processes = 10#min(60, max(1, os.cpu_count() - 1))

    # --- Load precomputed paths ---
    print(f" Loading precomputed pair_path from {PAIR_FILE}")
    with open(PAIR_FILE, 'rb') as f:
        pair_path = pickle.load(f)
        
    for i, (od_pair, paths) in enumerate(pair_path.items()):
        print(f"OD pair {od_pair}: {len(paths)} paths")
        # for j, path in enumerate(paths, 1):
        #    print(f"  Path {j} length: {len(path)}")
        if i == 5:  # stop after 3 OD pairs (0,1,2)
              break
    
    # --- Select and remove random links ---
    link_ids = count_links_in_tntp(NET_FILE)
    num_to_remove = max(0, int(len(link_ids) * REMOVE_PERCENT / 100))
    random.seed(42)
    remove_ids = random.sample(link_ids, num_to_remove)

    print(f"🕸 Network has {len(link_ids)} links.")
    print(f" Removing {num_to_remove} links ({REMOVE_PERCENT}%).")
    print(f"Removed link IDs (sample): {remove_ids[:10]}")

    # --- Create modified network file ---
    new_net_file = NET_FILE.replace(".tntp", f"_{REMOVE_PERCENT}pct_removed.tntp")
    remove_links_from_tntp(NET_FILE, new_net_file, remove_ids)
    print(f" New network saved to: {new_net_file}")

    # --- Clean pair_path ---
    new_pair_path = remove_links_from_path(pair_path, remove_ids)
    #print(new_pair_path)
    new_pair_path_file = PAIR_FILE.replace(".pkl", f"_{REMOVE_PERCENT}pct_removed.pkl")
    with open(new_pair_path_file, 'wb') as f:
        pickle.dump(new_pair_path, f)
    print(f" Updated pair_path saved to: {new_pair_path_file}")
    
    for i, (od_pair, paths) in enumerate(new_pair_path.items()):
        print(f"OD pair {od_pair}: {len(paths)} paths")
        # for j, path in enumerate(paths, 1):
        #    print(f"  Path {j} length: {len(path)}")
        if i == 2:  # stop after 3 OD pairs (0,1,2)
              break

    # --- Solve UE with modified network ---
    #all_files = sorted([f for f in os.listdir(DEMAND_DIR) if f.endswith('.pkl')])[:100]

    all_files = sorted(
        [f for f in os.listdir(DEMAND_DIR) if f.endswith('.pkl')],
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )#[:5]

    print(f"Found {len(all_files)} OD matrices in {DEMAND_DIR}")

    output_dir = os.path.dirname(OUTPUT_PREFIX)
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()
    total_batches = ceil(len(all_files) / batch_size)

    for start in range(0, len(all_files), batch_size):
        batch = all_files[start:start + batch_size]
        print(f"\n⚙️ Processing batch {start // batch_size + 1} of {total_batches} ...")

        args_list = [(new_net_file, DEMAND_DIR, f, new_pair_path, OUTPUT_PREFIX) for f in batch]

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = {executor.submit(solve_single_od, args): args for args in args_list}

            for future in as_completed(futures):
                args = futures[future]
                file_name = args[2]
                try:
                    file_name, output_file = future.result()
                    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
                        print(f" Failed to save output for {file_name}. Exiting!")
                        sys.exit(1)
                    else:
                        print(f" Completed UE for {file_name} → {output_file}")
                except Exception as e:
                    print(f" Failed {file_name}: {e}")
                    sys.exit(1)

    elapsed_minutes = (time.time() - start_time) / 60
    print(f"\n Finished all UE iterations. Total time: {elapsed_minutes:.2f} minutes")


# ========================================
# --- SAFE ENTRY POINT ---
# ========================================

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
