# -*- coding: utf-8 -*-
"""
Full pipeline: Generate network variants → OD matrices → Solve UE

"""
import os
import pickle
import random
import time
from math import ceil
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import re
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- Utils ----------------
from utils import (
    generate_OD_demand,
    solve_UE,
    readNet,
    get_full_paths_from_folder_filtered_timed_find_paths
)

# ---------------- Configuration ----------------
orig_tntp_file = r"E:\sshams\DL_TAP\Path_Flow_Prediction-main\Generate_data\SiouxFalls\SiouxFalls_net.tntp"
pos_file = r"E:\sshams\DL_TAP\Path_Flow_Prediction-main\Generate_data\SiouxFalls\SiouxFalls_node.csv"

base_trip_file = r"E:\sshams\DL_TAP\Path_Flow_Prediction-main\Generate_data\SiouxFalls\SiouxFalls_trips.tntp"
variants_dir = r"E:\sshams\DL_TAP\Path_Flow_Prediction-main\Generate_data\SiouxFalls\test"
od_dir_base = r"E:\sshams\DL_TAP\Path_Flow_Prediction-main\Generate_data\SiouxFalls\test"
ue_output_base = r"E:\sshams\DL_TAP\Path_Flow_Prediction-main\Generate_data\SiouxFalls\test\UE_Output"

n_variants = 5
link_change_frac = 0.75
increase_frac=0.75
num_od_matrices = 50

# ---------------- Module-level functions ----------------




def increase_network_size_variant(orig_tntp_file, pos_file, new_tntp_file,
                                  increase_frac=0.25, n_variants=5, seed=None,
                                  plot_networks=True):
    """
    Generate enlarged TNTP network variants from a base network (e.g., SiouxFalls).
    Adds new nodes and links whose features are sampled from the *global feature
    distributions* of the base network — not from neighbors.
    Ensures all new nodes are connected and plots results.
    """

    import os, random, pandas as pd, numpy as np, networkx as nx, matplotlib.pyplot as plt

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    os.makedirs(os.path.dirname(new_tntp_file), exist_ok=True)

    # --- Read original TNTP network ---
    with open(orig_tntp_file, 'r') as f:
        header_lines = [next(f) for _ in range(8)]
    net = pd.read_csv(orig_tntp_file, delimiter='\t', skiprows=8)
    net.columns = [c.strip() for c in net.columns]
    nodes_orig = set(net['init_node']).union(set(net['term_node']))

    # --- Read node positions ---
    pos_df = pd.read_csv(pos_file)
    pos_dict = {row['Node']: (row['X'], row['Y']) for _, row in pos_df.iterrows()}

    # --- Prepare feature distributions for sampling ---
    cap_vals = net['capacity'].values
    fft_vals = net['free_flow_time'].values
    len_vals = net['length'].values
    b_vals = net['b'].values
    p_vals = net['power'].values

    variant_files = []

    for i in range(1, n_variants + 1):
        print(f"\n🔹 Generating Size Variant #{i}...")

        all_nodes = list(nodes_orig)
        existing_links = set(zip(net['init_node'], net['term_node']))
        new_links_data = []

        # --- Create new nodes ---
        num_new_nodes = int(len(nodes_orig) * increase_frac)
        max_node_id = max(nodes_orig)
        new_nodes = [max_node_id + j + 1 for j in range(num_new_nodes)]
        all_nodes += new_nodes

        # --- Place new nodes slightly outside bounding box ---
        xs = [x for x, y in pos_dict.values()]
        ys = [y for x, y in pos_dict.values()]
        xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
        x_span, y_span = xmax - xmin, ymax - ymin
        for n in new_nodes:
            pos_dict[n] = (
                random.uniform(xmin - 0.2 * x_span, xmax + 0.2 * x_span),
                random.uniform(ymin - 0.2 * y_span, ymax + 0.2 * y_span)
            )

        last_link_id = int(net['link_id'].max()) if 'link_id' in net.columns else len(net)

        # --- Ensure connectivity: connect each new node bidirectionally ---
        for n in new_nodes:
            existing_node = random.choice(list(nodes_orig))
            for u, v in [(existing_node, n), (n, existing_node)]:
                last_link_id += 1
                new_links_data.append({
                    'link_id': last_link_id,
                    'init_node': int(u),
                    'term_node': int(v),
                    'capacity': int(random.choice(cap_vals)),
                    'length': float(random.choice(len_vals)),
                    'free_flow_time': float(random.choice(fft_vals)),
                    'b': float(random.choice(b_vals)),
                    'power': float(random.choice(p_vals)),
                    'speed': 0,
                    'toll': 0,
                    'link_type': 1
                })
                existing_links.add((u, v))

        # --- Add random additional links among all nodes ---
        num_new_links = int(len(net) * increase_frac)
        while len(new_links_data) < num_new_links:
            u, v = random.sample(all_nodes, 2)
            if (u, v) not in existing_links:
                last_link_id += 1
                new_links_data.append({
                    'link_id': last_link_id,
                    'init_node': int(u),
                    'term_node': int(v),
                    'capacity': int(random.choice(cap_vals)),
                    'length': float(random.choice(len_vals)),
                    'free_flow_time': float(random.choice(fft_vals)),
                    'b': float(random.choice(b_vals)),
                    'power': float(random.choice(p_vals)),
                    'speed': 0,
                    'toll': 0,
                    'link_type': 1
                })
                existing_links.add((u, v))

        # --- Combine old + new links ---
        net_new = pd.concat([net, pd.DataFrame(new_links_data)], ignore_index=True)
        net_new['link_id'] = range(len(net_new))

        # --- Ensure strong connectivity ---
        G_new = nx.DiGraph()
        for _, row in net_new.iterrows():
            G_new.add_edge(row['init_node'], row['term_node'])
        if not nx.is_strongly_connected(G_new):
            print("⚠️ Network not strongly connected. Adding repair links...")
            components = list(nx.strongly_connected_components(G_new))
            largest = max(components, key=len)
            for comp in components:
                if comp != largest:
                    u = random.choice(list(comp))
                    v = random.choice(list(largest))
                    last_link_id += 1
                    net_new = pd.concat([net_new, pd.DataFrame([{
                        'link_id': last_link_id,
                        'init_node': int(u),
                        'term_node': int(v),
                        'capacity': int(random.choice(cap_vals)),
                        'length': float(random.choice(len_vals)),
                        'free_flow_time': float(random.choice(fft_vals)),
                        'b': float(random.choice(b_vals)),
                        'power': float(random.choice(p_vals)),
                        'speed': 0,
                        'toll': 0,
                        'link_type': 1
                    }])], ignore_index=True)
                    G_new.add_edge(u, v)

        # --- Save TNTP variant ---
        
        with open(new_tntp_file, 'w') as f:
            for line in header_lines:
                f.write(line.strip() + '\n')
            f.write("\n~\tlink_id\tinit_node\tterm_node\tcapacity\tlength\tfree_flow_time\tb\tpower\tspeed\ttoll\tlink_type\t;\n")
            for _, row in net_new.iterrows():
                f.write(
                    f"\t{int(row['link_id'])}\t{int(row['init_node'])}\t{int(row['term_node'])}\t"
                    f"{float(row['capacity']):.0f}\t{float(row['length']):.2f}\t{float(row['free_flow_time']):.2f}\t"
                    f"{float(row['b']):.2f}\t{float(row['power']):.2f}\t"
                    f"{int(row['speed'])}\t{int(row['toll'])}\t{int(row['link_type'])}\t;\n"
                )

        print(f"✅ Size Variant #{i} saved → {new_tntp_file}")
        print(f"Nodes: {len(set(net_new['init_node']).union(set(net_new['term_node'])))} | Links: {len(net_new)}")
        

        # --- Plot enlarged network (optional) ---
        if plot_networks:
            plt.figure(figsize=(10, 6))
            nx.draw(G_new, pos=pos_dict, node_size=40, node_color='dodgerblue',
                    edge_color='gray', arrows=False)
            plt.title(f"SiouxFalls Size Variant #{i} (+{int(increase_frac*100)}% nodes)")
            plt.show()

    return variant_files, pos_dict




def perturb_network_links_tntp_1(
    orig_tntp_file, pos_file, new_tntp_file,
    link_change_frac=0.75, seed=None):
    """
    Perturb a TNTP network by randomly adding/removing links, keeping the network connected.
    Feature values (capacity, length, free_flow_time, b, power) for new links
    are computed as the average of existing links connected to the same nodes.
    """

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # --- Read TNTP file ---
    with open(orig_tntp_file, 'r') as f:
        header_lines = [next(f) for _ in range(8)]
    net = pd.read_csv(orig_tntp_file, delimiter='\t', skiprows=8)
    net.columns = [c.strip() for c in net.columns]

    nodes_orig = set(net['init_node']).union(set(net['term_node']))

    # --- Read node positions ---
    pos_df = pd.read_csv(pos_file)
    pos_dict = {row['Node']: (row['X'], row['Y']) for _, row in pos_df.iterrows()}

    # --- Normalize node IDs ---
    sorted_nodes = sorted(nodes_orig)
    node_map = {old: i + 1 for i, old in enumerate(sorted_nodes)}
    net['init_node'] = net['init_node'].map(node_map)
    net['term_node'] = net['term_node'].map(node_map)
    pos_dict = {node_map[k]: v for k, v in pos_dict.items() if k in node_map}
    nodes_orig = set(node_map.values())

    # --- Build directed graph ---
    G = nx.DiGraph()
    G.add_nodes_from(nodes_orig)
    for _, row in net.iterrows():
        G.add_edge(row['init_node'], row['term_node'])

    num_links = len(net)
    num_change = int(num_links * link_change_frac)

    # --- Remove some links randomly ---
    removable_links = list(G.edges)
    random.shuffle(removable_links)
    removed_links = []
    for u, v in removable_links:
        if len(removed_links) >= num_change // 2:
            break
        G.remove_edge(u, v)
        if nx.is_strongly_connected(G):
            removed_links.append((u, v))
        else:
            G.add_edge(u, v)

    # --- Add new random links ---
    all_possible_links = {(u, v) for u in nodes_orig for v in nodes_orig if u != v}
    potential_new_links = list(all_possible_links - set(G.edges))
    random.shuffle(potential_new_links)
    added_links = []
    for u, v in potential_new_links:
        if len(added_links) >= num_change - len(removed_links):
            break
        G.add_edge(u, v)
        added_links.append((u, v))

    # --- Helper: get neighbors of node ---
    def get_neighbor_links(node):
        """Return attribute values of all links connected to this node."""
        mask = (net['init_node'] == node) | (net['term_node'] == node)
        return net.loc[mask, ['capacity', 'length', 'free_flow_time', 'b', 'power']]

    # --- Compute averaged feature values for new links ---
    new_links_data = []
    for _, row in net.iterrows():
        if (row['init_node'], row['term_node']) not in removed_links:
            new_links_data.append(row.to_dict())

    last_link_id = int(net['link_id'].max()) if 'link_id' in net.columns else len(net)

    for u, v in added_links:
        # Get neighbors of both u and v
        neigh_u = get_neighbor_links(u)
        neigh_v = get_neighbor_links(v)
        neigh_all = pd.concat([neigh_u, neigh_v])
        if len(neigh_all) > 0:
            cap = neigh_all['capacity'].mean()
            length = neigh_all['length'].mean()
            fft = neigh_all['free_flow_time'].mean()
            b = neigh_all['b'].mean()
            power = neigh_all['power'].mean()
        else:
            # Fallback: global mean
            cap = net['capacity'].mean()
            length = net['length'].mean()
            fft = net['free_flow_time'].mean()
            b = net['b'].mean()
            power = net['power'].mean()

        last_link_id += 1
        new_links_data.append({
            'link_id': last_link_id,
            'init_node': int(u),
            'term_node': int(v),
            'capacity': int(round(cap)),
            'length': float(round(length, 2)),
            'free_flow_time': float(round(fft, 2)),
            'b': round(b, 2),
            'power': round(power, 2),
            'speed': 0,
            'toll': 0,
            'link_type': 1
        })

    # --- Final DataFrame ---
    net_new = pd.DataFrame(new_links_data).reset_index(drop=True)
    net_new['link_id'] = range(len(net_new))

    # --- Save TNTP ---
    os.makedirs(os.path.dirname(new_tntp_file), exist_ok=True)
    with open(new_tntp_file, 'w') as f:
        for line in header_lines:
            f.write(line.strip() + '\n')
        f.write("\n~\tlink_id\tinit_node\tterm_node\tcapacity\tlength\tfree_flow_time\tb\tpower\tspeed\ttoll\tlink_type\t;\n")
        for _, row in net_new.iterrows():
            f.write(
                f"\t{int(row['link_id'])}\t{int(row['init_node'])}\t{int(row['term_node'])}\t"
                f"{float(row['capacity']):.0f}\t{float(row['length']):.2f}\t{float(row['free_flow_time']):.2f}\t"
                f"{float(row['b']):.2f}\t{float(row['power']):.2f}\t"
                f"{int(row['speed'])}\t{int(row['toll'])}\t{int(row['link_type'])}\t;\n"
            )

    print(f" Variant saved → {new_tntp_file}")
    print(f"Removed: {len(removed_links)}, Added: {len(added_links)}")

    # --- Visualization of changes ---
    plt.figure(figsize=(10, 6))
    G_new = nx.DiGraph()
    for _, row in net_new.iterrows():
        G_new.add_edge(row['init_node'], row['term_node'])

    # Draw base network (gray)
    nx.draw(G_new, pos=pos_dict, node_size=40, node_color='lightblue',
            edge_color='gray', arrows=False, alpha=0.7)

    # Overlay added links (green)
    nx.draw_networkx_edges(G_new, pos=pos_dict, edgelist=added_links,
                           edge_color='green', width=2.5, arrows=False, label='Added links')

    # Overlay removed links (red bold)
    nx.draw_networkx_edges(G_new, pos=pos_dict, edgelist=removed_links,
                           edge_color='red', width=3.0, style='solid', arrows=False, label='Removed links')

    #plt.title(f"Network Topology Variation")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return new_tntp_file, net, net_new, pos_dict, added_links, removed_links, G



def generate_variant(i, seed=None):
    """Generate a single network variant using perturbation."""
    variant_file = os.path.join(variants_dir, f"variant_{i}.tntp")
    os.makedirs(variants_dir, exist_ok=True)
    perturb_network_links_tntp_1(orig_tntp_file, pos_file, variant_file, link_change_frac, seed=seed)#, mode_range=None, desc=None, seed=seed or i)
    
#     increase_network_size_variant(
#     orig_tntp_file, pos_file, variant_file,
#     increase_frac=increase_frac,
#     n_variants=n_variants,
#     seed=seed,
#     plot_networks=False
# )
    return variant_file

def generate_and_save_od_matrix(file_index, num_nodes, min_demand, max_demand, num_pairs, save_dir):
    """Top-level function to generate one OD matrix (Windows-safe)."""
    file_path = os.path.join(save_dir, f"od_matrix_{file_index}.pkl")
    if os.path.exists(file_path):
        return file_path
    od_demand = generate_OD_demand(num_nodes, min_demand, max_demand, num_pairs)
    with open(file_path, 'wb') as f:
        pickle.dump(od_demand, f)
    return file_path


def read_tntp_trip_file(file_path):
    """Parse a TNTP-format trip table → dict {(origin, destination): demand}."""
    import re
    od_demand = {}
    current_origin = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("origin"):
                current_origin = int(line.split()[1])
            elif ":" in line and current_origin is not None:
                pairs = re.findall(r"(\d+)\s*:\s*([\d.]+)", line)
                for dest, val in pairs:
                    dest = int(dest)
                    if current_origin != dest:
                        od_demand[(current_origin, dest)] = float(val)
    return od_demand


def generate_and_save_variations_od(base_od, save_dir, num_variations=4000,
                                 scale_min=0.2, scale_max=1.0, missing_frac=0.3):
    """Generate scaled/missing OD variants from a base trip table."""
    import pickle, random, os
    from tqdm import tqdm

    os.makedirs(save_dir, exist_ok=True)
    all_ods = list(base_od.keys())
    num_missing = int(len(all_ods) * missing_frac)
    file_paths = []

    for i in tqdm(range(num_variations), desc=f"Generating OD matrices → {os.path.basename(save_dir)}"):
        file_path = os.path.join(save_dir, f"od_matrix_{i}.pkl")
        if os.path.exists(file_path):
            file_paths.append(file_path)
            continue

        # Random scaling
        new_od = {od: round(val * random.uniform(scale_min, scale_max), 2)
                  for od, val in base_od.items()}

        # Random missing OD pairs
        for od in random.sample(all_ods, num_missing):
            del new_od[od]

        with open(file_path, 'wb') as f:
            pickle.dump(new_od, f)
        file_paths.append(file_path)

    return file_paths


def generate_od_for_variant(variant_file, save_dir, num_matrix=num_od_matrices):
    """
    Generate OD matrices for a variant.
    - If trip table exists: generate scaled OD variations.
    - Otherwise: fall back to synthetic generation.
    """
    import pickle, os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    Network, Nodes, links, cap, fft, alpha, beta, lengths = readNet(variant_file)
    num_nodes = len(Nodes)
    min_demand, max_demand = 100, 4000
    num_pairs = int(num_nodes**2 * 0.7)
    os.makedirs(save_dir, exist_ok=True)

    # --- Check for existing OD files ---
    existing_files = [f for f in os.listdir(save_dir) if f.endswith(".pkl")]
    if len(existing_files) >= num_matrix:
        print(f"ℹ️ OD matrices already exist in {save_dir} (skipping generation)")
        return

    # --- Try to find a TNTP trip file (base OD) ---
    base_dir = os.path.dirname(base_trip_file)
    trip_files = [f for f in os.listdir(base_dir)
                  if "trip" in f.lower() and f.endswith((".tntp", ".txt"))]

    if trip_files:
        trip_path = os.path.join(base_dir, trip_files[0])
        print(f"📊 Using base trip table: {trip_path}")
        base_od = read_tntp_trip_file(trip_path)
        print(f"✅ Parsed {len(base_od)} OD pairs from trip table.")
        generate_and_save_variations_od(base_od, save_dir,
                                     num_variations=num_matrix,
                                     scale_min=0.2, scale_max=1.0,
                                     missing_frac=0.3)
        print(f"🎯 Finished generating {num_matrix} OD variations for {variant_file}")
        return

    # --- Fall back to synthetic OD generation ---
    print(" No trip table found → generating synthetic OD matrices instead.")
    args_list = [(i, num_nodes, min_demand, max_demand, num_pairs, save_dir)
                 for i in range(num_matrix)]
    with ProcessPoolExecutor(max_workers=min(10, os.cpu_count() - 1)) as executor:
        futures = [executor.submit(generate_and_save_od_matrix, *args)
                   for args in args_list]
        for future in as_completed(futures):
            print(f"✅ Saved {future.result()}")


def solve_single_od(args):
    """Solve UE for one OD matrix."""
    net_file, od_path, pair_path, output_prefix = args
    base_number = os.path.splitext(os.path.basename(od_path))[0].split('_')[-1]
    output_file = os.path.join(os.path.dirname(output_prefix),
                               f"{os.path.basename(output_prefix)}{base_number}")
    solve_UE(net_file, od_path, pair_path, output_file, base_number, to_solve=1)
    return od_path, output_file

def solve_ue_for_variant(variant_file, od_dir, output_prefix, path_file, pair_file):
    """Solve UE for all OD matrices of a variant."""
    # Load or compute paths
    if os.path.exists(path_file) and os.path.exists(pair_file):
        with open(path_file, 'rb') as f:
            path_set_dict = pickle.load(f)
        with open(pair_file, 'rb') as f:
            pair_path = pickle.load(f)
    else:
        path_set_dict, pair_path = get_full_paths_from_folder_filtered_timed_find_paths(
            demand_dir=od_dir,
            net_file=variant_file,
            path_num=3,
            limit=10,
            num_processes=10,
            path_file=path_file,
            pair_file=pair_file
        )
        with open(path_file, 'wb') as f:
            pickle.dump(path_set_dict, f)
        with open(pair_file, 'wb') as f:
            pickle.dump(pair_path, f)

    # UE computation in parallel
    od_files = sorted([f for f in os.listdir(od_dir) if f.endswith('.pkl')],
                      key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
    args_list = [(variant_file, os.path.join(od_dir, f), pair_path, output_prefix) for f in od_files]
    with ProcessPoolExecutor(max_workers=min(10, os.cpu_count()-1)) as executor:
        futures = [executor.submit(solve_single_od, args) for args in args_list]
        for future in as_completed(futures):
            od_file, out_file = future.result()
            print(f"✅ UE completed: {od_file} → {out_file}")

# ---------------- Main Pipeline ----------------


def main():
    multiprocessing.freeze_support()  # Windows-safe

    variant_files = []

    # Generate all network variants
    print(f"\n🔹 Generating {n_variants} network variants...")
    for i in range(1, n_variants + 1):
        variant_file = os.path.join(variants_dir, f"variant_{i}.tntp")
        if os.path.exists(variant_file):
            print(f"ℹ️ Variant {i} already exists → {variant_file} (skipping generation)")
        else:
            generate_variant(i)
            print(f"✅ Variant {i} saved: {variant_file}")
        variant_files.append(variant_file)

    # Generate OD matrices for each variant
    for i, variant_file in enumerate(variant_files, start=1):
        od_dir = os.path.join(od_dir_base, f"OD_Matrices_{i}")
        if os.path.exists(od_dir) and len(os.listdir(od_dir)) >= num_od_matrices:
            print(f"ℹ️ OD matrices for Variant {i} already exist in {od_dir} (skipping generation)")
        else:
            print(f"\n🔹 Generating OD matrices for Variant {i}...")
            generate_od_for_variant(variant_file, od_dir, num_od_matrices)

    # Solve UE for each variant
    for i, variant_file in enumerate(variant_files, start=1):
        od_dir = os.path.join(od_dir_base, f"OD_Matrices_{i}")
        output_prefix = os.path.join(ue_output_base, f"Variant_{i}", "SiouxFalls_UE_")
        os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
        path_file = os.path.join(ue_output_base, f"Variant_{i}", "unique_paths.pkl")
        pair_file = os.path.join(ue_output_base, "pair_path.pkl")

        # Check if UE outputs already exist
        existing_ue_files = [f for f in os.listdir(os.path.dirname(output_prefix))
                             if f.startswith(os.path.basename(output_prefix))]
        if existing_ue_files:
            print(f"ℹ️ UE outputs for Variant {i} already exist → {len(existing_ue_files)} files (skipping UE)")
        else:
            print(f"\n🔹 Solving UE for Variant {i}...")
            solve_ue_for_variant(variant_file, od_dir, output_prefix, path_file, pair_file)

    print("\n🎉 All variants processed successfully!")

# ---------------- Entry Point ----------------
if __name__ == "__main__":
    main()
