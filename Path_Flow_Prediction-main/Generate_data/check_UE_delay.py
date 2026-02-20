# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 12:21:42 2025

@author: shams
"""

import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from tqdm import tqdm
import os


# -------------------------
# Utility Functions
# -------------------------

def read_file(filename):
    with open(filename, "rb") as file:
        stat = pickle.load(file)
    return stat


def get_origin_path(stat):
    path_link = stat['data']['paths_link']
    od = list(path_link.keys())
    path1 = [tuple(p[0]) if len(p) > 0 else np.nan for p in path_link.values()]
    path2 = [tuple(p[1]) if len(p) > 1 else np.nan for p in path_link.values()]
    path3 = [tuple(p[2]) if len(p) > 2 else np.nan for p in path_link.values()]

    demand_dic = stat["data"]["demand"]
    demand = list(demand_dic.values())

    path_link_df = pd.DataFrame({
        "od": od, 
        "demand": demand, 
        "path1": path1, 
        "path2": path2, 
        "path3": path3
    })
    return path_link_df


def get_UE_link_cost(stat, remove_ids=None):
    # Return DataFrame of link cost and flow
    link = stat['data']['network'].copy()

    if remove_ids:
        a = stat['link_flow']
        for index in sorted(remove_ids, reverse=True):
            del a[index]
        link['link_flow'] = a
    else:
        link['link_flow'] = stat['link_flow']

    # Compute link cost using BPR function
    link['link_cost'] = round(
        link['free_flow_time'] * (1 + link['b'] * ((link['link_flow'] / link['capacity']) ** link['power'])),
        4
    )
    return link


def calculate_path_cost(row, link_df):
    if pd.isna(row):
        return np.nan
    total_time = 0
    for link in row:
        cost = link_df.loc[link_df['link_id'] == link, 'link_cost']
        if not cost.empty:
            total_time += cost.iloc[0]
    return round(total_time, 4)


def extract_link_flow(path_link, flows):
    # input: a dictionary of {od pair: path_link} and list of flow distribution
    path_flow = {}
    for path_set, flow_set in zip(path_link.values(), flows):
        for path, flow in zip(path_set, flow_set):
            path_flow[tuple(path)] = flow

    aggregated_sums = defaultdict(float)
    for path, flow in path_flow.items():
        for link in path:
            aggregated_sums[link] += flow
    return dict(aggregated_sums)


# -------------------------
# UE Checking Functions
# -------------------------

def mean_path_cost(filename, remove_ids=None):
    stat = read_file(filename)
    path_link_df = get_origin_path(stat)
    UE_link = get_UE_link_cost(stat, remove_ids)

    # Compute path costs
    for i in range(3):
        path_link_df[f'path{i+1}_cost'] = path_link_df[f'path{i+1}'].apply(lambda x: calculate_path_cost(x, UE_link))

    # Extract path flows
    flows = stat['path_flow']
    for i in range(3):
        path_link_df[f'flow{i+1}'] = [f[i] if len(f) > i else 0 for f in flows]

    mean_cost = np.nanmean([
        path_link_df['path1_cost'].mean(),
        path_link_df['path2_cost'].mean(),
        path_link_df['path3_cost'].mean()
    ])
    return UE_link, path_link_df, mean_cost


def calculate_delay(pred_df, pred_link_flow):
    # Compute path costs again with predicted link flow
    for i in range(3):
        pred_df[f'path{i+1}_cost'] = pred_df[f'path{i+1}'].apply(lambda x: calculate_path_cost(x, pred_link_flow))

    pred_df['min_path_cost'] = pred_df[['path1_cost', 'path2_cost', 'path3_cost']].min(axis=1)

    # Delay relative to min cost
    for i in range(3):
        pred_df[f'delay{i+1}'] = round(
            (pred_df[f'path{i+1}_cost'] - pred_df['min_path_cost']) / pred_df['min_path_cost'] * 100, 4
        )

    # Consider only OD pairs with positive demand
    used_mask = (pred_df['flow1'] > 1e-6) | (pred_df['flow2'] > 1e-6) | (pred_df['flow3'] > 1e-6)
    used_paths = pred_df.loc[used_mask].copy()

    # Weighted average delay for used paths
    weighted_delay = (
        (used_paths['flow1'] * used_paths['delay1'] +
         used_paths['flow2'] * used_paths['delay2'] +
         used_paths['flow3'] * used_paths['delay3'])
        / (used_paths[['flow1', 'flow2', 'flow3']].sum(axis=1) + 1e-9)
    )

    avg_delay = weighted_delay.mean()
    return pred_df, avg_delay


# -------------------------
# 3️⃣ Main Function
# -------------------------

def main():
    # Configuration
    folder = r"E:\sshams\DL_TAP\Path_Flow_Prediction-main\Solution\SiouxFalls\Output1"
    dataset_name = "SiouxFall_UE_"  # change for EMA if needed
    num_files = 10  # number of UE output files


    avg_delays = []
    mean_costs = []

    for i in tqdm(range(num_files)):
        file_name = f"{folder}/{dataset_name}{i}"

        try:
            UE_link, df, mean_cost = mean_path_cost(file_name)
            df, delay = calculate_delay(df, UE_link)

            avg_delays.append(delay)
            mean_costs.append(mean_cost)
        except Exception as e:
            print(f"❌ Error reading {file_name}: {e}")

    overall_delay = np.mean(avg_delays)
    overall_mean_cost = np.mean(mean_costs)

    print("\n-------------------------------")
    print(f"✅ Mean path cost: {overall_mean_cost:.3f} mins")
    print(f"✅ Average UE delay: {overall_delay:.3f} mins = {round(overall_delay / overall_mean_cost * 100, 3)}%")
    print("-------------------------------")


# -------------------------
# 4️⃣ Run the script
# -------------------------
if __name__ == "__main__":
    main()
