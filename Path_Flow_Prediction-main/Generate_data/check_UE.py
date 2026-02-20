# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 18:42:21 2025

@author: shams
"""


#this code is to check the UE solution genrated. Path 1 being shortest would ideally show maximum flows while decreases with Path 2, 3

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
import pandas as pd

# --- Adjust path to your UE output folder ---
FOLDERS = [r'E:\sshams\DL_TAP\Path_Flow_Prediction-main\Solution\SiouxFalls\Output1']
DATA_SIZE = 50

def read_file(filename):
  with open(filename, "rb") as file:
      stat = pickle.load(file)
      file.close()
  return stat

def load_files_from_folders(folders, max_files):
    file_list = []
    for folder in folders:
        for i in range(max_files):
            file = ''.join([folder,str(f'/SiouxFall_UE_{i}')])
            file_list.append(file)
    return file_list

# --- Load files ---
files = load_files_from_folders(FOLDERS, max_files=DATA_SIZE)

path_flow_all = []

for filename in tqdm(files):
    stat = read_file(filename)
    flows = stat["path_flow"]
    path_flow_all.extend(flows)  # flatten all files

f1 = [f[0] for f in path_flow_all if len(f) > 0 and f[0] > 10]
f2 = [f[1] for f in path_flow_all if len(f) > 1 and f[1] > 10]
f3 = [f[2] for f in path_flow_all if len(f) > 2 and f[2] > 10]


# --- Plot histograms ---
plt.figure(figsize=(15, 5))
bins = 20

plt.subplot(1, 3, 1)
sns.histplot(f1, bins=bins, kde=True, alpha=0.7)
plt.title("1st Path Flow")
plt.xlabel("Flow")
plt.ylabel("Frequency")

plt.subplot(1, 3, 2)
sns.histplot(f2, bins=bins, kde=True, alpha=0.7)
plt.title("2nd Path Flow")
plt.xlabel("Flow")
plt.ylabel("Frequency")

plt.subplot(1, 3, 3)
sns.histplot(f3, bins=bins, kde=True, alpha=0.7)
plt.title("3rd Path Flow")
plt.xlabel("Flow")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# ------------------ Load link flows ------------------
Link_flow, Link_id = [], []

for folder in FOLDERS:
    link_flow, link_id = [], []
    for i in range(DATA_SIZE):
        file = ''.join([folder, f'/SiouxFall_UE_{i}'])
        stat = read_file(file)
        link_id.append(stat['data']['network']['link_id'])
        link_flow.append(stat['link_flow'])
    Link_flow.append(link_flow)
    Link_id.append(link_id)

# Create DataFrame for full network (example: first folder)
output1 = pd.DataFrame(Link_flow[0], columns=Link_id[0][0])

# ------------------ Plot link flows as boxplot ------------------
plt.figure(figsize=(15, 6))
sns.boxplot(data=output1, showfliers=False)
plt.xticks(np.arange(0, output1.shape[1], step=5))
plt.title('Optimal Link Flows (No Missing Data)')
plt.xlabel('Link ID')
plt.ylabel('Flow (veh)')
plt.show()
