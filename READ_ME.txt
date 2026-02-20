
FOLDERS and FILE DESCRIPTION


Folder: Generate_data
Contain network file of Sioux Falls, EMA, Anaheim
Store OD demand for each Network

File Generate_OD: To generate OD demand matrix
File Generate_UE: To solve UE solution with Gurobi
File utils: contains functions to read network, solve UE solution
File check_UE: to verify the solution solved by Gurobi, draw charts

Folder: Solution
To store UE solution solved by Gurobi for all networks

Folder: Model
Contains the code of pre-processing data, transformer, and evaluating model.
multi: Code for multi-class network
single: Code for single-class network
multi_main.py: to run training and predicting for multi class network
single_main.py: to run training and predicting for for single network
single_main_network_test.py: to run training and predicting for for single network for the topological variation of SiouxFalls network
plotting.py: plotting charts

HOW TO USE:

Running
1. Check file Generate_OD  to generate new demand. 
2. Then resolve the UE solution with your new demand using Generate_UE.

3. Train your model:

a) For single-class network:
Edit the parameters in file Model/single/single_params.py
In Anaconda Powershell Prompt, cd to folder Model, run: python single_main.py

b) For single class network to test on topological variations:
Edit the parameters in file Model/single/single_params_network_test.py
In Anaconda Powershell Prompt, cd to folder Model, run: python single_main_network_test.py


4. For multi-class network:
Edit the parameters in file Model/multi/multi_params.py
In Anaconda Powershell Prompt, cd to folder Model, run: python multi_main.py