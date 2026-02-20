import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import networkx as nx
from itertools import islice
import random
import pickle
import time
import numpy as np
from tqdm.notebook import tqdm
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import ast
import glob
import os

def create_Grid_Net(dim1, dim2):
    G = nx.DiGraph()
    for n1 in range(dim1):
        for n2 in range(dim2):
            G.add_node((n1, n2))
    for n1 in range(dim1):
        for n2 in range(dim2):
            if n2 + 1 < dim2:
                G.add_edge((n1, n2), (n1, n2 + 1))
                G.add_edge((n1, n2 + 1), (n1, n2))  # bidirectional
            if n1 + 1 < dim1:
                G.add_edge((n1, n2), (n1 + 1, n2))
                G.add_edge((n1 + 1, n2), (n1, n2))  # bidirectional

    pos = {(x, y): (y, -x) for x, y in G.nodes()}  # for the drawing
    return G, pos

### GENERATE A SYNTHETIC NETWORK ####
def generate_gridNet(dim1, dim2, file_name, draw=True, target_links=80):
    G, pos = create_Grid_Net(dim1, dim2)
    G = reduce_links(G, target_links)
    i = 1
    mapping = {}
    for e in G.nodes:
        mapping[e] = i
        i = i + 1
    convert_net_to_file(G, file_name, mapping)
    if draw:
        nx.draw_networkx(G, pos=pos, with_labels=True, labels=mapping, font_size=9, font_color='white')
    return G, pos

def reduce_links(G, target_links):
    current_links = len(G.edges)
    if target_links >= current_links:
        return G  # No reduction needed
    
    edges = list(G.edges)
    random.shuffle(edges)
    edges_to_remove = edges[:current_links - target_links]
    
    for edge in edges_to_remove:
        G.remove_edge(*edge)
        
    return G

def convert_net_to_file(net, file_name,labels_map) : 
    with open(file_name, 'w') as f:
        f.write("<NUMBER OF ZONES> "+str(0))
        f.write("\n")
        f.write("<NUMBER OF NODES> "+str(len(net.nodes)))
        f.write("\n")
        f.write("<FIRST THRU NODE> 1")
        f.write("\n")
        f.write("<NUMBER OF LINKS> "+str(len(net.edges)))
        f.write("\n")
        f.write("<ORIGINAL HEADER>~ \tInit node \tTerm node \tCapacity \tLength \tFree Flow Time \tB \tPower \tSpeed limit \tToll \tType \t;")
        f.write("\n")
        f.write("<END OF METADATA>")
        f.write("\n")
        f.write("\n")
        f.write("\n")
        f.write("~ \tlink_id\tinit_node\tterm_node\tcapacity\tlength\tfree_flow_time\tb\tpower\tspeed\ttoll\tlink_type\t;")
        f.write("\n")
        
        link_id = 0
        for n1,n2 in net.edges :
            n1_id = labels_map[n1]
            n2_id = labels_map[n2]
            cap = np.random.randint(1000, 2001)
            leng = np.random.randint(20, 41)
            fft = round(np.random.uniform(0.5, 1.0),1)
            speed = np.random.choice([60, 70, 80, 90, 100])
            f.write("\t" + str(link_id)+"\t"+str(n1_id)+"\t"+str(n2_id)+"\t"+str(cap)+"\t"+str(leng)+"\t"+str(fft)+"\t0.15\t4\t"+str(speed)+"\t0\t0\t;")
            f.write("\n")
            link_id += 1
            
            
def add_link_ids_to_tntp(original_file, output_file):
    """
    Adds link_id to TTNP network file while preserving exact formatting:
    - Tabs as delimiter
    - No extra spaces
    - Preserves the original 8-line header
    """
    # Read the header (first 8 lines)
    with open(original_file, 'r') as f:
        header_lines = [next(f) for _ in range(8)]

    # Read the network data
    net = pd.read_csv(original_file, delimiter='\t', skiprows=8)

    # Insert link_id column
    net.insert(1, 'link_id', range(len(net)))

    # Strip column names
    net.columns = [c.strip() for c in net.columns]

    # Write back header first
    with open(output_file, 'w') as f:
        for line in header_lines:
            f.write(line)
        # Then write data manually without quotes and extra spaces
        net.to_csv(
            f,
            sep='\t',
            index=False,
            header=True,
            quoting=3,  # csv.QUOTE_NONE
            escapechar='\\',
            lineterminator='\n'
        )

    print(f" Saved TTNP network with link_ids to {output_file}")


def readNet(fileN) : 
    net = pd.read_csv(fileN,delimiter='\t',skiprows=8)

    nodes = set(list(net['init_node'])+list(net['term_node']))
    
    links = int(net.shape[0])
    cap = [0 for i in range(links)]
    t0 = [0 for i in range(links)]
    alpha = [0 for i in range(links)]
    beta = [0 for i in range(links)]
    lengths = [0 for i in range(links)]
    
    i = 0
    for capacityi,fftti,alphai,betai,leni in zip(net['capacity'],net['free_flow_time'],net['power'],net['b'], net['length']):
        cap[i] = capacityi
        t0[i] = fftti
        alpha[i] = alphai
        beta[i] = betai
        lengths[i] = leni
        i = i + 1
    return net, nodes, links, cap, t0, alpha, beta, lengths

##### READ MULTI-CLASS NETWORK #####
# def readNet(fileN) : 
#     net = pd.read_csv(fileN,delimiter='\t',skiprows=8) 
#     nodes = set(list(net['init_node'])+list(net['term_node']))
#     links = int(net.shape[0])
#     cap = [0 for i in range(links)]
#     t0 = [[0 for j in range(2)] for i in range(links)]
#     alpha = [0 for i in range(links)]
#     beta = [0 for i in range(links)]
#     lengths = [0 for i in range(links)]
    
#     for i, (capacityi,fftti,alphai,betai,leni) in enumerate(zip(net['capacity'],net['fft'],net['power'],net['b'], net['length'])):
#         fftti = ast.literal_eval(fftti)
#         cap[i] = capacityi
#         for j in range(2):
#             t0[i][j] = fftti[j]
#         alpha[i] = alphai
#         beta[i] = betai
#         lengths[i] = leni

#     return net, nodes, links, cap, t0, alpha, beta, lengths

def k_shortest_paths(G, source, target, k):
    try : 
        paths = list(islice(nx.shortest_simple_paths(G, source, target, weight="free_flow_time"), k))
    except : 
        paths = []

    return paths

def transform_paths(network, paths) : # transform node path to edge path
    paths_OD = []
    for path in paths:
        pathEdges = [] # list of links in each path
        for i in range(len(path)-1):
            mask = (network['init_node']==path[i]) & (network['term_node']==path[i+1])
            pathEdges.append(network.loc[mask, 'link_id'].values[0])   
            # pathEdges.append(network.index[mask].tolist()[0])   
        paths_OD.append(pathEdges)
    return paths_OD


def generate_OD_demand(num_nodes, min_demand, max_demand, num_pairs):
    od_demand = {}
    # num_pairs = int((num_nodes**2)*0.8)
    # Generate unique OD pairs
    pairs = set()
    while len(pairs) < num_pairs:
        origin = random.randint(1, num_nodes)
        destination = random.randint(1, num_nodes)
        if origin != destination:  # Ensure origin is not equal to destination
            pairs.add((origin, destination))

    # Assign random demand values to each OD pair
    for origin, destination in pairs:
        demand_c = random.randint(min_demand, max_demand)
        # demand_t = int(demand_c/2)
        # od_demand[(origin, destination)] = [demand_c, demand_t]
        od_demand[(origin, destination)] = demand_c
    return od_demand

#### MULTI-CLASS DEMAND GENERATOR ####
# def generate_OD_demand(num_nodes, min_demand, max_demand, num_pairs):
#     od_demand = {}
#     pairs = set()
#     while len(pairs) < num_pairs:
#         origin = random.randint(1, num_nodes)
#         destination = random.randint(1, num_nodes)
#         if origin != destination:  # Ensure origin is not equal to destination
#             pairs.add((origin, destination))

#     # Assign random demand values to each OD pair
#     for origin, destination in pairs:
#         demand_c = random.randint(min_demand, max_demand)
#         demand_t = int(demand_c/2)
#         od_demand[(origin, destination)] = [demand_c, demand_t]
#     return od_demand

def find_paths(network, OD_Matrix, k) :
#def find_paths(netG, network, OD_Matrix,k) :
    
    netG = nx.from_pandas_edgelist(network,source='init_node',target='term_node',edge_attr='free_flow_time',create_using=nx.DiGraph())
    #print('Inside Find Paths func...')
    paths = {}
    paths_N = {}
    for key in OD_Matrix.keys():
        paths_OD = []
        o,d = key[0],key[1]
        try : 
            p = k_shortest_paths(netG,o,d,k)
            paths_OD = transform_paths(network, p) # paths_OD is list of feasible path set, each path includes list of links
            paths[(o,d)]=paths_OD
            paths_N[(o,d)]=p
        except :
            paths[(o,d)] = []
            paths_N[(o,d)]=[]
    
            
    return paths, paths_N

def translate(nodes, OD_demand) :
    Q = [value for value in OD_demand.values()]
    OD = len(OD_demand)
    O_D = [ [0 for i in nodes] for n in range(OD)]
    n= 0
    
    for key in OD_demand.keys() :
        for i in nodes : 
            if i==key[0] : #origin node
                O_D[n][i-1] = -1
            if i==key[1] : #destination node
                O_D[n][i-1] = 1
        n = n + 1
    
    return Q, OD, O_D

def create_Adj(net_df, links, nodes): 
    # links: number of links in the network (80, 75, 70, 65)
    # nodes: a list of all nodes (from 1 to 25)
    # Create adj matrix shape 25x80 for all scenarios
    Adj = [ [0 for i in range(links)] for n in nodes]
    print(links, nodes)
    for n in nodes :
        init = net_df[net_df['init_node']==n]
        for j in init['link_id'].values : 
            #print(n,j)
            Adj[n-1][j] = -1
            
        
        term = net_df[net_df['term_node']==n]
        for j in term['link_id'].values : 
            Adj[n-1][j] = 1
    return Adj

def create_delta(links, paths, od_matrix) :
    delta = [[[0 for i in range(links)] for p in paths[k]] for k in od_matrix.keys()]

    kk = 0
    for k in od_matrix.keys() : 
        value = paths[k] # get all paths of OD pair k
        pp = 0
        for p in value : # iterate each path of pair k
            for j in p : # iterate each link in path p
                delta[kk][pp][j]=1
            pp += 1
        kk += 1
    return delta

def get_origDest(OD_demand) : 
    orig = []
    dest = []
    for o,d in OD_demand.keys() :
        if o not in orig :
            orig.append(o)
        if d not in dest :
            dest.append(o)
    return orig, dest


# This function receives a fix path set dictionary for all OD demand

# FOR SINGLE CLASS NETWORK

def get_data(Network, Nodes, links, cap, fft, alpha, beta, lengths, OD_mat, paths) : 
    O,D = get_origDest(OD_mat)
    Q, OD, O_D = translate(Nodes, OD_mat)
    Adj = create_Adj(Network, links, Nodes)
    delta = create_delta(links, paths, OD_mat) #Adj of OD pair - path - link
    n = [ len(paths[h]) for h in OD_mat.keys() ]
    ### Linearizing variables
    seg = 1000
    Mflow = 10e4
    # define the segments
    segments = set([i for i in range(0,seg+1)])   
    eta = [ [ v for v in segments ] for i in range(links) ]  
    #step = Mflow/seg 
    
# ---------------- CHECK TOTAL LINK FLOW vs ETA_MAX ----------------
    total_link_flow = [0.0 for _ in range(links)]
    for od, paths_list in paths.items():  # paths dict: OD -> list of paths
        demand = OD_mat[od]
        for path in paths_list:
            for link in path:
                total_link_flow[link] += demand

    # for i in range(links):
    #     eta_max = max(eta[i])
    #     if total_link_flow[i] > eta_max:
    #         print(f" Link {i} has total flow {total_link_flow[i]:.2f} exceeding η_max={eta_max:.2f}")
    
#_________________________________________________________________________________

    for i in range(links):
        cnt = 0
        max_flow = max(Mflow, total_link_flow[i]*1.05)  # 5% safety margin
        step = max_flow / seg
        #step = cap[i]/seg
        for v in segments:
            eta[i][v] = cnt*step
            cnt += 1  
    #segments_p = segments.difference({0}) 
 #__________________________________________________________________________________________   
    data = {'network' :Network, 'demand' :OD_mat, 'nodes':Nodes,'links':links,'orig':O,'dest':D,'fftt':fft,'capacity':cap, 'length': lengths, 'beta':beta,
        'approx':segments,'eta':eta,'paths_link':paths, 'delta':delta,'alpha':alpha, 'Adjacency_matrix' : Adj}
    return data, Q, OD, O_D,n

### FOR MULTI-CLASS NETWORK###
# def get_data(Network, Nodes, links, cap, fft, alpha, beta, lengths, OD_mat, paths) : 
#     # print("No of links: ", links)
#     O,D = get_origDest(OD_mat)
#     Q, OD, O_D = translate(Nodes, OD_mat)
#     Adj = create_Adj(Network, links, Nodes)
#     delta = create_delta(links, paths, OD_mat)
#     n = [ len(paths[h]) for h in OD_mat.keys() ] # number of path for each OD pair 
#     ### Linearizing variables
#     seg = 1000
#     Mflow = 10e4   

#     # define the segments
#     segments = set([i for i in range(0,seg+1)])          
#     eta = [[[ v for j in range(2)] for v in segments] for i in range(links)]    
#     for i in range(links):
#         cnt = 0
#         step = Mflow/seg
#         for v in segments:
#             for j in range(2):
#                 eta[i][v][j] = cnt*step
#             cnt += 1  
  
#     data = {'network' :Network, 'demand' :OD_mat, 'nodes':Nodes,'links':links,'orig':O,'dest':D,'fftt_c':fft[0], 'fftt_t':fft[1], 'capacity':cap, 'length': lengths, 'beta':beta,
#         'approx':segments,'eta':eta,'paths_link':paths, 'delta':delta,'alpha':alpha, 'Adjacency_matrix' : Adj}
#     return data, Q, OD, O_D,n

#################### BRUE SOLVER ##########################

def BRUE(data, n, OD, Q):
    model = gp.Model("BRUE")
    model.setParam("OutputFlag", 0)
    # model.setParam("LogFile", "gurobi_log.txt")
    model.setParam("NumericFocus", 3)
    model.setParam("Method", 2)  # Barrier method

    a = data['links']
    segments = data['approx']
    t0 = data['fftt']
    eta = data['eta']
    alpha = data['alpha'] #4
    beta = data['beta'] #0.15
    sigma = data['delta'] # adj of od pair - path - link 
    cap = data['capacity']
    segments_p = segments.difference({0})
    M = 1e6
    
    x = [model.addVar(vtype=GRB.CONTINUOUS) for j in range(a)] # link flow
    x4 = [model.addVar(vtype=GRB.CONTINUOUS) for j in range(a)] # x^4
    link_cost = [model.addVar(vtype=GRB.CONTINUOUS) for j in range(a)]
    f = [ [model.addVar(vtype=GRB.CONTINUOUS) for i in range(n[k])] for k in range(OD)] # path flow
    path_cost = [ [model.addVar(vtype=GRB.CONTINUOUS) for i in range(n[k])] for k in range(OD)]
    
    ll = [ [model.addVar(vtype=GRB.CONTINUOUS) for l in segments] for i in range(a)]
    lr = [ [model.addVar(vtype=GRB.CONTINUOUS) for l in segments] for i in range(a)]

    y = [ [model.addVar(vtype=GRB.CONTINUOUS) for i in range(n[k])] for k in range(OD)] # BINARY VAR
    min_path_cost = [model.addVar(vtype=GRB.CONTINUOUS, name=f"min_path_cost_{k}") for k in range(OD)]

    for i in range(a):
        model.addConstr(x[i] == sum( sum(f[k][p]*sigma[k][p][i] for p in range(n[k])) for k in range(OD)), "link-path%d" %i)
        model.addConstr(x[i] == sum(ll[i][v]*eta[i][v-1] + lr[i][v]*eta[i][v] for v in segments_p), "Approx1%d" %i)
        model.addConstr(sum(ll[i][v] + lr[i][v] for v in segments_p) == 1, "Approx2%d" %i)
        model.addConstr(x[i] >= 1e-6, "integrality_x%d" %i)  

        for ss in segments:
            model.addConstr(ll[i][ss] >= 0, "integrality_ll%d%d" %(i,ss))
            model.addConstr(lr[i][ss] >= 0, "integrality_lr%d%d" %(i,ss))
        
        model.addGenConstrPow(x[i], x4[i], 4, name="x_to_the_power_of_4")

        model.addConstr(link_cost[i] == t0[i] * (1 + beta[i]/(cap[i]**alpha[i]) * 
                                                 sum(eta[i][v-1]**alpha[i] * ll[i][v] + eta[i][v]**alpha[i] * lr[i][v] 
                                                     for v in segments_p)),
                                                 name=f"link_cost_BPR_{i}")
        # model.addConstr(link_cost[i] == t0[i] * (1 + beta[i]/(cap[i]**alpha[i]) * x4[i]), name=f"link_cost_BPR_{i}")

    for k in range(OD) :
        model.addConstr( sum(f[k][p] for p in range(n[k])) == Q[k] , "FConservation%d" %k ) 

        if (n[k] > 0):
            model.addConstr(min_path_cost[k] == path_cost[k][0], name=f"init_min_cost_{k}")

        for p in range(n[k]) : 
            model.addConstr(f[k][p] >= 0, "integrality%d%d" %(k,p))

            model.addConstr(y[k][p] >= 0)
            model.addConstr(y[k][p] <= 1)
            model.addConstr(f[k][p] <= M * y[k][p]) # if f[k][i] = 0 then y[k][i] = 0
            model.addConstr(f[k][p] >= 1e-6 * y[k][p]) # if f[k][i] > 0 then y[k][i] = 1

            model.addConstr(path_cost[k][p] == sum(link_cost[i] * sigma[k][p][i] for i in range(a)), "path-cost%d%d" %(k,p)) # satisfied
            model.addConstr((min_path_cost[k] <= path_cost[k][p]), name=f"min_cost_{k}") # satisfied

            model.addConstr(path_cost[k][p] - min_path_cost[k] <= M* (1-y[k][p]) + min_path_cost[k]*0.05) #BRUE
            # model.addConstr(f[k][p] * (1-y[k][p]) <= f[k][p])
            # model.addConstr(f[k][p] * y[k][p] == f[k][p]) # add this one make infeasible
            # model.addConstr(M * (path_cost[k][p] - min_path_cost[k]) >= 1-y[k][p])
        
    Z = sum(M * y[k][p] for p in range(n[k]) for k in range(OD))

    model.setObjective(Z, GRB.MINIMIZE)
    t1 = time.time()
    model.optimize()
    t2 = time.time()
    print('model solved in:',t2-t1)
    
    if model.Status == GRB.OPTIMAL:
        flows =  [ [ f[k][p].X  for p in range(n[k])] for k in range(OD)]
        linkss = [ x[i].X  for i in range(a)]
        x4 = [ x4[i].X  for i in range(a)]
        min_cost = [min_path_cost[i].X for i in range(OD)]
        path_cost = [[path_cost[k][p].X for p in range(n[k])] for k in range(OD)]
        link_cost = [link_cost[i].X for i in range(a)]
        return flows, linkss, path_cost, min_cost, link_cost, x4
    
    elif model.Status == GRB.INFEASIBLE:
        model.computeIIS()
        model.write("model.ilp")
        print(f"Model is infeasible, status {model.Status}")
        return None, None, None, None, None, None
    
    else:
        print(f"Model did not solve to optimality. Status: {model.Status}")
        return None, None, None, None, None, None

################ SINGLE CLASS UE SOLVER ####################

def TA_MSA(data, n, Q, max_iter=1000, tol=1e-4):
    """
    Traffic Assignment using Method of Successive Averages (MSA)
    ------------------------------------------------------------
    Inputs:
        data : dict   - contains 'links', 'fftt', 'capacity', 'alpha', 'beta', 'paths_link', 'demand'
        n : list/int  - not directly used here (can be number of paths per OD)
        Q : list or dict - OD demands (optional, redundant since in data['demand'])
    """



    # --- Extract info ---
    num_links = data["links"] if isinstance(data["links"], int) else len(data["links"])
    fft = data["fftt"]
    cap = data["capacity"]
    alpha = data["alpha"]
    beta = data["beta"]
    paths = data["paths_link"]
    demand_dict = data.get("demand", {})

    # --- Initialize ---
    link_flow = np.zeros(num_links)
    tt = np.array(fft, dtype=float)  # current travel times

    # --- Detect if link IDs are 1-based ---
    # If any path link ID equals num_links, assume 1-based indexing
    sample_path = next(iter(paths.values()))[0]
    one_based = max(sample_path) == num_links
    offset = 1 if one_based else 0
    print(f" Detected {'1-based' if one_based else '0-based'} link IDs (offset={offset})")

    # --- Initialize path flows dict ---
    path_flow = {k: [0.0] * len(paths[k]) for k in paths.keys()}

    for it in range(1, max_iter + 1):
        print(f"\n MSA Iteration {it}")

        # Update travel times using BPR function
        for a in range(num_links):
            flow = max(link_flow[a], 0)
            tt[a] = fft[a] * (1 + alpha[a] * (flow / cap[a]) ** beta[a])

        # All-or-Nothing (AON) assignment
        aux_link_flow = np.zeros(num_links)

        for (o, d), demand in demand_dict.items():
            if demand <= 0 or (o, d) not in paths:
                continue

            shortest_path = min(
                paths[(o, d)], key=lambda p: sum(tt[a - offset] for a in p)
            )

            for a in shortest_path:
                aux_link_flow[a - offset] += demand

        # Update link flows using MSA averaging
        link_flow = link_flow + (1.0 / it) * (aux_link_flow - link_flow)

        # Compute relative gap
        diff = np.sum(np.abs(aux_link_flow - link_flow))
        total = np.sum(np.abs(link_flow))
        gap = diff / total if total > 0 else 0

        print(f"   Gap = {gap:.6f}")

        if gap < tol and it > 3:
            print(" Converged!")
            break

    # --- Assign flows to path_flow for inspection ---
    for (o, d), demand in demand_dict.items():
        if demand <= 0 or (o, d) not in paths:
            continue
        shortest_path = min(
            paths[(o, d)], key=lambda p: sum(tt[a - offset] for a in p)
        )
        path_flow[(o, d)] = [demand if p == shortest_path else 0 for p in paths[(o, d)]]

    return list(path_flow.values()), link_flow.tolist()
    
def TA_UE(data, n, OD, Q):
    model = gp.Model("UE")
    model.setParam("OutputFlag", 0)

    model.Params.Threads = 10         # use all cores (or set numeric)

    a = data['links']
    segments = data['approx']
    t0 = data['fftt']
    eta = data['eta']
    alpha = data['alpha']
    beta = data['beta']
    sigma = data['delta']
    cap = data['capacity']
    #segments = list(sorted(data['approx']))        # ensure it's an ordered list
    #segments_p = [v for v in segments if v != 0]   # keep all except 0
    
    segments_p = segments.difference({0})
    


    
    x = [model.addVar(vtype=GRB.CONTINUOUS) for j in range(a)]
    f = [ [model.addVar(vtype=GRB.CONTINUOUS) for i in range(n[k])] for k in range(OD)]
    ll = [ [model.addVar(vtype=GRB.CONTINUOUS) for l in segments] for i in range(a)]
    lr = [ [model.addVar(vtype=GRB.CONTINUOUS) for l in segments] for i in range(a)]

    
    for i in range(a):
        model.addConstr(x[i] == sum( sum(f[k][p]*sigma[k][p][i] for p in range(n[k])) for k in range(OD)), "link-path%d" %i)
        model.addConstr(x[i] == sum(ll[i][v]*eta[i][v-1] + lr[i][v]*eta[i][v] for v in segments_p), "Approx1%d" %i)
        model.addConstr(sum(ll[i][v] + lr[i][v] for v in segments_p) == 1, "Approx2%d" %i)
        model.addConstr(x[i] >= 0, "integrality_x%d" %i)  
        for ss in segments:
            model.addConstr(ll[i][ss] >= 0, "integrality_ll%d%d" %(i,ss))
            model.addConstr(lr[i][ss] >= 0, "integrality_lr%d%d" %(i,ss))

    for k in range(OD) :
        model.addConstr( sum(f[k][p] for p in range(n[k])) == Q[k] , "FConservation%d" %k ) 
        for p in range(n[k]) : 
            model.addConstr(f[k][p] >= 0, "integrality%d%d" %(k,p))
    
    Z = sum( t0[i]*(x[i]+1/(alpha[i]+1)*beta[i]/(cap[i]**alpha[i])*
                    sum(eta[i][v-1]**(alpha[i]+1)*ll[i][v]+eta[i][v]**(alpha[i]+1)*lr[i][v] for v in segments_p))
                    for i in range(a) )    

    model.setObjective(Z, GRB.MINIMIZE)
    t1 = time.time()
    model.optimize()
    t2 = time.time()
    print('model solved in:',t2-t1)
    
    # After model.optimize()
    print("Status:", model.Status, "SolCount:", model.SolCount)
    
    if model.SolCount > 0:  # there is at least one feasible (incumbent) solution
        print(" Feasible (possibly suboptimal) solution found.")
        flows = [[f[k][p].X for p in range(n[k])] for k in range(OD)]
        linkss = [x[i].X for i in range(a)]
    else:
        print("  No feasible solution found. Status:", model.Status)
        flows = [[0 for p in range(n[k])] for k in range(OD)]
        linkss = [0 for i in range(a)]
    
    # Handle specific statuses
    if model.Status == GRB.INFEASIBLE:
        print(" Model infeasible. Computing IIS...")
        model.computeIIS()
        model.write("infeasible_model.ilp")
        print("IIS written to infeasible_model.ilp")
    
    elif model.Status == GRB.TIME_LIMIT:
        print(" Time limit reached — extracted best available incumbent flows.")
    
    elif model.Status == GRB.SUBOPTIMAL:
        print(" Suboptimal solution (solver stopped early). Using incumbent flows.")
    
    elif model.Status == GRB.INTERRUPTED:
        print(" Optimization interrupted — using partial solution if available.")
    
    # Print objective safely
    if model.SolCount > 0:
        print("Objective value:", model.objVal)
    else:
        print("Objective value: N/A (no feasible solution)")

    return flows, linkss
    
############## MULTI CLASS UE SOLVER ###############
# def TA_UE(data, n, OD, Q):
#     # n: number of path of each OD pair
#     # OD: number of OD pair 
#     # Q: value of demand 
#     model = gp.Model("UE")
#     model.setParam("OutputFlag", 0)

#     a = data['links']
#     segments = data['approx']
#     t0 = data['fftt']
#     eta = data['eta']
#     alpha = data['alpha']
#     beta = data['beta']
#     sigma = data['delta']
#     cap = data['capacity']
#     segments_p = segments.difference({0})

#     pi = [[1, 2.5] for i in range(a)]
    
#     x = [[model.addVar(vtype=GRB.CONTINUOUS) for j in range(2)]for i in range(a)]
#     f = [ [[model.addVar(vtype=GRB.CONTINUOUS) for j in range(2)]for i in range(n[k])] for k in range(OD)]
#     ll = [ [[model.addVar(vtype=GRB.CONTINUOUS) for j in range(2)] for l in segments] for i in range(a)]
#     lr = [ [[model.addVar(vtype=GRB.CONTINUOUS) for j in range(2)] for l in segments] for i in range(a)]

#     for i in range(a):
#         # x[i][1] = x[i][1] * 2.5
#         for j in range(2):
#             model.addConstr(x[i][j] == sum(sum(f[k][p][j]*sigma[k][p][i] for p in range(n[k])) for k in range(OD)), "link-path%d" %i)
#             model.addConstr(x[i][j] == sum(ll[i][v][j]*eta[i][v-1][j] + lr[i][v][j]*eta[i][v][j] for v in segments_p), "Approx1%d" %i)

#             model.addConstr(sum(ll[i][v][j] + lr[i][v][j] for v in segments_p) == 1, "Approx2%d" %i)
#             model.addConstr(x[i][j] >= 0, "integrality_x%d" %i)   

#             for ss in segments:
#                 model.addConstr(ll[i][ss][j] >= 0, "integrality_ll%d%d" %(i,ss))
#                 model.addConstr(lr[i][ss][j] >= 0, "integrality_lr%d%d" %(i,ss))

#     for k in range(OD) :
#         for j in range(2):
#             model.addConstr( sum(f[k][p][j] for p in range(n[k])) == Q[k][j] , "FConservation%d" %k ) 
#             for p in range(n[k]) : 
#                 model.addConstr(f[k][p][j] >= 0, "integrality%d%d" %(k,p))
    
#     Z = sum( t0[i][j] * (x[i][j]+1/(alpha[i]+1)*beta[i]*(pi[i][j]**alpha[i])/(cap[i]**alpha[i])*
#                     sum(eta[i][v-1][j]**(alpha[i]+1)*ll[i][v][j] + eta[i][v][j]**(alpha[i]+1)*lr[i][v][j] for v in segments_p))
#                     for j in range(2)
#                     for i in range(a))

#     model.setObjective(Z, GRB.MINIMIZE)
#     t1 = time.time()
#     model.optimize()
#     t2 = time.time()
#     print('model solved in:',t2-t1)
    
#     if model.Status == GRB.OPTIMAL:
#         flows =  [[[f[k][p][j].X for j in range(2)] for p in range(n[k])] for k in range(OD)]
#         linkss = [[x[i][j].X for j in range(2)] for i in range(a)]
#         # linkss = None
#         return flows, linkss
    
#     elif model.Status == GRB.INFEASIBLE:
#         print(f"Model is infeasible, status {model.Status}")
#         model.computeIIS()
#         model.write("model.ilp")
#         return None, None
    
#     else:
#         print(f"Model did not solve to optimality. Status: {model.Status}")
#         return None, None

def read_file(filename):
  with open(filename, "rb") as file:
      stat = pickle.load(file)
      file.close()
  return stat

# This function get all feasible paths of all OD pairs.
# Each OD pair has 3 paths
# From this origin path set, when remove any link, we will remove the path containing that link, other paths keep no change.

def _process_od_pair_filtered_timed_find_paths(args):
    """
    Worker: compute paths for a single OD pair using find_paths and return results silently.
    """
    net_df, od_pair, path_num = args

    
    start = time.time()
    paths_transformed = []

    try:
        # Wrap OD pair into a small dict as expected by find_paths
        od_matrix = {od_pair: 1}
        paths_dict, _ = find_paths(net_df, od_matrix, path_num)
        print('finished find paths:', paths_dict[od_pair])

        # Extract the result
        if od_pair in paths_dict:
            paths_transformed = paths_dict[od_pair]

    except Exception as e:
        # Return an empty list if any error occurs
        paths_transformed = []

    end = time.time()
    duration = end - start

    return od_pair, paths_transformed, duration


def get_full_paths_from_folder_filtered_timed_find_paths(
    demand_dir,
    net_file,
    path_num,
    limit=None,
    num_processes=None,
    path_file='unique_paths.pkl',
    pair_file='pair_path.pkl'
):
    """
    Generate paths for unseen OD pairs across OD matrices using find_paths.
    Each matrix processed sequentially.
    Each OD pair processed in parallel with timing info.
    """

    # Load network
    net_df = pd.read_csv(net_file, delimiter='\t', skiprows=8)
    # Check
    print(net_df.head())
    print(net_df.dtypes)
    print(net_df[['init_node','term_node','link_id']].head())
  
    # Collect OD matrix files and randomly sample if needed
    pkl_files = sorted(glob.glob(os.path.join(demand_dir, "*.pkl")))
    
    # If only one file exists, check if it contains multiple OD matrices
    if len(pkl_files) == 1:
        single_file = pkl_files[0]
        print(f" Found single OD file: {os.path.basename(single_file)} — checking contents...")
    
        with open(single_file, 'rb') as f:
            data = pickle.load(f)
    
        # If it's a list of OD matrices, split them into separate files
        if isinstance(data, list) and len(data) > 1:
            print(f" Detected {len(data)} OD matrices inside — splitting...")
            for i, od_matrix in enumerate(data):
                out_path = os.path.join(demand_dir, f"od_matrix_{i+1}.pkl")
                if not os.path.exists(out_path):  # avoid overwriting if already split
                    with open(out_path, 'wb') as f_out:
                        pickle.dump(od_matrix, f_out)
            print(f" Split complete. Created {len(data)} files.")
        else:
            print(" Single OD matrix only — using as is.")
    
    # Reload after possible split
    pkl_files = sorted(glob.glob(os.path.join(demand_dir, "*.pkl")))
    if limit and len(pkl_files) > limit:
        random.seed(42)
        pkl_files = random.sample(pkl_files, limit)
    print(f"🎲 Selected {len(pkl_files)} OD matrices")

    if num_processes is None:
        num_processes = max(1, min(60, os.cpu_count() - 1))

    # Load existing results
    if os.path.exists(path_file) and os.path.exists(pair_file):
        with open(path_file, 'rb') as f:
            global_path_set = pickle.load(f)
        with open(pair_file, 'rb') as f:
            global_pair_path = pickle.load(f)
        print(f" Loaded {len(global_path_set)} paths, {len(global_pair_path)} OD pairs")
    else:
        global_path_set = set()
        global_pair_path = {}

    # Process matrices sequentially
    

    for matrix_idx, od_file in enumerate(pkl_files):
        print(f"\n Processing matrix {matrix_idx+1} / {len(pkl_files)}: {os.path.basename(od_file)}")

        with open(od_file, 'rb') as f:
            OD_matrix = pickle.load(f)

        unseen_pairs = [pair for pair in OD_matrix.keys() if pair not in global_pair_path]
        total_pairs = len(unseen_pairs)
        print(f" Total unseen OD pairs in this matrix: {total_pairs}")

        if not unseen_pairs:
            continue

        args_list = [(net_df, pair, path_num) for pair in unseen_pairs]

        cumulative_time = 0.0
        completed_pairs = 0
        
        start = time.time()

        # Process OD pairs in parallel
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            print(f"Submitting {len(args_list)} OD pairs to {num_processes} workers...")

            futures = {executor.submit(_process_od_pair_filtered_timed_find_paths, args): args[1] for args in args_list}

            for future in as_completed(futures):
                od_pair = futures[future]
                try:
                    pair, paths, duration = future.result()
                    cumulative_time += duration
                    completed_pairs += 1
                    remaining_pairs = total_pairs - completed_pairs
                    avg_time = cumulative_time / completed_pairs
                    eta_sec = remaining_pairs * avg_time
                    eta_min = eta_sec / 60
                    print(f"⏱ OD pair {pair} of Matrix {matrix_idx+1} done in {duration:.2f}s | ETA for remaining: {eta_min:.1f} min",flush=True)

                    global_pair_path[pair] = paths
                    for path in paths:
                        global_path_set.add(tuple(path))
                except Exception as e:
                    print(f" Failed OD pair {od_pair}: {e}", flush=True)

        end = time.time()
        total_time=end-start
        
        # Incremental save per matrix
        with open(path_file, 'wb') as f:
            pickle.dump(global_path_set, f)
        with open(pair_file, 'wb') as f:
            pickle.dump(global_pair_path, f)

        print(f" Completed matrix {matrix_idx+1} in {total_time:.2f}s . Total OD pairs now: {len(global_pair_path)}", flush=True)

    path_set_dict = {v: k for k, v in enumerate(global_path_set, start=1)}
    print(f"\n Path generation complete: {len(global_path_set)} unique paths, {len(global_pair_path)} OD pairs", flush=True)
    return path_set_dict, global_pair_path


        
def solve_UE(net_file, demand_file, pair_path, output_file, base_number, to_solve):
    """
    Solves UE for :
      - A folder containing multiple OD matrices 

    
    Behavior:
    - Skips OD pairs with no paths
    - Stops program if saving fails due to disk space
    """


    Network, Nodes, links, cap, fft, alpha, beta, lengths = readNet(net_file)

    # --- Detect OD demand file ---
    if os.path.isfile(demand_file) and demand_file.endswith(".pkl"):
        print(f"🔹 Loading single OD matrix from {demand_file}")
        with open(demand_file, 'rb') as f:
            OD_matrix = pickle.load(f)
        stat = [OD_matrix]  


    time_step = 0
    for OD_matrix in tqdm(stat[:to_solve]):
        print(f"\n Solving UE for OD_Matrix {base_number}")

        # --- Identify missing OD pairs ---
        od_pairs = set(OD_matrix.keys())
        path_pairs = set(pair_path.keys())
        missing_pairs = od_pairs - path_pairs
        available_pairs = od_pairs & path_pairs

        total_pairs = len(od_pairs)
        missing_count = len(missing_pairs)
        missing_pct = (missing_count / total_pairs) * 100 if total_pairs > 0 else 0

        print(f" Total OD pairs: {total_pairs}")
        print(f" Pairs with paths: {len(available_pairs)} ({100 - missing_pct:.2f}%)")
        if missing_count > 0:
            print(f" Missing pairs: {missing_count} ({missing_pct:.2f}%)")
            print(f"   Examples of missing pairs: {list(missing_pairs)[:5]}")

        # --- Skip if no pairs have paths ---
        if len(available_pairs) == 0:
            print(f" No feasible OD pairs in iteration {time_step}. Skipping this UE run.")
            time_step += 1
            continue

        # --- Extract relevant paths ---
        paths = {k: pair_path[k][:3] if len(pair_path[k]) >= 3 else pair_path[k] for k in available_pairs}
        filtered_OD_matrix = {k: OD_matrix[k] for k in available_pairs}

        # --- Build network data ---
        data, Q, OD, O_D, n = get_data(Network, Nodes, links, cap, fft, alpha, beta, lengths,
                                       filtered_OD_matrix, paths)

        # --- Solve UE ---
        flows, linkss = TA_UE(data, n, OD, Q)
        dataa = {'data': data, 'path_flow': flows, 'link_flow': linkss}
        
        # # --- Solve UE MSA ---
        # flows, linkss = TA_MSA(data, n, Q)
        # dataa = {'data': data, 'path_flow': flows, 'link_flow': linkss}

               # --- Save output ---
        try:
            with open(output_file, "wb") as f:  # use output_file directly
                pickle.dump(dataa, f)
            #print(f" Saved UE result: {output_file}")
        except OSError as e:
            if "No space" in str(e) or "disk full" in str(e).lower():
                print(" ERROR: No disk space left. Stopping program.")
                import sys
                sys.exit(1)
            else:
                raise e


        time_step += 1




def solve_BRUE(net_file, demand_file, pair_path, output_file, to_solve):
    stat = read_file(demand_file)
    Network, Nodes, links, cap, fft, alpha, beta, lengths = readNet(net_file)

    time = 0
    for OD_matrix in tqdm(stat[:to_solve]):
        print(time)
        paths = {k: (pair_path[k][:3] if len(pair_path[k]) >= 3 else pair_path[k]) for k in OD_matrix.keys()}
        data, Q, OD, O_D,n = get_data(Network, Nodes, links, cap, fft, alpha, beta, lengths, OD_matrix, paths)
        flows, linkss, path_cost, min_cost, link_cost, x4 = BRUE(data, n, OD, Q)
        if flows != None:
            dataa = {'data' : data, 'path_flow' : flows, 'link_flow' : linkss, 'path_cost': path_cost, 'min_cost': min_cost, 'link_cost': link_cost, 'x4': x4}
            # flows, linkss = BRUE(data, n, OD, Q)
            # dataa = {'data' : data, 'path_flow' : flows, 'link_flow' : linkss}
            file_data = open(output_file+str(time), "wb")
            pickle.dump(dataa , file_data)
            file_data.close()
            time +=1
        else:
            print(f"Can't solve OD matrix {time}")
            time += 1

# This function remove the link from the feasible path when that link is removed from the network
def remove_links_from_path(pair_path, remove_ids):
    remove_ids = set(remove_ids)
    new_dict = defaultdict(list)
    for key, value in pair_path.items():
        v = [p for p in value if not any(link in p for link in remove_ids)]
        new_dict[key] = v      
    return dict(new_dict)

def remove_links_from_tntp(input_file, output_file, remove_ids):
    with open(input_file, 'r') as file:
        lines = file.readlines()
    remove_ids = set(remove_ids)

    modified_lines = []
    for line in lines:
        parts = line.split()
        if parts and parts[0].isdigit() and int(parts[0]) in remove_ids:
            # Replace all values from column 2 onward with '0'
            parts[1:] = ['0'] * (len(parts) - 2)
            modified_line = '\t' + '\t'.join(parts) + '\t;\n'
            modified_lines.append(modified_line)
        else:
            modified_lines.append(line)
    
    with open(output_file, 'w') as file:
        file.writelines(modified_lines)
