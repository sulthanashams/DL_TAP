import os
import re
import random
import pickle
from tqdm import tqdm


def read_tntp_trip_file(file_path):
    """
    Parses a TNTP-format OD trip file and returns a dict {(origin, destination): demand}.
    """
    od_demand = {}
    current_origin = None
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Detect new origin
            if line.lower().startswith("origin"):
                current_origin = int(line.split()[1])
            
            # Detect OD pairs and values
            elif ":" in line and current_origin is not None:
                pairs = re.findall(r"(\d+)\s*:\s*([\d.]+)", line)
                for dest, val in pairs:
                    dest = int(dest)
                    demand = float(val)
                    if current_origin != dest:  # skip diagonal
                        od_demand[(current_origin, dest)] = demand
                        
    return od_demand


# Step 2: Generate and save scaled OD variations (with missing values)
def generate_and_save_variations(base_od, save_dir, num_variations=4000, 
                                 scale_min=0.2, scale_max=1.0, missing_frac=0.3):
    """
    base_od: dict {(origin, destination): demand}
    missing_frac: fraction of OD pairs to remove randomly (e.g., 0.3 = 30%)
    """
    os.makedirs(save_dir, exist_ok=True)
    file_paths = []

    all_ods = list(base_od.keys())
    total_pairs = len(all_ods)
    num_missing = int(total_pairs * missing_frac)

    for i in tqdm(range(num_variations), desc="Generating OD matrices"):
        try:
            file_path = os.path.join(save_dir, f"od_matrix_{i}.pkl")
            if os.path.exists(file_path):
                file_paths.append(file_path)
                continue  # skip existing

            # Random scaling
            new_od = {od: round(d * random.uniform(scale_min, scale_max), 2) for od, d in base_od.items()}
            
            # Random missing pairs
            missing_od_pairs = random.sample(all_ods, num_missing)
            for od in missing_od_pairs:
                del new_od[od]

            with open(file_path, 'wb') as f:
                pickle.dump(new_od, f)
            file_paths.append(file_path)
        except Exception as e:
            print(f" Error generating matrix {i}: {e}")
    
    return file_paths


# Step 3: Diagnostic check
def diagnostic_check(file_paths, base_od, num_samples=5):
    """
    Load two random OD matrices and compare random OD pairs to the baseline.
    """
    if len(file_paths) < 2:
        print("Not enough OD matrices to compare. Skipping diagnostic.")
        return
    
    sample_files = random.sample(file_paths, 2)
    print(f"\n🔍 Diagnostic Check: Comparing {os.path.basename(sample_files[0])} vs {os.path.basename(sample_files[1])}\n")

    with open(sample_files[0], 'rb') as f:
        od_1 = pickle.load(f)
    with open(sample_files[1], 'rb') as f:
        od_2 = pickle.load(f)

    sample_ods = random.sample(list(base_od.keys()), num_samples)
    for od in sample_ods:
        base_val = base_od[od]
        val1 = od_1.get(od)
        val2 = od_2.get(od)
        if val1 is not None and val2 is not None:
            print(f"OD {od}: base={base_val:.2f}, var1={val1:.2f}, var2={val2:.2f}, "
                  f"scales=({val1/base_val:.2f}, {val2/base_val:.2f})")
        elif val1 is None and val2 is None:
            print(f"OD {od}: missing in both matrices ")
        elif val1 is None:
            print(f"OD {od}: missing in matrix 1 ")
        elif val2 is None:
            print(f"OD {od}: missing in matrix 2 ")


# --- Usage ---
if __name__ == "__main__":
    base_file = r"E:\sshams\DL_TAP\Path_Flow_Prediction-main\Generate_data\Anaheim\Anaheim_trips.tntp"
    save_dir = r"E:\sshams\DL_TAP\Path_Flow_Prediction-main\Generate_data\Anaheim\OD_Matrices"
   
    base_od = read_tntp_trip_file(base_file)
    print(f"Parsed {len(base_od)} OD pairs from {os.path.basename(base_file)}")

    # Print min and max OD demand values
    od_values = list(base_od.values())
    print(f" Base OD demand range → min: {min(od_values):.2f}, max: {max(od_values):.2f}")


    
    file_paths = generate_and_save_variations(
        base_od, save_dir, num_variations=4000, scale_min=0.2, scale_max=1.0, missing_frac=0.5
    )

    print("Finished generating all OD variations.")
    diagnostic_check(file_paths, base_od, num_samples=8)




    
    
 
