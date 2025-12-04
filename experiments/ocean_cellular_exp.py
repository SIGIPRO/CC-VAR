from pathlib import Path
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from ccvar import CCVAR
from tqdm import tqdm

def compute_rolling_nmse(gt, est):
    """
    Computes the Rolling (Cumulative) NMSE over time.
    Stabilizes the curve by averaging error from t=0 up to t.
    """
    # 1. Compute Squared Error per time step (Summing over nodes/dim 0)
    # Shape: (T,)
    sq_error_t = np.sum((gt - est)**2, axis=0)
    
    # 2. Compute Squared Energy of Ground Truth per time step
    # Shape: (T,)
    sq_energy_t = np.sum(gt**2, axis=0)
    
    # 3. Compute Cumulative Sums (Integration over time)
    cum_sq_error = np.cumsum(sq_error_t)
    cum_sq_energy = np.cumsum(sq_energy_t)
    
    # 4. Compute Ratio (adding epsilon to avoid div by zero at start)
    rolling_nmse = cum_sq_error / (cum_sq_energy + 1e-10)
    
    return rolling_nmse


def comp_nmse(gt, est):
    """
    Computes Instantaneous NMSE per time step.
    Matches MATLAB's: sum((gt - est).^2, 1)./sum(gt.^2,1)
    """
    num = np.sum((gt - est)**2, axis=0) 
    den = np.sum(gt**2, axis=0)
    return num / (den + 1e-10)

def plot_nmse(nmse_data, feature_name, t_step, start_idx, output_dir):
    fig = plt.figure(figsize=(10, 8))
    
    # Generate Time Axis matching the valid prediction window
    t_axis = np.arange(start_idx, start_idx + len(nmse_data))
    
    plt.plot(t_axis, nmse_data, 'b', linewidth=2, label=f'CC-VAR (Python) : {t_step} step')

    plt.ylim([0, 1.2]) 
    plt.xlabel('t', fontsize=20)
    plt.ylabel('NMSE', fontsize=20)
    plt.title(f'{feature_name} Prediction Error', fontsize=22)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=16, loc='upper right')
    plt.tick_params(axis='both', which='major', labelsize=16)

    if output_dir:
        filename = f"noaa_{feature_name.lower()}_Tstep{t_step}.png"
        save_path = output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {save_path}")

if __name__ == "__main__":
    
    # 1. Setup Paths
    dataset_name = "noaa_coastwatch_cellular"
    current_dir = Path.cwd() 
    root_name = (current_dir / ".." / ".." / "data" / "Input").resolve()
    output_dir = (current_dir / ".." / ".." / "data" / "Output" / dataset_name / "Figures").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load Data
    try:
        m = sio.loadmat(root_name / dataset_name / "data_oriented_mov.mat")
        topology = sio.loadmat(root_name / dataset_name / "adjacencies_oriented.mat")
    except FileNotFoundError:
        print("Error: Data files not found. Check paths.")
        exit()

    signal_node = m['l'].T.astype(float)
    signal_edge = m['s'].T.astype(float)
    signal_poly = m['u'].T.astype(float)

    # Center Data (Essential for NMSE to match)
    signal_node -= np.mean(signal_node)
    signal_edge -= np.mean(signal_edge)
    signal_poly -= np.mean(signal_poly)

    T_total = signal_edge.shape[1]

    # 4. Setup Complex & Parameters
    cellularComplex = {
        1: topology['B1'].astype(float),
        2: topology['B2'].astype(float)
    }

    algorithmParam = {
        'Tstep': 6,
        'P': 2,
        'K': [2, (2, 2), 2],     
        'mu': [0, (0, 0), 0],    
        'lambda': 0.01,
        'gamma': 0.98,
        'enabler': [True, True, True], 
        'FeatureNormalzn': True,
        'BiasEn': True
    }

    agent = CCVAR(algorithmParam, cellularComplex)

    # =========================================================
    # 5. WARM START (Matches MATLAB P+1 Start)
    # =========================================================
    P = algorithmParam['P']
    
    print("Warm-starting buffers...")
    for t_init in range(P):
        input_init = {
            0: signal_node[:, t_init],
            1: signal_edge[:, t_init],
            2: signal_poly[:, t_init]
        }
        
        for key in agent._data_keys:
            val = input_init[key]
            # Manually Shift and Insert
            old_data = agent._data[key][:, 1:]
            new_col = val.reshape(-1, 1)
            agent._data[key] = np.hstack([old_data, new_col])

            # Pre-warm normalization scale
            if algorithmParam['FeatureNormalzn']:
                varV = np.sum(val.flatten()**2)
                if agent._norm_scale[key] == 0:
                     agent._norm_scale[key] = np.sqrt(varV)
                else:
                     b = algorithmParam.get('b', 1)
                     agent._norm_scale[key] = (1 - b) * agent._norm_scale[key] + b * np.sqrt(varV)

    # =========================================================
    # 6. Online Learning Loop
    # =========================================================
    Tstep = algorithmParam['Tstep']
    
    pred_node = np.zeros_like(signal_node)
    pred_edge = np.zeros_like(signal_edge)
    pred_poly = np.zeros_like(signal_poly)

    
    
    print(f"Starting Online Loop. Total T: {T_total}")
    for t in tqdm(range(P, T_total), desc="CC-VAR Running"):
        
        input_data_t = {
            0: signal_node[:, t],
            1: signal_edge[:, t],
            2: signal_poly[:, t]
        }
       
        # 1. Update Weights
        agent.update(input_data_t)

        # 2. Forecast
        # Matches SCVAR_Forecast_modified logic
        predictions = agent.forecast(steps=Tstep)
        
        # 3. Store Prediction
        # -----------------------------------------------------------------
        # CRITICAL FIX: STORAGE INDEX
        # MATLAB stores the prediction for (t + Tstep) at index (t + 1).
        # We store at (t + 1) to align the plot visually with MATLAB.
        # -----------------------------------------------------------------
        store_idx = t + 1
        
        if store_idx < T_total:
            # predictions[k][:, -1] is the prediction for t + Tstep
            pred_node[:, store_idx] = predictions[0][:, -1]
            pred_edge[:, store_idx] = predictions[1][:, -1]
            pred_poly[:, store_idx] = predictions[2][:, -1]


    # =========================================================
    # 7. Calculate NMSE and Plot
    # =========================================================
    
    # MATLAB CompNMSE compares the vectors directly.
    # Since we aligned the storage, we can slice them identically.
    # Valid range: The predictions started filling at P + 1.
    valid_start = P
    # We stop at end-1 because MATLAB CompNMSE usually trims the last partial point
    # or the loop ends at T.

      # --- POLYGON PLOTTING ---
    if algorithmParam['enabler'][2]:
        gt = signal_poly[:, valid_start:]
        est = pred_poly[:, valid_start:]
        
        nmse_poly_vec = compute_rolling_nmse(gt, est)
        print(f"Average Polygon NMSE: {np.mean(nmse_poly_vec):.4f}")
        plot_nmse(nmse_poly_vec, "Node", Tstep, valid_start, output_dir)
    
    # --- EDGE PLOTTING ---
    if algorithmParam['enabler'][1]:
        gt = signal_edge[:, valid_start:]
        est = pred_edge[:, valid_start:]
        
        nmse_edge_vec = compute_rolling_nmse(gt, est)
        print(f"Average Edge NMSE: {np.mean(nmse_edge_vec):.4f}")
        plot_nmse(nmse_edge_vec, "Edge", Tstep, valid_start, output_dir)
    
    # --- NODE PLOTTING ---
    if algorithmParam['enabler'][0]:
        gt = signal_node[:, valid_start:]
        est = pred_node[:, valid_start:]
        
        nmse_node_vec = compute_rolling_nmse(gt, est)
        print(f"Average Node NMSE: {np.mean(nmse_node_vec):.4f}")
        plot_nmse(nmse_node_vec, "Node", Tstep, valid_start, output_dir)

    plt.show()