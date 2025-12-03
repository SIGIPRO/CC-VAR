from pathlib import Path
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from functions.core.ccvar import CCVAR

# Helper to compute NMSE (Instantaneous per time step)
def comp_nmse(gt, est):
    # gt, est shape: (N, T)
    # Returns shape (T,)
    num = np.sum((gt - est)**2, axis=0) 
    den = np.sum(gt**2, axis=0)
    return num / (den + 1e-10)

def plot_nmse(nmse_data, feature_name, t_step, start_idx, output_dir):
    """
    Plots the NMSE over time, mimicking the MATLAB plotter style.
    """
    # Create figure (size approx similar to MATLAB's position)
    fig = plt.figure(figsize=(10, 8))
    
    # Generate Time Axis
    t_axis = np.arange(start_idx, start_idx + len(nmse_data))
    
    # Plot CC-VAR (Black solid line, matching 'k' in MATLAB)
    plt.plot(t_axis, nmse_data, 'k', linewidth=3, label=f'CC-VAR : {t_step} step')

    # Styling
    plt.ylim([0, 1.5]) # Slightly higher than 1 to see spikes, or set to [0,1] as in MATLAB
    plt.xlabel('t', fontsize=20)
    plt.ylabel('NMSE', fontsize=20)
    plt.title(f'{feature_name} Prediction Error', fontsize=22)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=16, loc='upper right')
    
    # Tick params
    plt.tick_params(axis='both', which='major', labelsize=16)

    # Save
    if output_dir:
        filename = f"noaa_{feature_name.lower()}_Tstep{t_step}.png"
        save_path = output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {save_path}")
    
    # Close to free memory
    plt.close(fig)

if __name__ == "__main__":
    
    # 1. Setup Paths
    dataset_name = "noaa_coastwatch_cellular"
    current_dir = Path.cwd() 
    root_name = (current_dir / ".." / ".." / "data" / "Input").resolve()
    
    # Output path for figures
    output_dir = (current_dir / ".." / ".." / "data" / "Output" / dataset_name / "Figures").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {root_name}")

    # 2. Load Data
    try:
        m = sio.loadmat(root_name / dataset_name / "data_oriented_mov.mat")
        topology = sio.loadmat(root_name / dataset_name / "adjacencies_oriented.mat")
    except FileNotFoundError:
        print("Error: Data files not found.")
        exit()

    # 3. Extract Signals & Center them
    signal_node = m['l'].T.astype(float)
    signal_edge = m['s'].T.astype(float)
    signal_poly = m['u'].T.astype(float)

    # Center Data
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
        'Tstep': 1,
        'P': 2,
        'K': [2, (2, 2), 2],     # K0, (K1_lower, K1_upper), K2
        'mu': [0, (0, 0), 0],    
        'lambda': 0.01,
        'gamma': 0.98,
        'enabler': [True, True, True], 
        'FeatureNormalzn': True,
        'BiasEn': True
    }

    # 5. Initialize Model
    agent = CCVAR(algorithmParam, cellularComplex)

    # 6. Online Learning Loop
    P = algorithmParam['P']
    Tstep = algorithmParam['Tstep']
    
    pred_node = np.zeros_like(signal_node)
    pred_edge = np.zeros_like(signal_edge)
    pred_poly = np.zeros_like(signal_poly)

    print(f"Starting Online Loop. Total T: {T_total}")

    for t in range(P, T_total - Tstep):
        
        input_data_t = {
            0: signal_node[:, t],
            1: signal_edge[:, t],
            2: signal_poly[:, t]
        }

        # Update
        agent.update(input_data_t)

        # Forecast
        predictions = agent.forecast(steps=Tstep)
        
        # Store
        target_idx = t + Tstep
        if target_idx < T_total:
            pred_node[:, target_idx] = predictions[0][:, -1]
            pred_edge[:, target_idx] = predictions[1][:, -1]
            pred_poly[:, target_idx] = predictions[2][:, -1]

        if t % 100 == 0:
            print(f"Step {t}/{T_total}")

    # 7. Calculate NMSE and Plot
    valid_idx = P + Tstep
    
    # --- EDGE PLOTTING ---
    if algorithmParam['enabler'][1]:
        # Compute NMSE vector (over time)
        nmse_edge_vec = comp_nmse(signal_edge[:, valid_idx:], pred_edge[:, valid_idx:])
        mean_nmse_edge = np.mean(nmse_edge_vec)
        print(f"Average Edge NMSE: {mean_nmse_edge:.4f}")
        
        # Plot
        plot_nmse(nmse_edge_vec, "Edge", Tstep, valid_idx, output_dir)
    
    # --- NODE PLOTTING ---
    if algorithmParam['enabler'][0]:
        nmse_node_vec = comp_nmse(signal_node[:, valid_idx:], pred_node[:, valid_idx:])
        mean_nmse_node = np.mean(nmse_node_vec)
        print(f"Average Node NMSE: {mean_nmse_node:.4f}")
        
        # Plot
        plot_nmse(nmse_node_vec, "Node", Tstep, valid_idx, output_dir)

    # --- POLYGON PLOTTING ---
    if algorithmParam['enabler'][2]:
        nmse_poly_vec = comp_nmse(signal_poly[:, valid_idx:], pred_poly[:, valid_idx:])
        mean_nmse_poly = np.mean(nmse_poly_vec)
        print(f"Average Poly NMSE: {mean_nmse_poly:.4f}")
        
        # Plot
        plot_nmse(nmse_poly_vec, "Polygon", Tstep, valid_idx, output_dir)

        plt.show()