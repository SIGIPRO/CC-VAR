from pathlib import Path

import numpy as np
from scipy.io import loadmat
from functions.core.ccvar import CCVAR



if __name__ == "__main__":

    dataset_name = "noaa_coastwatch_cellular"
    current_dir = Path.cwd()
    root_name = (current_dir / ".." / ".." / "data" / "Input").resolve()
    output_dir = (current_dir / ".." / ".." / "data" / "Output" / dataset_name).resolve()

    data_path = root_name / dataset_name / "data_oriented_mov.mat"
    adjacency_path = root_name / dataset_name / "adjacencies_oriented.mat"

    m = loadmat(data_path)
    topology = loadmat(adjacency_path)

    algorithmParam = dict() #No parameter overloading is needed for this experiment
    cellularComplex = {0:np.zeros(), 1:topology['B1'], 2:topology['B2']}

    print(type(topology['B1']))
    print(m)

    agent = CCVAR(algorithmParam=algorithmParam,
                  cellularComplex=cellularComplex)
    

    



 

    

