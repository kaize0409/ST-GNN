import numpy as np
import pandas as pd


watershed_matrix = np.load("data/watershed.npy")

watershed_data = pd.read_csv("data/watershed_avg.csv")


print(watershed_matrix.shape)
