#file to test whether python has permissions to save files within this directory
import numpy as np 
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)
xx = np.arange(0,100)
np.savetxt('/home/bay22/github_repos/RIMSAR/data/03_processed/xx.dat',xx)