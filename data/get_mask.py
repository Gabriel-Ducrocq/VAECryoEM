import sys
import os
path = os.path.abspath("model")
sys.path.append(path)
import mrc
import yaml
import torch
import utils
import mrcfile
import argparse
import starfile
import numpy as np
from ctf import CTF
import seaborn as sns
from time import time
from tqdm import tqdm
import Bio.PDB as bpdb
from Bio.PDB import PDBIO
from polymer import Polymer
from Bio.PDB import PDBParser
import matplotlib.pyplot as plt
from dataset import ImageDataSet
from gmm import Gaussian, EMAN2Grid
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from scipy.spatial.distance import cdist
from pytorch3d.transforms import quaternion_to_axis_angle, quaternion_to_matrix






cols = ["red", "blue", "orange", "green", "pink", "yellow", "black"]*10
parser_arg = argparse.ArgumentParser()
parser_arg.add_argument('--model_path', type=str, required=True)
args = parser_arg.parse_args()
model_path = args.model_path
vae = torch.load(model_path, map_location='cpu')
vae.device="cpu"
segments = vae.sample_mask(1)
hard_segments = np.argmax(segments.detach().cpu().numpy(), axis=-1)
all_segments = []
for l in range(vae.N_domains):
	all_segments.append(np.sum(hard_segments[0] == l))


A= np.cumsum(all_segments)
ll= np.zeros(vae.N_domains+1)
ll[1:] = A
list_coloring = []
for i in range(len(ll)-1):
    print(f"color #1:{ll[i]}-{ll[i+1]} {cols[i]}")  


