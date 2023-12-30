import os
import argparse
import numpy as np
from tqdm import tqdm
#import utils_data as utils


parser_arg = argparse.ArgumentParser()
parser_arg.add_argument('--folder_structures', type=str, required=True)
parser_arg.add_argument('--output_folder', type=str, required=True)
parser_arg.add_argument('--Apix', type=str, required=True)
parser_arg.add_argument('--Bsize', type=str, required=True)


args = parser_arg.parse_args()
folder_structures = args.folder_structures
output_folder= args.output_folder
Apix = args.Apix
box_size = args.Bsize

path_structures = [folder_structures + path for path in os.listdir(folder_structures) if ".pdb" in path]
print(path_structures)
indexes = [int(name.split("/")[-1].split(".")[0].split("_")[-1]) for name in path_structures]
print(f"Example indexes: indexes1 {indexes[0]}")
path_structures = list(zip(indexes, path_structures))


#parser = PDBParser(PERMISSIVE=0)
#centering_structure = parser.get_structure("A", path_structures[0])
N_volumes = len(path_structures)
#center_vector = utils.compute_center_of_mass(centering_structure)

for i in tqdm(range(N_volumes)):
	#os.system(f"pdb_selatom -CA,C,N {sorted1[i][1]} > backbone1.pdb")
	#os.system(f"pdb_selatom -CA,C,N {sorted2[i][1]} > backbone2.pdb")
	#os.system(f"pdb2mrc.py {backbone1} volume1 --apix={Apix}")
	#os.system(f"pdb2mrc.py {backbone2} volume2 --apix={Apix}")
	os.system(f" e2pdb2mrc.py {path_structures[i][1]} {output_folder}vol_{int(path_structures[i][0])}.mrc   --apix={Apix} --box={box_size} --center")