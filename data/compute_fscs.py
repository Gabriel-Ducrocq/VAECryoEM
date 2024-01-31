import os
from tqdm import tqdm
import argparse

parser_arg = argparse.ArgumentParser()
parser_arg.add_argument('--folder1', type=str, required=True)
parser_arg.add_argument('--folder2', type=str, required=True)
parser_arg.add_argument('--output', type=str, required=True)
parser_arg.add_argument('--Apix', type=str, required=True)

args = parser_arg.parse_args()
folder_1 = args.folder1
folder_2 = args.folder2
output = args.output
Apix = args.Apix

volumes1 = [folder_1 + "volume_6zp5.mrc", folder_1 + "volume_6zp7.mrc"]
#volumes1 = [folder_1 + path for path in os.listdir(folder_1) if ".mrc" in path]
volumes2 = [folder_2 + path for path in os.listdir(folder_2) if ".mrc" in path]
#indexes1 = [int(name.split("/")[-1].split(".")[0].split("_")[-1]) for name in volumes1]
indexes2 = [int(name.split("/")[-1].split(".")[0].split("_")[-1]) for name in volumes2]
#print(f"Example indexes: indexes1 {indexes1[0]}, indexes2 {indexes2[0]}")
print(f"Example indexes: indexes2 {indexes2[0]}")

#sorted1 = sorted(zip(indexes1, volumes1))
sorted2 = sorted(zip(indexes2, volumes2))

#print(sorted(indexes1))
print("\n\n\n\n")
print(sorted(indexes2))
N_volumes = len(sorted2)

for i in tqdm(range(0, N_volumes,10)):
	#os.system(f"pdb_selatom -CA,C,N {sorted1[i][1]} > backbone1.pdb")
	#os.system(f"pdb_selatom -CA,C,N {sorted2[i][1]} > backbone2.pdb")
	#os.system(f"pdb2mrc.py {backbone1} volume1 --apix={Apix}")
	#os.system(f"pdb2mrc.py {backbone2} volume2 --apix={Apix}")
	#os.system(f" e2proc3d.py {sorted1[i][1]} {output}/fsc_{i}.txt --calcfsc={sorted2[i][1]}  --apix={Apix}")
	if i <= 7000:
		os.system(f" e2proc3d.py {volumes1[0]} {output}/fsc_{i}.txt --calcfsc={sorted2[i][1]}  --apix={Apix}")
	else:
		os.system(f" e2proc3d.py {volumes1[1]} {output}/fsc_{i}.txt --calcfsc={sorted2[i][1]}  --apix={Apix}")


