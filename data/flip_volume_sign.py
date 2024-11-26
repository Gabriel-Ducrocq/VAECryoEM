import os
import mrcfile
import argparse

parser_arg = argparse.ArgumentParser()
parser_arg.add_argument('--folder_path', type=str, required=True)
args = parser_arg.parse_args()
folder_path = args.folder_path
paths = [folder_path + path for path in os.listdir(folder_path) if ".mrc" in path]


for path in paths:
	volume = mrcfile.read(path)
	volume_inverted = -1*volume
	prefix, _ = path.split(".")
	mrcfile.write(prefix + "_inverted.mrc", volume_inverted)

