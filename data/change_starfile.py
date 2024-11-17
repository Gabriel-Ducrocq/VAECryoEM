import argparse
import starfile

def change_mrcs_path(x, new_path):
	image_number, _ = x.split("@")
	return image_number + "@" + new_path

def change_starfile(starfile_path, new_image_path, new_starfile_path):
	df = starfile.read(starfile_path)
	df["particles"]["rlnImageName"] = df["particles"]["rlnImageName"].apply(lambda x: starfile_path(x, new_image_path))
	starfile.save(new_starfile_path, df)



if __name__ == '__main__':
    parser_arg = argparse.ArgumentParser()
    parser_arg.add_argument('--starfile', type=str, required=True)
    parser_arg.add_argument('--new_image_path', type=str, required=True)
    parser_arg.add_argument('--new_starfile_path', type=str, required=True)
    args = parser_arg.parse_args()
    starfile_path = args.starfile
    new_image_path = args.new_image_path
    new_starfile_path = args.new_starfile_path
    change_starfile(starfile_path, new_image_path, new_starfile_path)
