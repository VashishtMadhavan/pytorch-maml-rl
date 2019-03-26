import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, help='txt file with paths')
args = parser.parse_args()

if not args.file:
	raise NotImplementedError("Please list file")

HDFS_DIR_PREFIX='file://'
HDFS_DIR='/ailabs/vash/grid_game/'
HDFS_LOCAL_DIR='/Users/vashishtmadhavan/Documents/Code/gen_proj_data/'

hdfs_paths = [x.rstrip() for x in open(args.file).readlines()]
trunc_paths = [l.replace(HDFS_DIR, '') for l in hdfs_paths]
local_paths = [HDFS_DIR_PREFIX + HDFS_LOCAL_DIR + t for t in trunc_paths]

base_paths = [HDFS_LOCAL_DIR + t for t in trunc_paths]
#for b in base_paths:
#	os.makedirs(b, exist_ok=True)

for h,l in zip(hdfs_paths, local_paths):
	print("Copying {}...".format(h))
	os.system('opus -da wbu2/da1 hdfs cp {} {}'.format(h, l))
