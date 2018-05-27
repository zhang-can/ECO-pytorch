# processing the raw data of the video datasets (something-something and jester)
# generate the meta files:
#   category.txt:               the list of categories.
#   train_videofolder.txt:      each row contains [videoname num_frames classIDX]
#   val_videofolder.txt:        same as above
#
# Created by: Can Zhang
# github: @zhang-can, May,27th 2018
#

import argparse
import os
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, choices=['something', 'jester'])
parser.add_argument('frame_path', type=str, help="root directory holding the frames")
parser.add_argument('--labels_path', type=str, default='data/dataset_labels/', help="root directory holding the csv files: labels, train & validation")
parser.add_argument('--out_list_path', type=str, default='data/')

args = parser.parse_args()

dataset = args.dataset
labels_path = args.labels_path
frame_path = args.frame_path

if dataset == 'something':
    dataset_name = 'something-something-v1'
elif dataset == 'jester':
    dataset_name = 'jester-v1'

print('\nProcessing dataset: {}\n'.format(dataset))

print('- Generating {}_category.txt ......'.format(dataset))
with open(os.path.join(labels_path, '{}-labels.csv'.format(dataset_name))) as f:
    lines = f.readlines()
categories = []
for line in lines:
    line = line.rstrip()
    categories.append(line)
categories = sorted(categories)
open(os.path.join(args.out_list_path, '{}_category.txt'.format(dataset)),'w').write('\n'.join(categories))
print('- Saved as:', os.path.join(args.out_list_path, '{}_category.txt!\n'.format(dataset)))

dict_categories = {}
for i, category in enumerate(categories):
    dict_categories[category] = i

files_input = ['{}-validation.csv'.format(dataset_name),'{}-train.csv'.format(dataset_name)]
files_output = ['{}_val.txt'.format(dataset),'{}_train.txt'.format(dataset)]
for (filename_input, filename_output) in zip(files_input, files_output):
    with open(os.path.join(labels_path, filename_input)) as f:
        lines = f.readlines()
    folders = []
    idx_categories = []
    for line in lines:
        line = line.rstrip()
        items = line.split(';')
        folders.append(items[0])
        idx_categories.append(os.path.join(str(dict_categories[items[1]])))
    output = []
    for i in range(len(folders)):
        curFolder = folders[i]
        curIDX = idx_categories[i]
        # counting the number of frames in each video folders
        dir_files = os.listdir(os.path.join(frame_path, curFolder))
        output.append('{} {} {}'.format(os.path.join(frame_path, curFolder), len(dir_files), curIDX))
        if i % 1000 == 0:
            print('- Generating {} ({}/{})'.format(filename_output, i, len(folders)))
    with open(os.path.join(args.out_list_path, filename_output),'w') as f:
        f.write('\n'.join(output))
    print('- Saved as:', os.path.join(args.out_list_path, '{}!\n'.format(filename_output)))
