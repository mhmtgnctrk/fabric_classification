import os

# specify the root directory
root_dir = 'https://github.com/mhmtgnctrk/fabric_classification/blob/04837efcdfde5078d2c80a087390ba7ec0df2c8a/data/raw_data'

# use os.walk to iterate through subfolders
for root, dirs, files in os.walk(root_dir):
    # print the current directory
    print(f'Current directory: {root}')

    # print the subdirectories
    for d in dirs:
        print(f'Subdirectory: {d}')

    # print the files
    for f in files:
        print(f'File: {f}')