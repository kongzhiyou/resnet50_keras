import os

root_path = '/Users/peter/data/jshl/train'
class_list = []

for subdir in sorted(os.listdir(root_path)):
    if os.path.isdir(os.path.join(root_path, subdir)):
        class_list.append(subdir)
