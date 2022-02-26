import os
import argparse
from collections import defaultdict


def parse_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--root',
        help = 'Path to the dataset',
        default = '../../dataset/CASIA_aligned',
        type = str
    )

    parser.add_argument(
        '--save_dir',
        help = 'Save directory',
        default = '../../dataset/CASIA_aligned_list.txt'
    )

    return parser.parse_args()


def generate_list(root, save_dir, dataset_name = 'casia'):
    """
    Generate list file along with class id. If the dataset_name is casia,
    then the dataset must follow CASIA_Webface format, e.g. in the form of:
        root/class_id_0/img1.jpg
        root/class_id_0/img2.jpg
        root/class_id_1/img1.jpg
        ...
    
    If the mode is ar_face, then the dataset must follow this format:
        root/m-class_id-1.jpg
        root/m-class_id-2.jpg
        root/w-class_id-1.jpg
        ...
    (m and w represent images of man and women)
    """
    assert dataset_name in ['casia', 'ar_face']

    id_to_class = defaultdict(lambda: -1)
    class_count = 0
    num_images = 0

    if dataset_name == 'casia':
        for dir in os.listdir(root):
            # encode new identity to dictionary
            if id_to_class[dir] == -1:
                id_to_class[dir] = class_count
                class_count += 1
            
            num_images += len(os.listdir(os.path.join(root, dir)))

            for filename in os.listdir(os.path.join(root, dir)):
                img_path = os.path.join(dir, filename)
                
                with open(save_dir, 'a') as txt_file:
                    txt_file.write(f'{img_path} {id_to_class[dir]}\n')

    elif dataset_name == 'ar_face':
        for img_path in os.listdir(root):
            id_name = img_path[:5]
            num_images += 1
            
            if id_to_class[id_name] == -1:
                id_to_class[id_name] = class_count
                class_count += 1
            
            with open(save_dir, 'a') as txt_file:
                txt_file.write(f'{img_path} {id_to_class[id_name]}\n')
    

if __name__ == '__main__':
    args = parse_argument()

    if 'casia' in args.root.lower():
        dataset_name = 'casia'
    elif 'ar_face' in args.root.lower():
        dataset_name = 'ar_face'

    generate_list(args.root, args.save_dir, dataset_name)