import os
import argparse
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

from torchvision import transforms
from facenet_pytorch import MTCNN


def parse_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--root',
        help = 'Path to the dataset',
        default = '../dataset/CASIA_maxpy_clean',
        type = str
    )

    parser.add_argument(
        '--annot_path',
        help = 'Path to the annotation file',
        default = '../dataset/CASIA_clean_list.txt',
        type = str
    )

    parser.add_argument(
        '--save_dir',
        help = 'Path to the save directory',
        default = '../dataset/CASIA_aligned',
        type = str
    )

    parser.add_argument(
        '--batch_size',
        help = 'Batch size for MTCNN model',
        default = 1024,
        type = int
    )

    parser.add_argument(
        '--device',
        help = 'Device used for loading the model',
        default = 'cuda',
        type = str
    )
    
    return parser.parse_args()


##############################################################################
# Below methods are copied from: https://github.com/MuggleWang/CosFace_pytorch
##############################################################################

def default_reader(fileList):
    imgList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            imgPath, label = line.strip().split(' ')
            imgList.append((imgPath, int(label)))
    return imgList


def pairs_reader(fileList, to_int = True):
    imgList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            data = line.strip().split()
            
            if len(data) == 3:
                imgPath1 = imgPath2 = data[0]
                label1, label2 = data[1], data[2]
            elif len(data) == 4:
                imgPath1 = data[0]
                imgPath2 = data[2]
                label1, label2 = data[1], data[3]
            
            if to_int:
                label1, label2 = int(label1), int(label2)
                
            imgList.append((imgPath1, label1))
            imgList.append((imgPath2, label2))

    return imgList

##############################################################################

def align_face(model, root, img_list, save_dir, batch_size = 128, pairs = False):
    if pairs:
        img_data = pairs_reader(img_list, to_int = False)
    else:
        img_data = default_reader(img_list)

    idx = 0
    while idx * batch_size < len(img_data):
        idx_start = idx * batch_size
        idx_end = min([len(img_data), (idx+1) * batch_size])

        # get filenames
        if pairs:
            batch_data = [f'{name}_' + '0'*(4-len(fileid)) + fileid + '.jpg' for name, fileid in img_data[idx_start:idx_end]]
        else:
            batch_data = [i for i, _ in img_data[idx_start:idx_end]]
    
        # Resize the image into 224, 224
        resize_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            transforms.ToPILImage()
        ])
        batch_img = [resize_transform(Image.open(os.path.join(root, img_path)).convert('RGB')) for img_path in batch_data]

        save_path = [os.path.join(save_dir, img_path) for img_path in batch_data]
        model(batch_img, save_path = save_path)

        idx += 1


if __name__ == '__main__':
    args = parse_argument()

    if 'casia' in args.root.lower() or 'ar_face' in args.root.lower():
        pairs = False
    elif 'lfw' in args.root.lower():
        pairs = True

    mtcnn = MTCNN(
        image_size=224, 
        margin = 20, 
        device=args.device, 
        selection_method='center_weighted_size'
    )

    align_face(
        mtcnn,
        args.root,
        args.annot_path,
        args.save_dir,
        args.batch_size,
        pairs = pairs
    )