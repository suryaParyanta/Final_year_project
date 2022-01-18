import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import transforms


def PIL_loader(file_path:str):
    """
    PIL Image Loader. Read image from specified path.

    :param file_path: Path to the image file
    
    :returns: RGB Image
    :rtype:   PIL.Image
    """
    return Image.open(file_path).convert('RGB')


def filelist_reader(file_path:str):
    """
    Read the content (file list) from specified txt file. The file list format are as follows:
       class_folder1/img1.jpg
       class_folder1/img2.jpg
       class_folder2/img5.jpg
       class_folder3/img4.jpg
       ...
    
    :param file_path: Path to the txt file

    :returns: List of file directories
    :rtype:   list
    """
    image_list = []
    with open(file_path, 'r') as file:
        for data in file:
            img_path, _ = data.split()
            label, _ = img_path.split('/')
            image_list.append((img_path, int(label)))

    return image_list


def annotation_reader(file_path:str):
    """
    Read the content (annotations) from specified txt file. The annotation format are as follows:
       path_to_image  class0
       path_to_image  class1
       path_to_image  class2
       ...
    
    :param file_path: Path to the txt file
    
    :returns: List of file directories
    :rtype:   list
    """
    image_list = []
    with open(file_path, 'r') as file:
        for data in file:
            img_path, label = data.split()
            image_list.append((img_path, int(label)))

    return image_list


class ImageDataset(Dataset):
    """
    Image dataset class implementation. This class will be used together with Pytorch DataLoader. 
    Implements __init__, __getitem__, and __len___ methods.
    """

    def __init__(self, root:str, data_dir:str, annot_path:str, transform = None, reader = filelist_reader, img_loader = PIL_loader):
        """
        ImageDataset initialization.

        :param root:       Root path of the dataset
        :param data_dir:   Images directory from root
        :param annot_path: Annotation file path from root
        :param transform:  Image preprocessing (transformation) method
        :param reader:     Txt file reader method (filelist_reader or annotation_reader)
        :param img_loader: Image loader method
        """
        super(ImageDataset, self).__init__()

        self.image_annots = reader(os.path.join(root, annot_path))
        self.data_dir = os.path.join(root, data_dir)
        self.img_loader = img_loader
        self.transform = transform
    
    def __getitem__(self, index:int):
        """
        Get image and ground-truth label based on index.

        :param index: Index number

        :returns: Image and ground-truth label
        :rtype:   tuple
        """
        img_path, label = self.image_annots[index]
        img = self.img_loader(os.path.join(self.data_dir, img_path))

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        """
        Get the dataset size.

        :returns: Dataset size
        :rtype:   int
        """
        return len(self.image_annots)


class DatasetSampler(Sampler):
    """
    Dataset sampler based on specified conditions (mask).
    """

    def __init__(self, mask:list, data_source):
        """
        DatasetSampler initialization.

        :param mask:        Array of boolean specifying the conditions to sample
        :param data_source: Pytorch Dataset class
        """
        super(DatasetSampler, self).__init__(data_source)

        self.mask = mask
        self.data_source = data_source

    def __iter__(self):
        """
        Get class iterator.
        """
        return iter([i.item() for i in torch.nonzero(self.mask)])

    def __len__(self):
        """
        Get dataset size.

        :returns: Dataset size
        :rtype:   int
        """
        return len(self.data_source)


def get_dataset(root:str, 
                train_data_dir:str, train_annot_path:str, transform_train = None, 
                test_data_dir:str = None, test_annot_path:str = None, transform_test = None, 
                mode:str = 'list'):
    """
    Get training and test (if specified) dataset.

    :param root: Dataset root directory
    :param train_data_dir:   Training images directory from root
    :param train_annot_path: Training list/annotation file path from root
    :param transform_train:  Image preprocessing for training
    :param test_data_dir:    Test images directory from root
    :param test_annot_path:  Test list/annotation file path from root
    :param transform_test:   Image preprocessing for testing
    :param mode:             Txt file reader mode (list, annotation)

    :returns: Training and test dataset
    :rtype:   tuple
    """
    assert mode in ['list', 'annotation']
    
    if mode == 'list':
        reader = filelist_reader
    elif mode == 'annotation':
        reader = annotation_reader

    dataset_train = ImageDataset(root, train_data_dir, train_annot_path, transform = transform_train, reader=reader)
    
    dataset_test = None
    if test_data_dir is not None and test_annot_path is not None:
        dataset_test = ImageDataset(root, test_data_dir, test_annot_path, transform = transform_test, reader=reader)
    
    return dataset_train, dataset_test


def get_data_loader(dataset, batch_size:int = 128, num_workers:int = 0, shuffle:bool = True, sampler = None):
    """
    Get Pytorch DataLoader from Dataset class.

    :param dataset:     Pytorch Dataset class
    :param batch_size:  Number of images to feed into NN model for one forward + backward pass
    :param num_workers: How many workers to load the data; typically, num_workers = 4 × num_gpus
    :param shuffle:     Whether to shuffle the dataset
    :param sampler:     Dataset sampler
    """
    loader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        pin_memory = True,
        num_workers = num_workers,
        sampler = sampler
    )

    return loader


def get_dataset_from_name(name):
    supp_train_data = ["", "mnist_train", "casia"]
    supp_test_data = ["", "mnist_test", "lfw", "lfw_masked"]
    black_occ = ["mnist_test_occ_black_1", "mnist_test_occ_black_2", "mnist_test_occ_black_3"]
    white_occ = ["mnist_test_occ_white_1", "mnist_test_occ_white_2", "mnist_test_occ_white_3"]
    gauss_occ = ["mnist_test_occ_gauss_1", "mnist_test_occ_gauss_2", "mnist_test_occ_gauss_3"]
    natural_occ = ["mnist_test_occ_natural_1", "mnist_test_occ_natural_2", "mnist_test_occ_natural_3"]

    assert name in supp_train_data + supp_test_data + black_occ + white_occ + gauss_occ + natural_occ, "Unsupported dataset name"

    # define transformation
    if "mnist" in name:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
    elif name == "casia":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
        ])

    # get pytorch dataset
    if name == "mnist_train":
        root = "dataset/MNIST_224X224_3"
        annot = "pairs_train.txt"
        data = "train"

        dataset, _ = get_dataset(
            root,
            data,
            annot,
            transform
        )

    elif name == "casia":
        root = "dataset"
        annot = "CASIA_aligned_list.txt"
        data = "CASIA_aligned"

        dataset, _ = get_dataset(
            root,
            data,
            annot,
            transform,
            mode = 'annotation'
        )

    elif name == "mnist_test":
        root = "dataset/MNIST_224X224_3"
        annot = "pairs_test.txt"
        data = "test"

        dataset, _ = get_dataset(
            root,
            data,
            annot,
            transform
        )

    elif "mnist_test_occ_black" in name :
        level = name.split("_")[-1]
        root = "dataset/MNIST_224X224_3"
        annot = "pairs_test.txt"
        data = f"test_occ_black_{level}"

        dataset, _ = get_dataset(
            root,
            data,
            annot,
            transform
        )

    elif "mnist_test_occ_white" in name:
        level = name.split("_")[-1]
        root = "dataset/MNIST_224X224_3"
        annot = "pairs_test.txt"
        data = f"test_occ_white_{level}"

        dataset, _ = get_dataset(
            root,
            data,
            annot,
            transform
        )

    elif "mnist_test_occ_gauss" in name:
        level = name.split("_")[-1]
        root = "dataset/MNIST_224X224_3"
        annot = "pairs_test.txt"
        data = f"test_occ_gauss_{level}"

        dataset, _ = get_dataset(
            root,
            data,
            annot,
            transform
        )

    elif "mnist_test_occ_natural" in name:
        level = name.split("_")[-1]
        root = "dataset/MNIST_224X224_3"
        annot = "pairs_test.txt"
        data = f"test_occ_natural_{level}"

        dataset, _ = get_dataset(
            root,
            data,
            annot,
            transform
        )
    
    elif name == "lfw_masked":
        root = "dataset/LFW_pairs_aligned"
        annot = "pairs_masked.txt"
        data = "Combined"
        dataset = None
    
    elif name == "lfw":
        root = "dataset/LFW_pairs_aligned"
        annot = "pairs.txt"
        data = "Images"
        dataset = None

    else:
        root = "Not found"
        annot = "Not found"
        data = "Not found"
        dataset = None
    
    return dataset, (root, annot, data)


if __name__ == '__main__':
    root = '../dataset/MNIST_224X224_3'
    
    # Train dataset
    train_annot = 'pairs_train.txt'
    train_data = 'train'

    # Test dataset
    test_annot = 'pairs_test.txt'
    test_data = 'test'

    train_mnist = ImageDataset(root, train_data, train_annot)

    pil_im, label = train_mnist[1]
    pil_im.show()
    print(label)