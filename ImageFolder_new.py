import torch.utils.data as data

from PIL import Image

import os
import os.path
import sys
import pdb
import numpy as np



def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir, class_to_idx, extensions,num_instance_per_class):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            if num_instance_per_class==0:
                num = len(fnames)
            else:
                num = min(num_instance_per_class,len(fnames))
            for fname in sorted(fnames)[:num]:
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class DatasetFolder(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None,
                loader=default_loader, index=None,num_instance_per_class=0,class_mean_old=None,class_index_old=None,repeat=False):

        super(DatasetFolder, self).__init__()


        classes, class_to_idx = self._find_classes(root)
        extensions=IMG_EXTENSIONS
        np.random.seed(0)
        list_permutaion = np.random.permutation( len(class_to_idx.items()))
        class_to_idx = {k: list_permutaion[v] for k, v in class_to_idx.items() if list_permutaion[v] in index}
        samples = make_dataset(root, class_to_idx, extensions,num_instance_per_class)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root + "\n" "Supported extensions are: " + ",".join(
                extensions)))

        self.root = root
        self.loader = default_loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = target_transform


        self.class_mean_old=class_mean_old
        self.label_index_old=class_index_old
        self.repeat=repeat


    def _find_classes(self, dir):

        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            sample = self.target_transform(sample)
        if self.repeat==True:
            rand_idx=np.random.randint(0, len(self.label_index_old))
            class_mean_old_idx=self.class_mean_old[rand_idx]
            class_label_old_idx=self.label_index_old[rand_idx]
            return sample, target,class_mean_old_idx,class_label_old_idx
        else:
            return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class  DatasetFolder_feature(data.Dataset):
    def __init__(self, feature_current,label_current,train_embeddings_cl_old, train_labels_cl_old,class_mean_old=None,class_index_old=None,repeat=False):
        super(DatasetFolder_feature, self).__init__()
        self.feature_current=feature_current
        self.label_current=label_current
        self.feature_old=train_embeddings_cl_old
        self.label_old=train_labels_cl_old
        self.repeat=repeat
        self.class_mean_old=class_mean_old
        self.label_index_old=class_index_old

    def __getitem__(self, index):
        feature = self.feature_current[index]
        label=self.label_current[index]
        feature_old = self.feature_old[index]
        label_old=self.label_old[index]
        if self.repeat==False:
            return feature, label,feature_old,label_old
        else:
            rand_idx=np.random.randint(0, len(self.label_index_old))
            class_mean_old_idx=self.class_mean_old[rand_idx]
            class_label_old_idx=self.label_index_old[rand_idx]
            return feature, label,feature_old,label_old,class_mean_old_idx,class_label_old_idx
    def __len__(self):
        return len(self.feature_current)


class DatasetFolder_feature_val(data.Dataset):
    def __init__(self, feature_current, label_current):
        super(DatasetFolder_feature_val, self).__init__()
        self.feature_current = feature_current
        self.label_current = label_current
    def __getitem__(self, index):
        feature = self.feature_current[index]
        label = self.label_current[index]
        return feature, label

    def __len__(self):
        return len(self.feature_current)


