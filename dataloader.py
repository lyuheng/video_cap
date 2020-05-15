import torch
import torch.utils.data as data
import numpy as np

class Dataset(data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        caption = self.labels[index]
        image = self.images[index]
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.labels)

class DatasetMultilabel(data.Dataset):
    def __init__(self, images, labels):
        lab = []
        self.label2image = []
        for video in range(len(labels)):
            for label in labels[video]:
                lab.append(label)        # label is a sentence
                self.label2image.append(video)
        self.labels = np.array(lab)
        self.images = images

    def __getitem__(self, index):
        caption = self.labels[index]
        image = self.images[self.label2image[index]]  # feature  # 一个caption对应一个image
        # Convert caption (string) to word ids.
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.labels)

#collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果

def collate_fn(data):
    """
    Creates mini-batch tensors from the list of tuples (image, caption).
    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (80, 4096).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images:  torch tensor of shape (batch_size, 80, 4096).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)       # caption最长的在最前面
    images, captions = zip(*data)

    # Merge images (from tuple of 2D tensor to 3D tensor).
    images = torch.Tensor(images)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap)-1 for cap in captions]  # 一个batch所有caption的长度（全减1）
    targets = torch.zeros(len(captions), max(lengths)).long() # 这个地方取所有长度中最长的一个
    inputs = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[1:end+1]     # 要输出第二个到最后一个
        inputs[i, :end] = cap[:end]         # 要输入第一个到倒数第二个
    return images, inputs, targets, lengths