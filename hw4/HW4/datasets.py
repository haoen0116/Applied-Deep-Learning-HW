import numpy as np
import torch
from PIL import Image


class Anime:
    """ Dataset that loads images and image tags from given folders.

    Attributes:
        root_dir: folder containing training images
        tags_file: a dictionary object that contains class tags of images.
        transform: torch.Transform() object to perform image transformations.
        img_files: a list of image file names in root_dir
        dataset_len: number of training images.
    """

    def __init__(self, root_dir, transform):
        self.file_names = []
        self.root_dir = root_dir
        self.transform = transform

        self.hair_tag = []
        self.eye_tag = []
        self.face_tag = []
        self.glasses_tag = []

        with open(self.root_dir + 'cartoon_attr.txt') as f:
            lines = f.readlines()
            for line in lines[2:]:
                split_data = line.strip().split()
                self.file_names.append(split_data[0])
                label = split_data[1:]
                self.hair_tag.append(list(map(int, label[:6])))
                self.eye_tag.append(list(map(int, label[6:10])))
                self.face_tag.append(list(map(int, label[10:13])))
                self.glasses_tag.append(list(map(int, label[13:])))

        self.num_samples = len(self.file_names)

    def length(self):
        return self.num_samples

    def get_item(self, idx):
        """ Return '[idx].jpg' and its tags. """

        # img = mpimg.imread(self.root_dir + 'images/' + self.file_names[idx])
        # img = img[:, :, (2, 1, 0)]
        
        img = Image.open(self.root_dir + 'images/' + self.file_names[idx])
        # img = img.resize((64, 64))
        if self.transform:
            img = self.transform(img)

        return img, \
               torch.tensor(self.hair_tag[idx]), \
               torch.tensor(self.eye_tag[idx]), \
               torch.tensor(self.face_tag[idx]), \
               torch.tensor(self.glasses_tag[idx])


class Shuffler:
    """ Class that supports andom sampling of training data.

    Attributes:
        dataset: an Anime dataset object.
        batch_size: size of each random sample.
        dataset_len: size of dataset.
    
    """

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.dataset_len = self.dataset.length()

    def get_batch(self):
        """ Returns a batch of randomly sampled images and its tags. 

        Args:
            None.

        Returns:
            Tuple of tensors: img_batch, hair_tags, eye_tags
            img_batch: tensor of shape N * 3 * 64 * 64
            hair_tags: tensor of shape N * hair_classes
            eye_tags: tensor of shape N * eye_classes
        """
        indices = np.random.choice(self.dataset_len, self.batch_size)  # Sample non-repeated indices
        img_batch, hair_tags, eye_tags, face_tags, glasses_tags = [], [], [], [], []
        for i in indices:
            img, hair_tag, eye_tag, face_tag, glasses_tag = self.dataset.get_item(i)
            img_batch.append(img.unsqueeze(0))
            hair_tags.append(hair_tag.unsqueeze(0))
            eye_tags.append(eye_tag.unsqueeze(0))
            face_tags.append(face_tag.unsqueeze(0))
            glasses_tags.append(glasses_tag.unsqueeze(0))
        img_batch = torch.cat(img_batch, 0)
        hair_tags = torch.cat(hair_tags, 0)
        eye_tags = torch.cat(eye_tags, 0)
        face_tags = torch.cat(face_tags, 0)
        glasses_tags = torch.cat(glasses_tags, 0)

        return img_batch, hair_tags, eye_tags, face_tags, glasses_tags
    