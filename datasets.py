from torch import Tensor
from PIL import Image
import glob
import os

import torch

import input_target_transforms as TT
from visualize import visualize_outputs

# Expects directory of .png's as root
# Transforms should include ToTensor (also probably Normalize)
# Can apply different transform to output, returns image as input and label
class GameImagesDataset(torch.utils.data.Dataset):
    def __init__(self, root='/faim/datasets/test_icarus', train_or_val="train", transform=TT.ToTensor()):
        self.image_dir = os.path.join(root)
        # Get abs file paths
        self.image_list = glob.glob(f'{self.image_dir}/*.png')
        
        if train_or_val == "val":
            self.image_list = self.image_list[:int(len(self.image_list) * 0.2)]
        # self.image_folders = next(os.walk(self.image_dir))[1]
        self.length = len(self.image_list)
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        screenshot_file = self.image_list[idx]

        image = Image.open(screenshot_file).convert('RGB')
        target = image.copy()

        if self.transform:
            image, target = self.transform(image, target)

        sample = {'image': image, 'target': target}
        return sample

class OverfitDataset(torch.utils.data.Dataset):
    def __init__(self, root='./overfit.png', train_or_val="train", transform=TT.ToTensor(), num_images=2000):
        self.image_file = root
        self.length = num_images
        if train_or_val == "val":
            self.length = 1
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        image = Image.open(self.image_file).convert('RGB')

        target = image.copy()

        if self.transform:
            image, target = self.transform(image, target)

        sample = {'image': image, 'target': target}
        return sample

def get_transform(train):

    transforms = []
    transforms.append(TT.CenterCrop(224))
    if train:
        transforms.append(TT.RandomHorizontalFlip(0.5))
        transforms.append(TT.RandomVerticalFlip(0.5))
        # transforms.append(TT.RandomCrop(crop_size))
    transforms.append(TT.ToTensor())
    transforms.append(TT.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))

    return TT.Compose(transforms)

if __name__ == "__main__":
    print('test Game Image Dataset')

    trainset = GameImagesDataset(root='/faim/datasets/mario_images', train_or_val='train', transform=None)
    print(f'len trainset: {len(trainset)}')
    data = trainset[1]
    # data['image'].show()
    image = data['image']
    target = data['target']
    
    print(f'Image and Target with Transform = None')
    print(f'types: {type(image)}, {type(target)}')
    print(f'shapes: {(image.size)}, {(target.size)}')
    print(f'extrema: [{image.getextrema()}], [{target.getextrema()}]')

    do_transforms = get_transform(False)
    image, target = do_transforms(image, target)
    
    print(f'Image and Target with Transform = val')
    print(f'types: {type(image)}, {type(target)}')
    print(f'shapes: {(image.shape)}, {(target.shape)}')
    print(f'ranges: [{torch.min(image).item()} - {torch.max(image).item()}], [{torch.min(target).item()} - {torch.max(target).item()}]')

    image, target = image.unsqueeze(0), target.unsqueeze(0)
    
    print(f'Image and Target post Batching')
    print(f'shapes: {(image.shape)}, {(target.shape)}')


    image = image.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    print(f'Image and Target post detach, cpu, numpy for viz')
    print(f'types: {type(image)}, {type(target)}')
    print(f'shapes: {(image.shape)}, {(target.shape)}')
    print(f'ranges: [{image.min()} - {image.max()}], [{target.min()} - {target.max()}]')
    
    visualize_outputs(image, target, titles=['Image', 'Target'])
