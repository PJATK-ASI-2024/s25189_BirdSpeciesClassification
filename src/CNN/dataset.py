import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class BirdDataset(Dataset):
    def __init__(self, image_dir, label_file, image_size=(224,224), train=True):
        self.img_dir = image_dir
        self.df = pd.read_csv(label_file)
        self.train = train

        # random image transformation

        if train: 
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image_path']
        label = self.df.iloc[idx]['label'] - 1  # Ensure classes start at 0
        full_path = os.path.join(self.img_dir, img_path)
        image = Image.open(full_path).convert('RGB')
        image = self.transform(image)
        return image, label

        