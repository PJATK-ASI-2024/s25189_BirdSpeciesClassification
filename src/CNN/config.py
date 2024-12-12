import torch

class Config:
    def __init__(self):
        self.train_img_dir = "data/train/split/train/images/raw"
        self.test_img_dir = "data/train/split/test/images"
        self.train_label_path = "data/train/split/train/train_labels.csv"
        self.test_label_path = "data/train/split/test/test_labels.csv"

        # Hyperparameters
        self.batch_size = 32
        self.num_workers = 4
        self.lr = 0.001
        self.num_epochs = 10
        self.image_size = (244,244)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_classes = 200 # CUB -> 200 labels