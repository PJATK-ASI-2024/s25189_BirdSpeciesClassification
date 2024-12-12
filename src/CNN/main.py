from config import Config
from dataset import BirdDataset
from model import BirdCNNModel
from trainer import Trainer
from torch.utils.data import DataLoader

def main():
    config = Config()

    # Load datasets
    train_dataset = BirdDataset(config.train_img_dir, config.train_label_path, config.image_size, train=True)
    test_dataset = BirdDataset(config.test_img_dir, config.test_label_path, config.image_size, train=False)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    # Initialize model
    model = BirdCNNModel(num_classes=config.num_classes)

    # Initialize trainer
    trainer = Trainer(model, train_loader, test_loader, config)

    # Train model
    trainer.fit()

if __name__ == "__main__":
    main()