# test_trainer.py

import os
import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from trainer import Trainer

# A simple config-like object
class Config:
    def __init__(self):
        self.device = 'cpu'
        self.lr = 0.001
        self.num_epochs = 1  # For quick testing

# A dummy dataset to provide random images & labels
class DummyImageDataset(Dataset):
    def __init__(self, length=8):
        self.length = length
        # create random images, say 3 channels, 32x32
        # random labels in 0..1 (2 classes)
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # image shape (3, 32, 32)
        image = torch.randn(3, 32, 32)
        label = torch.randint(0, 2, size=(1,)).item()  # random label: 0 or 1
        return image, label

# A simple CNN for tests
class DummyCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(DummyCNN, self).__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x):
        x = self.conv(x)     # shape: [batch, 8, 32, 32]
        x = self.pool(x)     # shape: [batch, 8, 1, 1]
        x = x.view(x.size(0), -1)  # shape: [batch, 8]
        x = self.fc(x)       # shape: [batch, num_classes]
        return x

@pytest.fixture
def trainer_fixture(tmp_path):
    """
    Returns a Trainer object with a dummy CNN, DataLoaders, and config.
    Also sets up a temporary directory for saving.
    """
    model = DummyCNN(num_classes=2)
    config = Config()

    # create dataset & DataLoader
    train_dataset = DummyImageDataset(length=8)
    test_dataset = DummyImageDataset(length=4)
    train_loader = DataLoader(train_dataset, batch_size=2)
    test_loader = DataLoader(test_dataset, batch_size=2)

    # Override the saved_models path to tmp_path
    # so we don't clutter the real directory
    os.makedirs(tmp_path / "saved_models", exist_ok=True)
    # We'll monkey-patch the Trainer fit method to write to tmp_path
    # but simpler is to just do it once we create the trainer

    trainer = Trainer(model, train_loader, test_loader, config)
    return trainer, tmp_path

def test_trainer_init(trainer_fixture):
    """
    Basic test to confirm the Trainer object can initialize.
    """
    trainer, _ = trainer_fixture
    assert isinstance(trainer.model, nn.Module)
    assert trainer.config.num_epochs == 1
    assert hasattr(trainer, 'train_loader')
    assert hasattr(trainer, 'test_loader')

def test_train_epoch(trainer_fixture):
    """
    Test that train_epoch runs without error and returns (loss, acc).
    """
    trainer, _ = trainer_fixture
    loss, acc = trainer.train_epoch()
    assert isinstance(loss, float)
    assert isinstance(acc, float)
    # loss >= 0
    assert loss >= 0
    # acc between 0..1
    assert 0 <= acc <= 1

def test_validate_epoch(trainer_fixture):
    """
    Test that validate_epoch runs and returns (loss, acc).
    """
    trainer, _ = trainer_fixture
    loss, acc = trainer.validate_epoch()
    assert isinstance(loss, float)
    assert isinstance(acc, float)
    assert loss >= 0
    assert 0 <= acc <= 1

def test_fit_saves_model(trainer_fixture):
    """
    Test the fit() method runs a quick epoch and saves a model file.
    """
    trainer, tmp_path = trainer_fixture
    # We'll do 1 epoch, at end best_model.pth saved
    trainer.fit()
    # Check the file
    model_path = os.path.join("saved_models", "best_model.pth")
    # The code in trainer saves: 'saved_models/best_model.pth'
    # Confirm it exists
    assert os.path.isfile(model_path), "best_model.pth was not saved!"
