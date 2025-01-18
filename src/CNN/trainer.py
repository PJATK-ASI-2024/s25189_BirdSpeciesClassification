import os
import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime

class Trainer:
    def __init__(self, model, train_loader, test_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        self.model.to(self.config.device)

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(self.train_loader, desc="Training", leave=False):
            images, labels = images.to(self.config.device), labels.to(self.config.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        return epoch_loss, epoch_acc

    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Validating", leave=False):
                images, labels = images.to(self.config.device), labels.to(self.config.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def fit(self):
        best_acc = 0.0

        for epoch in range(self.config.num_epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate_epoch()

            print(f'Epoch [{epoch+1}/{self.config.num_epochs}] '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

            if val_acc > best_acc:
                best_acc = val_acc
                model_path = 'saved_models/best_model.pth'
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(self.model.state_dict(), model_path)
                print(f'Model saved as {model_path}')
        