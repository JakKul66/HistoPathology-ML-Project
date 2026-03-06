import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
import os

def setup_environment():
    pl.seed_everything(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(f"System gotowy do pracy na: {torch.cuda.get_device_name(0)}")
    else:
        print("UWAGA: System pracuje na procesorze")
    
    BATCH_SIZE = 32
    IMG_SIZE = 256
    return device, BATCH_SIZE, IMG_SIZE

class CancerDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, img_size):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):
        train_dir = os.path.join(self.data_dir, "train")
        val_dir = os.path.join(self.data_dir, "val")
        test_dir = os.path.join(self.data_dir, "test")

        self.train_ds = torchvision.datasets.ImageFolder(train_dir, transform=self.transform)
        self.val_ds = torchvision.datasets.ImageFolder(val_dir, transform=self.transform)
        self.test_ds = torchvision.datasets.ImageFolder(test_dir, transform=self.transform)

    def train_dataloader(self):

        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=0)


class CancerClassifier(pl.LightningModule):
    def __init__(self, num_classes=5, learning_rate=0.001):
        super().__init__()
        self.lr = learning_rate
        

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        

        self.fc1 = nn.Linear(128 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, num_classes) 

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

if __name__ == "__main__":
    device, BATCH_SIZE, IMG_SIZE = setup_environment()
    DATA_PATH = r"C:\Users\Kuba\Desktop\Projekt_ID374\data"

    print("ladowanie danych")
    dm = CancerDataModule(DATA_PATH, BATCH_SIZE, IMG_SIZE)
    

    print("Tworzenie modelu sieci neuronowej")
    model = CancerClassifier(num_classes=5)
    

    print("ROZPOCZYNAM TRENING")
    trainer = pl.Trainer(
        max_epochs=5,               
        accelerator="gpu",             
        devices=1,
        log_every_n_steps=10
    )
    
    trainer.fit(model, dm)
    
    print("Trening zakonczony!")


 