import argparse
from lightning_model import DCSwin
from loader import get_dataloader

import lightning as L

def parse_args():
    pass

if __name__ == "__main__":
    

    train_loader, num_classes = get_dataloader("veg", "train", 2, 1.0)
    val_loader, num_classes = get_dataloader("veg", "val", 1, 1.0)
    

    learning_rate = 1e-4

    lightning_model = DCSwin(num_classes, learning_rate, train_loader, val_loader)

    trainer = L.Trainer(max_epochs=2, accelerator="cpu")

    trainer.fit(lightning_model, train_loader, val_loader)

