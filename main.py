import argparse
from lightning_model import DCSwin
from loader import get_dataloader

import lightning as L

def parse_args():
    pass

if __name__ == "__main__":

    train_loader, num_classes = get_dataloader("combined", "train", 16, 1.0)
    val_loader, num_classes = get_dataloader("combined", "val", 1, 1.0)

    learning_rate = 1e-3
    
    model_size = "base"

    lightning_model = DCSwin(num_classes, learning_rate, train_loader, val_loader, model_size=model_size)

    model_checkpoint = L.pytorch.callbacks.ModelCheckpoint(
        dirpath="checkpoints/",
        filename=f"{model_size}_"+ "{epoch}-{val_iou:.2f}",
        monitor="val_iou",
        mode="max",
        save_top_k=1,
        save_last=True,
        verbose=True
    )

    trainer = L.Trainer(max_epochs=100, accelerator="gpu", devices=[7], callbacks=[model_checkpoint])

    trainer.fit(lightning_model, train_loader, val_loader)

