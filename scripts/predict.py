import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from pytorch_lightning.loggers import CSVLogger, WandbLogger

import argparse
import logging
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateFinder
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from torchmdnet.module import LNNP
from torchmdnet.data import DataModule

from pathlib import Path
import urllib

from scripts.train import get_args


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from torchmdnet import datasets


def main():
    args = get_args()
    pl.seed_everything(args.seed, workers=True)
    print(args)

    data = DataModule(args)
    data.prepare_data()
    data.setup("fit")

    model = LNNP(args, mean=data.mean, std=data.std)
    metric_name = args.metric_name

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.log_dir,
        monitor=metric_name,
        save_top_k=3,
        filename="{step}-{epoch}-{"
        + metric_name
        + ":.4f}-{test_loss:.4f}-{train_per_step:.4f}",
        # every_n_epochs=args.save_interval,
        # save_last=True,
        mode=args.callback_mode,
    )
    early_stopping = EarlyStopping(
        metric_name, patience=args.early_stopping_patience, mode=args.callback_mode
    )

    tb_logger = pl.loggers.TensorBoardLogger(
        args.log_dir, name="tensorbord", version="", default_hp_metric=False
    )
    csv_logger = CSVLogger(args.log_dir, name="", version="")

    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        max_steps=args.num_steps,
        num_nodes=args.num_nodes,
        accelerator=args.accelerator,
        default_root_dir=args.log_dir,
        # resume_from_checkpoint=args.load_model, # TODO (armin) resume_from_chechpoint is deprecated but since load_model is None at moment, we will ignore it
        callbacks=[early_stopping, checkpoint_callback],
        logger=[tb_logger, csv_logger],
        reload_dataloaders_every_n_epochs=0,
        precision=args.precision,
        strategy="ddp",  # not supported for mps, REMEMBER!
    )

    trainer.test(model, datamodule=data)

    print("Done!")


if __name__ == "__main__":
    main()
