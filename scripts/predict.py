import sys
import os


import argparse
import logging
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateFinder
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from pathlib import Path


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from torchmdnet import datasets


def get_args():
    parser = argparse.ArgumentParser(description="Do a prediction on test set from saved checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint")
    parser.add_argument('--dataset', default=None, type=str, choices=datasets.__all__, help='Name of the torch_geometric dataset')
    parser.add_argument('--dataset-root', default='data', type=str, help='Data storage directory (not used if dataset is "CG")')
    parser.add_argument('--dataset-args', default=None, type=str, help='Additional dataset argument, e.g. an array for target properties for QM9 or molecule for MD17. If not provided, all properties are used')
    # TODO (armin) add literal_eval for dataset-args
    parser.add_argument('--structure', choices=["precise3d", "rdkit3d", "optimized3d", "rdkit2d"], default="precise3d", help='Structure of the input data')

    parser.add_argument('--log-dir', '-l', default='/tmp/logs', help='log file')

    parser.add_argument('--inference-batch-size', type=int, default=32, help='Batch size for inference')


    args = parser.parse_args()

    args.log_dir = str(Path(args.log_dir, args.job_id))
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    if args.redirect:
        sys.stdout = open(os.path.join(args.log_dir, "log"), "w")
        sys.stderr = sys.stdout
        logging.getLogger("pytorch_lightning").addHandler(
            logging.StreamHandler(sys.stdout)
        )

    return args

def main():
    pass 