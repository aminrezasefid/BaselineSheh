import numpy as np  # sometimes needed to avoid mkl-service error
import sys
import os

import urllib

import argparse
import logging
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateFinder
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from torchmdnet.module import LNNP
from torchmdnet import datasets, priors, models
from torchmdnet.data import DataModule
from torchmdnet.models import output_modules
from torchmdnet.models.utils import rbf_class_mapping, act_class_mapping
from torchmdnet.utils import LoadFromFile, LoadFromCheckpoint, save_argparse, number
from pathlib import Path
import wandb
import csv


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--load-model', action=LoadFromCheckpoint, help='Restart training using a model checkpoint')  # keep first
    parser.add_argument('--conf', '-c', type=open, action=LoadFromFile, help='Configuration yaml file')  # keep second
    parser.add_argument('--num-epochs', default=300, type=int, help='number of epochs')
    parser.add_argument('--num-steps', default=None, type=int, help='Maximum number of gradient steps.')
    parser.add_argument('--batch-size', default=32, type=int, help='batch size')
    parser.add_argument('--inference-batch-size', default=None, type=int, help='Batchsize for validation and tests.')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--lr-schedule', default="reduce_on_plateau", type=str, choices=['cosine', 'reduce_on_plateau'], help='Learning rate schedule.')
    parser.add_argument('--lr-patience', type=int, default=10, help='Patience for lr-schedule. Patience per eval-interval of validation')
    parser.add_argument('--lr-min', type=float, default=1e-6, help='Minimum learning rate before early stop')
    parser.add_argument('--lr-factor', type=float, default=0.8, help='Minimum learning rate before early stop')
    parser.add_argument('--lr-warmup-steps', type=int, default=0, help='How many steps to warm-up over. Defaults to 0 for no warm-up')
    parser.add_argument('--lr-cosine-length', type=int, default=400000, help='Cosine length if lr_schedule is cosine.')
    parser.add_argument('--early-stopping-patience', type=int, default=30, help='Stop training after this many epochs without improvement')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay strength')
    parser.add_argument('--ema-alpha-y', type=float, default=1.0, help='The amount of influence of new losses on the exponential moving average of y')
    parser.add_argument('--ema-alpha-dy', type=float, default=1.0, help='The amount of influence of new losses on the exponential moving average of dy')
    parser.add_argument('--ngpus', type=int, default=-1, help='Number of GPUs, -1 use all available. Use CUDA_VISIBLE_DEVICES=1, to decide gpus')
    parser.add_argument('--num-nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--precision', type=int, default=32, choices=[16, 32], help='Floating point precision')
    parser.add_argument('--log-dir', '-l', default='/tmp/logs', help='log file')
    parser.add_argument('--splits', default=None, help='Npz with splits idx_train, idx_val, idx_test')
    parser.add_argument('--train-size', type=number, default=None, help='Percentage/number of samples in training set (None to use all remaining samples)')
    parser.add_argument('--val-size', type=number, default=0.05, help='Percentage/number of samples in validation set (None to use all remaining samples)')
    parser.add_argument('--test-size', type=number, default=0.1, help='Percentage/number of samples in test set (None to use all remaining samples)')
    parser.add_argument('--test-interval', type=int, default=10, help='Test interval, one test per n epochs (default: 10)')
    parser.add_argument('--save-interval', type=int, default=10, help='Save interval, one save per n epochs (default: 10)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--distributed-backend', default='ddp', help='Distributed backend: dp, ddp, ddp2')
    parser.add_argument('--accelerator', default='auto', help='Accelerator types: “cpu”, “gpu”, “tpu”, “ipu”, “hpu”, “mps”, “auto”')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for data prefetch')
    parser.add_argument('--redirect', type=bool, default=False, help='Redirect stdout and stderr to log_dir/log')
    parser.add_argument('--wandb-notes', default="", type=str, help='Notes passed to wandb experiment.')
    parser.add_argument('--job-id', default="auto", type=str, help='Job ID. If auto, pick the next available numeric job id.')
    parser.add_argument('--pretrained-model', default=None, type=str, help='Pre-trained weights checkpoint.')
    parser.add_argument('--strict-load', type=bool, default=False, help='load weights strictly.')
    parser.add_argument('--split', type=str, default='scaffold', choices=['random', 'scaffold'], help='Split type')


    # dataset specific
    parser.add_argument('--dataset', default=None, type=str, choices=datasets.__all__, help='Name of the torch_geometric dataset')
    parser.add_argument('--dataset-root', default='data', type=str, help='Data storage directory (not used if dataset is "CG")')
    parser.add_argument('--dataset-args', default=None, type=list[str], help='Additional dataset argument, e.g. an array for target properties for QM9 or molecule for MD17. If not provided, all properties are used')
    # TODO (armin) add literal_eval for dataset-args
    parser.add_argument('--structure', choices=["precise3d", "rdkit3d", "optimized3d", "rdkit2d", "pubchem3d"], default="precise3d", help='Structure of the input data')
    parser.add_argument('--coord-files', default=None, type=str, help='Custom coordinate files glob')
    parser.add_argument('--embed-files', default=None, type=str, help='Custom embedding files glob')
    parser.add_argument('--energy-files', default=None, type=str, help='Custom energy files glob')
    parser.add_argument('--force-files', default=None, type=str, help='Custom force files glob')
    parser.add_argument('--energy-weight', default=1.0, type=float, help='Weighting factor for energies in the loss function')
    parser.add_argument('--force-weight', default=1.0, type=float, help='Weighting factor for forces in the loss function')
    parser.add_argument('--position-noise-scale', default=0., type=float, help='Scale of Gaussian noise added to positions.')
    parser.add_argument('--denoising-weight', default=0., type=float, help='Weighting factor for denoising in the loss function.')
    parser.add_argument('--denoising-only', type=bool, default=False, help='If the task is denoising only (then val/test datasets also contain noise).')

    # TODO (armin) ask what the conformer is

    # model architecture
    parser.add_argument('--model', type=str, default='graph-network', choices=models.__all__, help='Which model to train')
    parser.add_argument('--output-model', type=str, default='Scalar', choices=output_modules.__all__, help='The type of output model')
    parser.add_argument('--prior-model', type=str, default=None, choices=priors.__all__, help='Which prior model to use')
    parser.add_argument('--output-model-noise', type=str, default=None, choices=output_modules.__all__ + ['VectorOutput'], help='The type of output model for denoising')

    # architectural args
    parser.add_argument('--embedding-dimension', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--num-layers', type=int, default=6, help='Number of interaction layers in the model')
    parser.add_argument('--num-rbf', type=int, default=64, help='Number of radial basis functions in model')
    parser.add_argument('--activation', type=str, default='silu', choices=list(act_class_mapping.keys()), help='Activation function')
    parser.add_argument('--rbf-type', type=str, default='expnorm', choices=list(rbf_class_mapping.keys()), help='Type of distance expansion')
    parser.add_argument('--trainable-rbf', type=bool, default=False, help='If distance expansion functions should be trainable')
    parser.add_argument('--neighbor-embedding', type=bool, default=False, help='If a neighbor embedding should be applied before interactions')
    parser.add_argument('--aggr', type=str, default='add', help='Aggregation operation for CFConv filter output. Must be one of \'add\', \'mean\', or \'max\'')
    parser.add_argument('--task-type',type=str,default="regr",choices=["regr","class"],help="model used for classification or regression")
    parser.add_argument('--out-channels',type=int,default=1,help="number of output neurons, must be the same as the number of properties to predict")
    loss_function_choices = ["mse_loss", "l1_loss", "cross_entropy", "binary_cross_entropy"]
    parser.add_argument('--train-loss-fn', choices= loss_function_choices, default="l1_loss", help='Loss function for training')
    parser.add_argument('--val-test-loss-fn', choices= loss_function_choices, default="l1_loss", help='Loss function for validation and test')

    # Transformer specific
    parser.add_argument('--distance-influence', type=str, default='both', choices=['keys', 'values', 'both', 'none'], help='Where distance information is included inside the attention')
    parser.add_argument('--attn-activation', default='silu', choices=list(act_class_mapping.keys()), help='Attention activation function')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--layernorm-on-vec', type=str, default=None, choices=['whitened'], help='Whether to apply an equivariant layer norm to vec features. Off by default.')

    # other args
    parser.add_argument('--derivative', default=False, type=bool, help='If true, take the derivative of the prediction w.r.t coordinates')
    parser.add_argument('--cutoff-lower', type=float, default=0.0, help='Lower cutoff in model')
    parser.add_argument('--cutoff-upper', type=float, default=5.0, help='Upper cutoff in model')
    parser.add_argument('--atom-filter', type=int, default=-1, help='Only sum over atoms with Z > atom_filter')
    parser.add_argument('--max-z', type=int, default=100, help='Maximum atomic number that fits in the embedding matrix')
    parser.add_argument('--max-num-neighbors', type=int, default=32, help='Maximum number of neighbors to consider in the network')
    parser.add_argument('--standardize', type=bool, default=False, help='If true, multiply prediction by dataset std and add mean')
    parser.add_argument('--reduce-op', type=str, default='add', choices=['add', 'mean'], help='Reduce operation to apply to atomic predictions')
    parser.add_argument('--metric-name', type=str, default='val_loss', help='parameter name to be followed by call backs')
    parser.add_argument('--callback-mode', type=str, default='min', help='callback mode')
    # fmt: on

    parser.add_argument(
        "--testing-noisy",
        type=bool,
        default=False,
        help="If true, add noise to the test set",
    )
    parser.add_argument(
        "--fine-tuned-checkpoint",
        type=str,
        default=None,
        help="If specified, download and load fine-tuned checkpoint from this URL",
    )

    args = parser.parse_args()

    args.log_dir = str(Path(args.log_dir, args.job_id))
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    if args.redirect:
        sys.stdout = open(os.path.join(args.log_dir, "log"), "w")
        sys.stderr = sys.stdout
        logging.getLogger("pytorch_lightning").addHandler(
            logging.StreamHandler(sys.stdout)
        )

    if args.inference_batch_size is None:
        args.inference_batch_size = args.batch_size

    if args.fine_tuned_checkpoint:
        finetuned_ckpt_name = os.path.basename(args.fine_tuned_checkpoint)
        finetuned_ckpt_path = os.path.join("checkpoints", finetuned_ckpt_name)
        if not os.path.exists(finetuned_ckpt_path):
            try:
                urllib.request.urlretrieve(
                    args.fine_tuned_checkpoint, finetuned_ckpt_path
                )
            except:
                raise ValueError(
                    f"Could not download pretrained model from {args.fine_tuned_checkpoint}."
                )
        args.fine_tuned_checkpoint = finetuned_ckpt_path

    save_argparse(args, os.path.join(args.log_dir, "input.yaml"), exclude=["conf"])

    return args


def main():
    args = get_args()
    pl.seed_everything(args.seed, workers=True)
    print(args)

    # initialize data module
    data = DataModule(args)
    data.prepare_data()
    data.setup("fit")

    prior = None
    if args.prior_model:
        assert hasattr(priors, args.prior_model), (
            f"Unknown prior model {args['prior_model']}. "
            f"Available models are {', '.join(priors.__all__)}"
        )
        # initialize the prior model
        prior = getattr(priors, args.prior_model)(dataset=data.dataset)
        args.prior_args = prior.get_init_args()

    # initialize lightning module
    model = LNNP(args, prior_model=prior, mean=data.mean, std=data.std)
    metric_name = args.metric_name

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.log_dir,
        monitor=metric_name,
        save_top_k=3,
        filename="{step}-{epoch}-{"
        + metric_name
        + ":.4f}-{test_loss:.4f}-{train_per_step:.4f}",
        # every_n_epochs=args.save_interval,
        save_last=True,
        mode=args.callback_mode,
    )
    early_stopping = EarlyStopping(
        metric_name, patience=args.early_stopping_patience, mode=args.callback_mode
    )

    tb_logger = pl.loggers.TensorBoardLogger(
        args.log_dir, name="tensorbord", version="", default_hp_metric=False
    )
    csv_logger = CSVLogger(args.log_dir, name="", version="")

    # wandb_logger = WandbLogger(
    #     name=args.job_id,
    #     project='pre-training-via-denoising',
    #     log_model='all',
    #     settings=wandb.Settings(start_method='fork'),
    #     notes=args.wandb_notes,
    #     save_dir=args.log_dir,
    #     id=args.job_id + f"_{wandb.util.generate_id()}",  # Ensures unique run ID
    #     tags=[args.dataset]
    # )

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

    trainer.fit(model, datamodule=data)

    tester = pl.Trainer(
        default_root_dir=args.log_dir,
        max_epochs=1,
        max_steps=1,
        num_nodes=args.num_nodes,
        accelerator=args.accelerator,
        logger=False,
        callbacks=[early_stopping, checkpoint_callback],
        precision=args.precision,
        strategy="ddp",  # not supported for mps, REMEMBER!
    )
    tester.test(datamodule=data, ckpt_path="best", model=model)

    print("Done!")


if __name__ == "__main__":
    main()
