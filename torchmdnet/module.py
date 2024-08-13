import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.nn.functional import mse_loss, l1_loss, cross_entropy, binary_cross_entropy

from torchmetrics.classification import BinaryAUROC
from torch.nn import functional

from pytorch_lightning import LightningModule
from torchmdnet.models.model import create_model, load_model


class LNNP(LightningModule):
    def __init__(self, hparams, prior_model=None, mean=None, std=None):
        super(LNNP, self).__init__()
        self.save_hyperparameters(hparams)
        self.val_labels = []
        self.val_preds = []
        if self.hparams.load_model:
            self.model = load_model(self.hparams.load_model, args=self.hparams)
        elif self.hparams.pretrained_model:
            self.model = load_model(
                self.hparams.pretrained_model, args=self.hparams, mean=mean, std=std
            )
        else:
            self.model = create_model(self.hparams, prior_model, mean, std)

        # initialize exponential smoothing
        self.ema = None
        self._reset_ema_dict()

        # initialize loss collection
        self.losses = None
        self._reset_losses_dict()

        self.auc = {"val": [], "test": [], "train": []}
        self.b_AUROC = BinaryAUROC()

        self.preds_csv = []

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        if self.hparams.lr_schedule == "cosine":
            scheduler = CosineAnnealingLR(optimizer, self.hparams.lr_cosine_length)
            lr_scheduler = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        elif self.hparams.lr_schedule == "reduce_on_plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                "min",
                factor=self.hparams.lr_factor,
                patience=self.hparams.lr_patience,
                min_lr=self.hparams.lr_min,
            )
            lr_scheduler = {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }
        else:
            raise ValueError(f"Unknown lr_schedule: {self.hparams.lr_schedule}")
        return [optimizer], [lr_scheduler]

    def forward(self, z, pos, batch=None):
        return self.model(z, pos, batch=batch)

    def training_step(self, batch):
        return self.step(
            batch, getattr(functional, self.hparams.train_loss_fn), "train"
        )

    def validation_step(self, batch, *args):
        if len(args) == 0 or (len(args) > 0 and args[0] == 0):
            # validation step
            return self.step(
                batch, getattr(functional, self.hparams.val_test_loss_fn), "val"
            )
        # test step
        return self.step(
            batch, getattr(functional, self.hparams.val_test_loss_fn), "test"
        )

    def test_step(self, batch):
        return self.step(
            batch, getattr(functional, self.hparams.val_test_loss_fn), "test"
        )

    def on_test_end(self):
        # TODO (armin) doesn't work with multiple GPUs
        import csv

        header = ["smiles"]
        for target in self.hparams.dataset_args:
            header.append(f"pred_{target}")
            header.append(f"actual_{target}")
            if self.hparams.task_type == "class":
                header.append(f"pred_{target}_class")
        with open(self.hparams.log_dir + "/preds.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(self.preds_csv)

        result_dict = {
            "test_loss": torch.stack(self.losses["test"]).mean(),
        }
        print(f'Test loss: {result_dict["test_loss"]}')
        if self.hparams.task_type == "class":
            result_dict["test_auc"] = torch.stack(self.auc["test"]).mean()
            print(f'Test AUC: {result_dict["test_auc"]}')

        self.logger.log_metrics(result_dict)

    def step(self, batch, loss_fn, stage):
        with torch.set_grad_enabled(stage == "train" or self.hparams.derivative):
            # TODO: the model doesn't necessarily need to return a derivative once
            # Union typing works under TorchScript (https://github.com/pytorch/pytorch/pull/53180)
            pred, noise_pred, deriv = self(batch.z, batch.pos, batch.batch)

        denoising_is_on = (
            ("pos_target" in batch)
            and (self.hparams.denoising_weight > 0)
            and (noise_pred is not None)
        )

        loss_y, loss_dy, loss_pos = 0, 0, 0
        if self.hparams.derivative:
            if "y" not in batch:
                # "use" both outputs of the model's forward function but discard the first
                # to only use the derivative and avoid 'Expected to have finished reduction
                # in the prior iteration before starting a new one.', which otherwise get's
                # thrown because of setting 'find_unused_parameters=False' in the DDPPlugin
                deriv = deriv + pred.sum() * 0

            # force/derivative loss
            # TODO (armin) this is temporary
            if self.hparams.task_type == "class":
                target_not_minus_one = batch.dy != -1
                loss_dy = loss_fn(
                    deriv[target_not_minus_one], batch.dy[target_not_minus_one]
                )
            else:
                loss_dy = loss_fn(deriv, batch.dy)

            if stage in ["train", "val"] and self.hparams.ema_alpha_dy < 1:
                if self.ema[stage + "_dy"] is None:
                    self.ema[stage + "_dy"] = loss_dy.detach()
                # apply exponential smoothing over batches to dy
                loss_dy = (
                    self.hparams.ema_alpha_dy * loss_dy
                    + (1 - self.hparams.ema_alpha_dy) * self.ema[stage + "_dy"]
                )
                self.ema[stage + "_dy"] = loss_dy.detach()

            if self.hparams.force_weight > 0:
                self.losses[stage + "_dy"].append(loss_dy.detach())

        if "y" in batch:
            if (noise_pred is not None) and not denoising_is_on:
                # "use" both outputs of the model's forward (see comment above).
                pred = pred + noise_pred.sum() * 0

            if batch.y.ndim == 1:
                batch.y = batch.y.unsqueeze(1)

            # energy/prediction loss
            # TODO (armin) here as well, these are all generalizable and shouldn't be hardcoded based on task type
            if self.hparams.task_type == "class":
                target_not_minus_one = (
                    batch.y != -1
                )  # -1 is the default value for missing labels
                loss_y = loss_fn(
                    pred[target_not_minus_one], batch.y[target_not_minus_one]
                )
            else:
                loss_y = loss_fn(pred, batch.y)

            if stage in ["train", "val"] and self.hparams.ema_alpha_y < 1:
                if self.ema[stage + "_y"] is None:
                    self.ema[stage + "_y"] = loss_y.detach()
                # apply exponential smoothing over batches to y
                loss_y = (
                    self.hparams.ema_alpha_y * loss_y
                    + (1 - self.hparams.ema_alpha_y) * self.ema[stage + "_y"]
                )
                self.ema[stage + "_y"] = loss_y.detach()

            if self.hparams.energy_weight > 0:
                self.losses[stage + "_y"].append(loss_y.detach())

            # calculate AUC if the task is classification
            if self.hparams.task_type == "class":
                auc = self.b_AUROC(pred, batch.y)
                self.auc[stage].append(auc)

        if denoising_is_on:
            if "y" not in batch:
                # "use" both outputs of the model's forward (see comment above).
                noise_pred = noise_pred + pred.sum() * 0

            normalized_pos_target = self.model.pos_normalizer(batch.pos_target)
            loss_pos = mse_loss(noise_pred, normalized_pos_target)
            self.losses[stage + "_pos"].append(loss_pos.detach())

        # total loss
        loss = (
            loss_y * self.hparams.energy_weight
            + loss_dy * self.hparams.force_weight
            + loss_pos * self.hparams.denoising_weight
        )

        self.losses[stage].append(loss.detach())

        # Frequent per-batch logging for training
        if stage == "train":
            train_metrics = {
                k + "_per_step": v[-1]
                for k, v in self.losses.items()
                if (k.startswith("train") and len(v) > 0)
            }
            train_metrics["lr_per_step"] = self.trainer.optimizers[0].param_groups[0][
                "lr"
            ]
            train_metrics["step"] = self.trainer.global_step
            train_metrics["batch_pos_mean"] = batch.pos.mean().item()
            self.log_dict(train_metrics, sync_dist=True)
        elif stage == "test":
            for i in range(len(pred)):
                row = [batch.name[i]]
                for j in range(len(self.hparams.dataset_args)):
                    row.append(pred[i][j].item())
                    row.append(batch.y[i][j].item())
                    if self.hparams.task_type == "class":
                        row.append(int(round(pred[i][j].item())))
                        # row.append(int(pred[i][j].item() > 0.5))

                self.preds_csv.append(row)

        # if batch size is 1, and the loss was NaN, print the batch index
        if torch.isnan(loss_y):
            print(f"Processing data: {batch.name}")
            print(f"NaN loss in {batch.name}")
        # Add print statement here to see the name of the data being processed
        
        return loss

    def optimizer_step(self, *args, **kwargs):
        optimizer = kwargs["optimizer"] if "optimizer" in kwargs else args[2]
        if self.trainer.global_step < self.hparams.lr_warmup_steps:
            lr_scale = min(
                1.0,
                float(self.trainer.global_step + 1)
                / float(self.hparams.lr_warmup_steps),
            )

            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.lr
        super().optimizer_step(*args, **kwargs)
        optimizer.zero_grad()

    def on_train_epoch_end(self):
        dm = self.trainer.datamodule
        if hasattr(dm, "test_dataset") and len(dm.test_dataset) > 0:
            should_reset = (
                self.current_epoch % self.hparams.test_interval == 0
                or (self.current_epoch - 1) % self.hparams.test_interval == 0
            )
            if should_reset:
                # reset validation dataloaders before and after testing epoch, which is faster
                # than skipping test validation steps by returning None
                self.trainer.fit_loop.setup_data()

    # TODO(shehzaidi): clean up this function, redundant logging if dy loss exists.
    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            # construct dict of logged metrics
            result_dict = {
                "epoch": self.current_epoch,
                "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
                "train_loss": torch.stack(self.losses["train"]).mean(),
                "val_loss": torch.stack(self.losses["val"]).mean(),
            }

            # add test loss if available
            if len(self.losses["test"]) > 0:
                result_dict["test_loss"] = torch.stack(self.losses["test"]).mean()

            # if prediction and derivative are present, also log them separately
            if len(self.losses["train_y"]) > 0 and len(self.losses["train_dy"]) > 0:
                result_dict["train_loss_y"] = torch.stack(self.losses["train_y"]).mean()
                result_dict["train_loss_dy"] = torch.stack(
                    self.losses["train_dy"]
                ).mean()
                result_dict["val_loss_y"] = torch.stack(self.losses["val_y"]).mean()
                result_dict["val_loss_dy"] = torch.stack(self.losses["val_dy"]).mean()

                if len(self.losses["test"]) > 0:
                    result_dict["test_loss_y"] = torch.stack(
                        self.losses["test_y"]
                    ).mean()
                    result_dict["test_loss_dy"] = torch.stack(
                        self.losses["test_dy"]
                    ).mean()

            if len(self.losses["train_y"]) > 0:
                result_dict["train_loss_y"] = torch.stack(self.losses["train_y"]).mean()
            if len(self.losses["val_y"]) > 0:
                result_dict["val_loss_y"] = torch.stack(self.losses["val_y"]).mean()
            if len(self.losses["test_y"]) > 0:
                result_dict["test_loss_y"] = torch.stack(self.losses["test_y"]).mean()

            # if denoising is present, also log it
            if len(self.losses["train_pos"]) > 0:
                result_dict["train_loss_pos"] = torch.stack(
                    self.losses["train_pos"]
                ).mean()

            if len(self.losses["val_pos"]) > 0:
                result_dict["val_loss_pos"] = torch.stack(self.losses["val_pos"]).mean()

            if len(self.losses["test_pos"]) > 0:
                result_dict["test_loss_pos"] = torch.stack(
                    self.losses["test_pos"]
                ).mean()

            if self.hparams.task_type == "class":
                result_dict["val_auc"] = torch.stack(self.auc["val"]).mean()
                if len(self.auc["test"]) > 0:
                    result_dict["test_auc"] = torch.stack(self.auc["test"]).mean()

            self.log_dict(result_dict, sync_dist=True)
        self._reset_losses_dict()

    def _reset_losses_dict(self):
        self.losses = {
            "train": [],
            "val": [],
            "test": [],
            "train_y": [],
            "val_y": [],
            "test_y": [],
            "train_dy": [],
            "val_dy": [],
            "test_dy": [],
            "train_pos": [],
            "val_pos": [],
            "test_pos": [],
        }

    def _reset_ema_dict(self):
        self.ema = {"train_y": None, "val_y": None, "train_dy": None, "val_dy": None}
