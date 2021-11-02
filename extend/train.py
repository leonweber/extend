import logging
import os

import hydra
import pytorch_lightning as pl
from pprint import pprint

import torch.cuda
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from extend.data import transforms
from extend.data.processing import LinearBertProcessing
from extend.models import LinearBert

log = logging.getLogger(__name__)


@hydra.main(config_path="../config", config_name="train")
def main(config):

    checkpoint_callback = ModelCheckpoint(
        dirpath=".",
        save_weights_only=False,
        verbose=True,
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )

    dataset = hydra.utils.instantiate(config["dataset"])

    # TODO port to hydra
    # if config["transform"]:
    #     transform = getattr(transforms, config["transform"])()
    # else:
    #     transform = None

    train_data = hydra.utils.instantiate(
        config["processing"],
        dataset.train_data,
        linearize_graph=dataset.linearize_graph,
        transform=None,
    )
    dev_data = hydra.utils.instantiate(
        config["processing"],
        dataset.dev_data,
        edge_label_dict=train_data.edge_label_dict,
        node_label_dict=train_data.node_label_dict,
        linearize_graph=dataset.linearize_graph,
        transform=None,
    )

    logger = []
    pl.seed_everything(config["seed"])
    if not config["disable_wandb"]:
        logger.append(WandbLogger(project="extend"))

    if torch.cuda.is_available():
        gpus = 1
        precision = 16
    else:
        gpus = 0
        precision = 32

    trainer = pl.Trainer(
        gpus=gpus,
        accumulate_grad_batches=config["accumulate_grad_batches"],
        max_epochs=config["model"]["num_epochs"],
        callbacks=[checkpoint_callback],
        logger=logger,
        precision=precision,
        track_grad_norm=-1,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        gradient_clip_val=5.0
    )
    model: pl.LightningModule
    model = hydra.utils.instantiate(config["model"],
                                    edge_label_dict=train_data.edge_label_dict,
                                    node_label_dict=train_data.node_label_dict,
                                    train_size=len(train_data))
    if config["checkpoint"]:
        model = model.load_from_checkpoint(hydra.utils.to_absolute_path(config["checkpoint"]))

    trainer.fit(
        model,
        train_dataloader=DataLoader(
            train_data, batch_size=config["model"]["batch_size"], collate_fn=model.collate,
            shuffle=True
        ),
        val_dataloaders=DataLoader(
            dev_data, batch_size=config["model"]["batch_size"], collate_fn=model.collate
        ),
    )


if __name__ == '__main__':
    main()