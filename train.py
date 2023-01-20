import os.path as osp
import argparse
import torch
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import Trainer
from config.default import get_config
from model import SaccadingRNN
import os
from pathlib import Path


def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=0,
    )

    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    return parser


if __name__ == '__main__':

    OVERFIT = False
    seed = 0
    config = './config/large_sin.yaml'
    version = 34

    variant = osp.split(config)[1].split('.')[0]
    config = get_config(config)
    root = Path(f'runs/{variant}-{seed}/lightning_logs/')

    model_ckpt = list(root.joinpath(f"version_{version}").joinpath('checkpoints').glob("*"))[0]
    checkpoint_dir = Path(root.joinpath(f"version_{version}").joinpath('checkpoints'))
    
    prev_dir = None
    save_to = 0

    for i in range(50):

        to_save_dir = str(checkpoint_dir) + '/' + 'final'+ str(save_to) + ".ckpt"

        if not os.path.isfile(to_save_dir):
            if prev_dir is not None:
                model_ckpt = prev_dir
            break

        save_to += 1
        prev_dir = to_save_dir

    epochs = config.TRAIN.EPOCHS


    if config.TASK.NAME == 'TROXLER':
        dataset = TroxlerDataset(config, split="train50")
    else:
        raise NotImplementedError

    length = len(dataset)

    if OVERFIT:
        train = dataset
        val = dataset
    else:
        train, val = random_split(
            dataset,
            [int(length * 0.93), length - int(length * 0.93)],
            generator=torch.Generator().manual_seed(42)
        )
    print("Training on ", len(train), " examples")

    model = SaccadingRNN(config)
    lr_logger = LearningRateMonitor(logging_interval='step')

    trainer = Trainer(resume_from_checkpoint=model_ckpt, max_epochs = epochs,
            gpus=1,
        val_check_interval=1.0,
        callbacks=[lr_logger],
        default_root_dir=f"./runs/{config.VARIANT}-{config.SEED}",
                )
    num_workers = 32
    trainer.fit(model,
           DataLoader(train, batch_size=config.TRAIN.BATCH_SIZE, num_workers=num_workers),
        DataLoader(val, batch_size=config.TRAIN.BATCH_SIZE, num_workers=num_workers))
    trainer.save_checkpoint(to_save_dir)
    # automatically restores model, epoch, step, LR schedulers, apex, etc...

