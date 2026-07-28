import os
from pathlib import Path
from venv import logger
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from tool.GenDataset import (
    Stage1_TrainDataset,
    Stage1CurriculumDataset,
)


def _build_train_dataloader(
    config,
    iteration,
    iteration_manager,
    runtime,
    logger=None,
):
    """
    Build the Stage-1 training dataloader.

    Responsibilities
    ----------------
    1. Create training transforms.
    2. Select the appropriate dataset.
    3. Construct the DataLoader.
    """
    if logger is None:
        logger = runtime.logger

    logger.info("-" * 80)
    logger.info("Building Training Dataset")
    logger.info("-" * 80)

    # ------------------------------------------------------------
    # Training augmentation
    # ------------------------------------------------------------

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
    ])

    # ------------------------------------------------------------
    # Dataset selection
    # ------------------------------------------------------------

    use_pseudo = iteration > config.use_pseudo_iteration

    if use_pseudo:
        logger.info("Curriculum Mode : Ground Truth + Pseudo Labels")

        pseudo_label_file = os.path.join(config.pseudo_labels_root, "pseudo_weak_labels.json")

        logger.info("Curriculum Mode : Ground Truth + Pseudo Labels")
        
        
        if not Path(pseudo_label_file).exists():

            logger.warning(
                "Pseudo label file not found: %s",
                pseudo_label_file,
            )

            logger.warning(
                "Using ground-truth labels as pseudo labels for this iteration."
            )

            dataset = Stage1CurriculumDataset(
                data_path=config.stage1_trainroot,
                pseudo_label_file=None,
                transform=train_transform,
                dataset=config.dataset,
                use_ground_truth_as_pseudo=True,
            )

        else:

            dataset = Stage1CurriculumDataset(
                data_path=config.stage1_trainroot,
                pseudo_label_file=pseudo_label_file,
                transform=train_transform,
                dataset=config.dataset,
            )

        # dataset = Stage1CurriculumDataset(
        #     data_path=config.stage1_trainroot,
        #     pseudo_label_file=iteration_manager.get_previous_pseudo_label_file(),
        #     transform=train_transform,
        #     dataset=config.dataset,
        # )

    else:

        logger.info("Curriculum Mode : Ground Truth Only")

        dataset = Stage1_TrainDataset(
            data_path=config.stage1_trainroot,
            transform=train_transform,
            dataset=config.dataset,
        )

    # ------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    logger.info(f"Training Samples : {len(dataset):,}")
    logger.info(f"Batch Size       : {config.batch_size}")
    logger.info(f"Iterations/Epoch : {len(train_loader):,}")
    logger.info(f"Workers          : {config.num_workers}")
    logger.info("Training dataloader successfully created.\n")

    return train_loader