"""
stage2_train.py

Stage-2 Semantic Segmentation Training

This module orchestrates the complete Stage-2 training pipeline.
It does not implement the individual algorithms; instead it
coordinates model creation, dataset preparation, training,
validation, testing, and checkpoint management.
"""

import os
import time

from stages.stage2.stage2_utils import (
    _build_stage2_dataloader,
    _build_stage2_model,
    _build_optimizer,
    _build_scheduler,
    _build_loss_function,
    _build_evaluator,
    _load_stage1_classifier,
    _load_pretrained_weights,
    _load_stage2_checkpoint,
    _train_one_epoch,
    _validate_one_epoch,
    _test_model,
    _create_stage2_result,
)
from utils.logger import get_logger




def stage2_train(
    config,
    iteration,
    iteration_manager,
):
    """
    Train the Stage-2 semantic segmentation network.

    Parameters
    ----------
    config : CurriculumConfig

    iteration : int

    iteration_manager : IterationManager

    Returns
    -------
    Stage2TrainingResult
    """

    logger = get_logger(
        name="stage2_train",
        log_directory="logs/stage2_train",
        log_level=config.log_level,
    )

    logger.info("=" * 80)
    logger.info("Stage-2 Semantic Segmentation Training")
    logger.info("=" * 80)
    logger.info(f"{'Curriculum Iteration':<25}: {iteration}")
    logger.info(f"{'Dataset':<25}: {config.dataset}")
    logger.info(f"{'Backbone':<25}: {config.backbone}")
    logger.info(f"{'Epochs':<25}: {config.stage2_epochs}")
    logger.info(f"{'Learning Rate':<25}: {config.stage2_lr}")
    logger.info(f"{'Batch Size':<25}: {config.batch_size}")
    logger.info(f"{'Resume Checkpoint':<25}: {config.stage2_resume}")
    logger.info(f"{'Fine-tune Mode':<25}: {config.stage2_ft}")
    logger.info("")

    # ------------------------------------------------------------
    # Initialize
    # ------------------------------------------------------------

    start_time = time.time()

    # ------------------------------------------------------------
    # Build dataloaders
    # ------------------------------------------------------------

    logger.info("[1/10] Building Stage-2 dataloaders...")

    # Train
    config.stage2_train_image_dir = os.path.join(config.dataroot, "train5")
    config.stage2_train_label_dir = os.path.join(config.dataroot, "train_PM", "PM_bn7")
    config.stage2_train_label_dir_a = os.path.join(config.dataroot, "train_PM", "PM_b5_2")
    config.stage2_train_label_dir_b = os.path.join(config.dataroot, "train_PM", "PM_b4_5")

    # Validation
    config.stage2_val_image_dir = os.path.join(config.dataroot, "val", "img")
    config.stage2_val_label_dir = os.path.join(config.dataroot, "val", "mask")

    # Test
    config.stage2_test_image_dir = os.path.join(config.dataroot, "test", "img")
    config.stage2_test_label_dir = os.path.join(config.dataroot, "test", "mask")

    logger.info(f"{'Pseudo-mask dir (main)':<25}: {config.stage2_train_label_dir}")
    logger.info(f"{'Pseudo-mask dir (A)':<25}: {config.stage2_train_label_dir_a}")
    logger.info(f"{'Pseudo-mask dir (B)':<25}: {config.stage2_train_label_dir_b}")

    dataloaders = _build_stage2_dataloader(
        config=config,
        logger=logger,
    )

    # Needed by `_build_scheduler`, matching the baseline's
    # `LR_Scheduler(..., len(self.train_loader))`.
    config.num_train_iterations = len(dataloaders["train"])
 
    logger.info("Stage-2 dataloaders created successfully.")
    logger.info("")


    # ------------------------------------------------------------
    # Build segmentation model
    # ------------------------------------------------------------

    logger.info("[2/10] Building segmentation model...")

    model = _build_stage2_model(
        config=config,
        logger=logger,
    )

    logger.info("Segmentation model created successfully.")
    logger.info("")

    # ------------------------------------------------------------
    # Build optimizer
    # ------------------------------------------------------------

    logger.info("[3/10] Building optimizer...")

    optimizer = _build_optimizer(
        config=config,
        model=model,
        logger=logger,
    )

    logger.info("Optimizer initialized.")
    logger.info("")

    # ------------------------------------------------------------
    # Build scheduler
    # ------------------------------------------------------------

    logger.info("[4/10] Building learning-rate scheduler...")

    scheduler = _build_scheduler(
        config=config,
        optimizer=optimizer,
        logger=logger,
    )

    logger.info("Scheduler initialized.")
    logger.info("")

    # ------------------------------------------------------------
    # Build loss function
    # ------------------------------------------------------------

    logger.info("[5/10] Building loss function...")

    criterion = _build_loss_function(
        config=config,
        logger=logger,
    )

    logger.info("Loss function initialized.")
    logger.info("")

    # ------------------------------------------------------------
    # Build evaluator
    # ------------------------------------------------------------

    logger.info("[6/10] Building evaluator...")

    evaluator = _build_evaluator(
        config=config,
        logger=logger,
    )

    logger.info("Evaluator initialized.")
    logger.info("")

    # ------------------------------------------------------------
    # Load Stage-1 classifier
    # ------------------------------------------------------------

    logger.info("[7/10] Loading Stage-1 classifier...")

    stage1_model = _load_stage1_classifier(
        config=config,
        logger=logger,
    )

    logger.info("Stage-1 classifier loaded successfully.")
    logger.info("")

    # ------------------------------------------------------------
    # Load Stage-2 weights
    # ------------------------------------------------------------

    logger.info("[8/10] Loading Stage-2 weights...")

    if iteration == 0:

        logger.info(
            "Iteration 0 detected. Loading ImageNet/DeepLab pretrained weights."
        )

        _load_pretrained_weights(
            config=config,
            model=model,
            logger=logger,
        )

    else:

        logger.info(
            "Loading previous best Stage-2 checkpoint for curriculum learning."
        )

        _load_stage2_checkpoint(
            config=config,
            model=model,
            logger=logger,
        )

    logger.info("Stage-2 weights loaded successfully.")
    logger.info("")

    state = {"best_pred": 0.0, "loss_history": [], "optimizer": optimizer}

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------

    logger.info("[9/10] Starting Stage-2 training...")
    logger.info("")

    for epoch in range(config.stage2_epochs):

        logger.info("-" * 80)
        logger.info(
            f"Epoch [{epoch + 1}/{config.stage2_epochs}]"
        )
        logger.info("-" * 80)

        _train_one_epoch(
            config=config,
            epoch=epoch,
            model=model,
            stage1_model=stage1_model,
            dataloaders=dataloaders,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            evaluator=evaluator,
            logger=logger,
            state=state,
        )

        _validate_one_epoch(
            config=config,
            epoch=epoch,
            model=model,
            dataloaders=dataloaders,
            criterion=criterion,
            evaluator=evaluator,
            logger=logger,
            state=state,
        )

        logger.info("")

    logger.info("Stage-2 training completed.")
    logger.info("")

    # ------------------------------------------------------------
    # Final Testing
    # ------------------------------------------------------------

    logger.info("[10/10] Evaluating best Stage-2 model on test dataset...")

    test_results = _test_model(
        config=config,
        model=model,
        dataloaders=dataloaders,
        evaluator=evaluator,
        logger=logger,
        stage1_model=stage1_model,
    )

    logger.info("Testing completed successfully.")
    logger.info("")

    # ------------------------------------------------------------
    # Package Results
    # ------------------------------------------------------------

    runtime = time.time() - start_time

    logger.info("=" * 80)
    logger.info("Stage-2 Training Completed")
    logger.info("=" * 80)
    logger.info(f"Curriculum Iteration : {iteration}")
    logger.info(f"Total Runtime        : {runtime:.2f} seconds")
    logger.info("=" * 80)

    return _create_stage2_result(
        config=config,
        iteration=iteration,
        model=model,
        test_results=test_results,
        runtime_seconds=runtime,
        logger=logger,
        state=state,
    )