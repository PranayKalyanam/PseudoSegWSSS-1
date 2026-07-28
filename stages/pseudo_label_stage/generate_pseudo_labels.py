"""
stages/pseudo_label_stage/generate_pseudo_labels.py

Stage-4: Pseudo Weak Label Generation

This module orchestrates the complete Stage-4 pipeline. It contains
no algorithm implementations -- it only calls helper functions from
`pseudo_label_utils` (construction / loading / packaging) and
`generate_labels` (the inference pipeline itself), in this order:

    build segmentation model
    -> load best Stage-2 checkpoint
    -> create output directory
    -> build inference dataloader
    -> run inference and generate pseudo weak labels
    -> save pseudo weak labels
    -> package result

Stage-4 is pure inference. It never trains, and it never modifies the
original ground-truth weak labels.
"""

import os
import time
from datetime import datetime, timezone

from stages.stage2.stage2_utils import _load_stage1_classifier
from utils.logger import get_logger

from stages.pseudo_label_stage.pseudo_label_utils import (
    _build_pseudo_label_model,
    _load_stage2_checkpoint,
    _create_output_directory,
    _build_inference_dataloader,
    _create_generation_result,
)
from stages.pseudo_label_stage.generate_labels import (
    _run_pseudo_label_inference,
    _save_pseudo_labels,
)


def generate_pseudo_labels(
    config,
    iteration,
    iteration_manager,
):
    """
    Generate image-level multi-label pseudo weak labels from the
    trained Stage-2 segmentation model's predictions.

    Parameters
    ----------
    config : CurriculumConfig
    iteration : int
    iteration_manager : IterationManager

    Returns
    -------
    PseudoLabelGenerationResult
    """

    logger = get_logger(
        name="generate_pseudo_labels",
        log_directory="logs/pseudo_label_stage",
        log_level=config.log_level,
    )

    # ------------------------------------------------------------
    # Experiment banner
    # ------------------------------------------------------------

    config.stage4_input_image_dir = os.path.join(config.dataroot, "train5")

    logger.info("=" * 80)
    logger.info("Pseudo Weak Label Generation")
    logger.info("=" * 80)
    logger.info(f"Dataset             : {config.dataset}")
    logger.info(f"Iteration           : {iteration}")
    logger.info(f"Segmentation Model  : {config.backbone}")
    logger.info(f"Input Directory     : {config.stage4_input_image_dir}")
    logger.info(f"Device              : {'CUDA' if config.cuda else 'CPU'}")
    logger.info("")

    start_time = time.time()
    success = False

    # ------------------------------------------------------------
    # Build segmentation model
    # ------------------------------------------------------------

    model = _build_pseudo_label_model(
        config=config,
        logger=logger,
    )
    
    stage1_model = _load_stage1_classifier(
            config=config,
            logger=logger,
        )

    # ------------------------------------------------------------
    # Load best Stage-2 checkpoint
    # ------------------------------------------------------------

    checkpoint_path = _load_stage2_checkpoint(
        config=config,
        model=model,
        iteration_manager=iteration_manager,
        iteration=iteration,
        logger=logger,
    )

    # ------------------------------------------------------------
    # Create output directory
    # ------------------------------------------------------------

    output_directory = _create_output_directory(
        config=config,
        logger=logger,
    )

    # ------------------------------------------------------------
    # Build inference dataloader
    # ------------------------------------------------------------

    dataloader = _build_inference_dataloader(
        config=config,
        logger=logger,
    )

    # ------------------------------------------------------------
    # Generate pseudo weak labels
    # ------------------------------------------------------------

    pseudo_labels = _run_pseudo_label_inference(
        config=config,
        model=model,
        dataloader=dataloader,
        iteration=iteration,
        stage1_model=stage1_model,
        logger=logger,
    )

    output_path = os.path.join(output_directory, "pseudo_weak_labels.json")

    _save_pseudo_labels(
        pseudo_labels=pseudo_labels,
        output_path=output_path,
        logger=logger,
    )

    success = True
    number_of_images = len(dataloader.dataset)
    number_of_labels_generated = len(pseudo_labels)

    runtime = time.time() - start_time

    # ------------------------------------------------------------
    # Package results
    # ------------------------------------------------------------

    generation_time = datetime.now(timezone.utc).isoformat()

    logger.info("=" * 80)
    logger.info("Pseudo Weak Label Generation Summary")
    logger.info("=" * 80)
    logger.info(f"Images Processed        : {number_of_images}")
    logger.info(f"Pseudo Labels Generated : {number_of_labels_generated}")
    logger.info(f"Runtime                 : {runtime:.2f} seconds")
    logger.info(f"Output Directory        : {output_directory}")
    logger.info(f"Checkpoint Used         : {checkpoint_path}")
    logger.info("=" * 80)

    return _create_generation_result(
        config=config,
        iteration=iteration,
        checkpoint_path=checkpoint_path,
        output_directory=output_directory,
        number_of_images=number_of_images,
        number_of_labels_generated=number_of_labels_generated,
        generation_time=generation_time,
        runtime_seconds=runtime,
        success=success,
        logger=logger,
    )