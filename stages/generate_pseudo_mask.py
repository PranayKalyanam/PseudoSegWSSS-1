from stages.pseudo_mask_stage.generate_masks import _generate_masks
from stages.pseudo_mask_stage.pseudo_mask_utils import _build_pseudo_mask_model, _create_generation_result, _create_output_directories, _create_palette
from stages.stage1.stage1_utils import _load_stage1_checkpoint
from utils.logger import get_logger


def generate_pseudo_masks(
    config,
):
    """
    Generate pseudo segmentation masks using the trained
    Stage-1 classification network.

    Returns
    -------
    PseudoMaskGenerationResult
    """
    
    logger = get_logger(
            name="pseudo_masks_generation",
            log_directory="logs/pseudo_masks_generation",
            log_level=config.log_level,
        )
    
    logger.info("=" * 80)
    logger.info("Stage-1 Pseudo Mask Generation")
    logger.info("=" * 80)
    logger.info(f"Dataset            : {config.dataset}")
    logger.info(f"Training Data      : {config.dataroot}")
    logger.info(f"Network            : {config.stage1_network}")
    logger.info(f"Number of Classes  : {config.n_class}")
    logger.info("")


    # ------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------
    logger.info("[1/5] Building Stage-1 classification model")

    model = _build_pseudo_mask_model(
        config=config,
        logger=logger,
    )
    
    logger.info("Stage-1 model constructed successfully.")
    logger.info("")

    # ------------------------------------------------------------
    # Load Stage-1 checkpoint
    # -----------------------------------------------------------
    
    logger.info("[2/5] Loading best Stage-1 checkpoint")

    checkpoint_path = _load_stage1_checkpoint(
        config=config,
        model=model,
        logger=logger,
    )
    
    logger.info(f"Checkpoint Loaded : {checkpoint_path}")
    logger.info("")

    # ------------------------------------------------------------
    # Create color palette
    # ------------------------------------------------------------
    
    logger.info("[3/5] Creating visualization palette")

    palette = _create_palette(
        config.dataset,
        logger=logger,
    )
    
    logger.info("Color palette created.")
    logger.info("")

    # ------------------------------------------------------------
    # Create output directories
    # ------------------------------------------------------------
    
    logger.info("[4/5] Preparing output directories")

    output_paths = _create_output_directories(
        config=config,
        logger=logger,
    )
    
    logger.info("Output directories created.")


    # ------------------------------------------------------------
    # Generate pseudo masks
    # ------------------------------------------------------------
    
    logger.info("[5/5] Generating pseudo masks")

    _generate_masks(
        config=config,
        model=model,
        palette=palette,
        output_paths=output_paths,
        logger=logger,
    )
    
    logger.info("")
    logger.info("Pseudo-mask generation completed successfully.")
    logger.info("")
    
   

    # ------------------------------------------------------------
    # Package results
    # ------------------------------------------------------------

    result = _create_generation_result(
        config=config,
        checkpoint_path=checkpoint_path,
        output_paths=output_paths,
        logger=logger,
    )
    
    logger.info("=" * 80)
    logger.info("Pseudo Mask Generation Completed")
    logger.info("=" * 80)
    logger.info(f"Checkpoint Used : {result.checkpoint_path}")

    logger.info("Generated Directories:")

    for feature_map, directory in result.output_paths.items():

        logger.info(
            f"  {feature_map:<8} : {directory}"
        )

    logger.info(f"Generation Time : {result.generation_time}")
    logger.info("=" * 80)

    return result