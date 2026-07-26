
import importlib

import torch
from datetime import datetime

from data.pseudo_mask_result import PseudoMaskGenerationResult
from stages.stage1.stage1_utils import _load_stage1_checkpoint


def _build_pseudo_mask_model(
    config, 
    logger=None,
):
    """
    Build the trained Stage-1 classifier used for pseudo-mask generation.

    Parameters
    ----------
    config : CurriculumConfig

    Returns
    -------
    torch.nn.Module
        Stage-1 classifier loaded with the latest best checkpoint.
    """
    
    if logger:
        logger.info("=" * 80)
        logger.info("Building Stage-1 Model for Pseudo Mask Generation")
        logger.info("=" * 80)

    # ------------------------------------------------------------
    # Device
    # ------------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    
    if logger:
        logger.info(f"Device           : {device}")
        logger.info(f"Network          : {config.stage1_network}")
        logger.info(f"Number of Classes: {config.n_class}")

    # ------------------------------------------------------------
    # Build network
    # ------------------------------------------------------------

    model = getattr(
        importlib.import_module(
            config.stage1_network
        ),
        "Net_CAM",
    )(
        n_class=config.n_class,
    )

    model = model.to(device)

    # ------------------------------------------------------------
    # Load latest best Stage-1 checkpoint
    # ------------------------------------------------------------

    _load_stage1_checkpoint(
        config=config,
        model=model,
    )

    model.eval()
    
    if logger:
        logger.info("Stage-1 model ready for pseudo-mask generation.")
        logger.info("")

    return model



def _create_palette(dataset, logger=None,):
    """
    Create the color palette for pseudo-mask generation.

    Parameters
    ----------
    dataset : str
        Dataset name.

    Returns
    -------
    list[int]
        PIL color palette.
    """

    dataset = dataset.lower()
    
    if logger:
        logger.info("=" * 80)
        logger.info("Creating Color Palette")
        logger.info("=" * 80)
        logger.info(f"Dataset : {dataset}")

    palette = [0] * 15

    # ------------------------------------------------------------
    # LUAD-HistoSeg
    # ------------------------------------------------------------

    if dataset == "luad":

        palette[0:3] = [205, 51, 51]      # Tumor
        palette[3:6] = [0, 255, 0]        # Necrosis
        palette[6:9] = [65, 105, 225]     # Lymphocyte
        palette[9:12] = [255, 165, 0]     # Stroma
        palette[12:15] = [255, 255, 255]  # Background

    # ------------------------------------------------------------
    # BCSS
    # ------------------------------------------------------------

    elif dataset == "bcss":

        palette[0:3] = [255, 0, 0]        # Tumor
        palette[3:6] = [0, 255, 0]        # Stroma
        palette[6:9] = [0, 0, 255]        # Lymphocyte
        palette[9:12] = [153, 0, 255]     # Necrosis
        palette[12:15] = [255, 255, 255]  # Background

    else:

        raise ValueError(
            f"Unsupported dataset: {dataset}"
        )
        
    if logger:
        logger.info("Palette created successfully.")
        logger.info("")

    return palette



from pathlib import Path


def _create_output_directories(
    config,
    logger=None,
):
    """
    Create the output directories used for pseudo-mask generation.

    Directory Structure
    -------------------
    <dataset_root>/
    └── train_PM/
        ├── PM_b4_5/
        ├── PM_b5_2/
        └── PM_bn7/

    Parameters
    ----------
    config : CurriculumConfig

    Returns
    -------
    dict[str, Path]
        Dictionary containing the pseudo-mask output directories
        for each feature map.
    """
    
    if logger:
        logger.info("=" * 80)
        logger.info("Creating Output Directories")
        logger.info("=" * 80)

    # ------------------------------------------------------------
    # Root directory
    # ------------------------------------------------------------

    root_directory = Path(config.dataroot) / "train_PM"

    root_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    
    if logger:
        logger.info(f"Root Directory : {root_directory}")

    # ------------------------------------------------------------
    # Feature-map directories
    # ------------------------------------------------------------

    output_paths = {}

    feature_maps = [
        "b4_5",
        "b5_2",
        "bn7",
    ]

    for feature_map in feature_maps:

        directory = root_directory / f"PM_{feature_map}"

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_paths[feature_map] = directory
        
        if logger:
            logger.info(
                f"{feature_map:<8} -> {directory}"
            )
            
    if logger:
        logger.info("Output directories created successfully.")
        logger.info("")

    return output_paths




def _create_generation_result(
    config,
    checkpoint_path,
    output_paths,
    logger=None
):
    """
    Package pseudo-mask generation results.

    Parameters
    ----------
    config : CurriculumConfig

    checkpoint_path : pathlib.Path
        Stage-1 checkpoint used for pseudo-mask generation.

    output_paths : dict[str, pathlib.Path]
        Output directories containing the generated pseudo masks.

    Returns
    -------
    PseudoMaskGenerationResult
    """
    result = PseudoMaskGenerationResult(
        checkpoint_path=checkpoint_path,
        dataset=config.dataset,
        data_root=config.dataroot,
        output_paths=output_paths,
        generation_time=datetime.now(),
    )
    
    if logger:
        logger.info("=" * 80)
        logger.info("Pseudo Mask Generation Completed")
        logger.info("=" * 80)
        logger.info(f"Checkpoint Used : {checkpoint_path}")
        logger.info(f"Dataset         : {config.dataset}")
        logger.info(f"Data Root       : {config.dataroot}")
        logger.info("")

        logger.info("Generated Directories:")
        
    for feature_map, path in output_paths.items():
            logger.info(f"  {feature_map:<8}: {path}")

    logger.info("=" * 80)
    
    return result

   