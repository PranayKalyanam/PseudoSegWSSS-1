
import torch

from tool.infer_fun import create_pseudo_mask_v2, create_pseudo_mask_v3


def _generate_masks(
    config,
    model,
    palette,
    output_paths,
    logger=None
):
    """
    Generate pseudo masks from multiple feature maps.

    Parameters
    ----------
    config : CurriculumConfig

    model : torch.nn.Module
        Trained Stage-1 classifier.

    palette : list[int]
        Color palette used for visualization.

    output_paths : dict[str, Path]
        Output directories for each feature map.

    Returns
    -------
    dict[str, Path]
        Generated pseudo-mask directories.
    """

    # ------------------------------------------------------------
    # Feature maps used by the baseline
    # ------------------------------------------------------------

    feature_maps = [
        "b4_5",
        "b5_2",
        "bn7",
    ]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if logger is not None:
        logger.info("=" * 80)
        logger.info("Generating Pseudo Masks")
        logger.info("=" * 80)
        logger.info(f"Dataset            : {config.dataset}")
        logger.info(f"Training Root      : {config.dataroot}")
        logger.info(f"Number of Classes  : {config.n_class}")
        logger.info(f"Feature Maps       : {', '.join(feature_maps)}")
        logger.info("")

    # ------------------------------------------------------------
    # Generate pseudo masks
    # ------------------------------------------------------------

    for index, feature_map in enumerate(feature_maps, start=1):

        save_path = output_paths[feature_map]

        if logger is not None:
            logger.info(
                f"[{index}/{len(feature_maps)}] "
                f"Generating pseudo masks using feature map '{feature_map}'"
            )
            logger.info(f"Output Directory : {save_path}")


        # create_pseudo_mask_v2(
        #     model=model,
        #     dataroot=config.dataroot,
        #     fm=feature_map,
        #     savepath=str(save_path),
        #     n_class=config.n_class,
        #     palette=palette,
        #     dataset=config.dataset,
        #     logger=logger
        # )
        print(f"{feature_map} -> {save_path}")
        create_pseudo_mask_v3(
            model=model,
            layer_name=feature_map,
            dataroot=config.dataroot,
            savepath=str(save_path),
            n_class=config.n_class,
            palette=palette,
            dataset=config.dataset,
            device=device,
            logger=logger
        )
        
        if logger is not None:
            logger.info(
                f"Completed pseudo-mask generation for '{feature_map}'."
            )
            logger.info("")

    
    if logger is not None:
        logger.info("=" * 80)
        logger.info("All pseudo masks generated successfully.")
        logger.info("=" * 80)

    return output_paths