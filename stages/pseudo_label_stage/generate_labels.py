"""
stages/pseudo_label_stage/generate_labels.py

The complete Stage-4 inference pipeline: run the segmentation model
over the training images, convert each predicted mask into an
image-level multi-label pseudo weak label, and save the results as
JSON.

This module contains no orchestration logic (that belongs to
`generate_pseudo_labels.py`) and no model/checkpoint/directory
construction (that belongs to `pseudo_label_utils.py`).
"""

import json
from datetime import datetime, timezone

import numpy as np
import torch
from tqdm import tqdm


# ------------------------------------------------------------
# Mask -> multi-label conversion
# ------------------------------------------------------------

def _convert_segmentation_to_multilabel(segmentation_prediction, num_classes, ignore_class):
    """
    Convert a single predicted segmentation mask into an image-level
    multi-label vector.

    A class is marked present (1) if it appears at least once anywhere
    in the predicted mask; otherwise it is marked absent (0). No
    confidence heuristics, thresholds, or prediction modifications are
    applied -- this is a pure presence/absence lookup.

    Parameters
    ----------
    segmentation_prediction : numpy.ndarray
        2-D array of predicted class indices, shape (H, W).
    num_classes : int
        Number of real segmentation classes (the length of the
        returned multi-label vector).
    ignore_class : int
        Pixel value representing the excluded/background class
        (matches Stage-2's convention of treating pixel value
        `config.n_class` as "exclude"). This value is skipped when
        determining which classes are present.

    Returns
    -------
    list of int
        Binary multi-label vector of length `num_classes`.
    """
    present_classes = set(np.unique(segmentation_prediction).tolist())
    present_classes.discard(ignore_class)

    multilabel = [
        1 if class_index in present_classes else 0
        for class_index in range(num_classes)
    ]

    return multilabel


# ------------------------------------------------------------
# Inference loop
# ------------------------------------------------------------

def _run_pseudo_label_inference(config, model, dataloader, iteration, logger=None):
    """
    Iterate over the inference dataloader, run segmentation inference,
    and convert each predicted mask into a pseudo weak label.

    Parameters
    ----------
    config : CurriculumConfig
    model : torch.nn.Module
        Segmentation model with Stage-2 weights already loaded, in
        eval mode.
    dataloader : torch.utils.data.DataLoader
        Yields (image_batch, filename_batch) pairs.
    iteration : int
        Current curriculum iteration, stored as metadata on each label.
    logger : logging.Logger, optional

    Returns
    -------
    dict
        Mapping of image filename -> label record, e.g.::

            {
                "image_0001": {
                    "pseudo_label": [1, 0, 1, 0],
                    "iteration": 0,
                    "source": "stage2",
                    "generated_at": "2026-07-26T18:30:00+00:00",
                }
            }
    """
    device = next(model.parameters()).device
    ignore_class = config.n_class

    if logger is not None:
        logger.info("-" * 80)
        logger.info("Generating Pseudo Weak Labels")
        logger.info("-" * 80)

    pseudo_labels = {}
    tbar = tqdm(dataloader, desc="\r")

    with torch.no_grad():
        for batch_index, (images, filenames) in enumerate(tbar):
            images = images.to(device)

            output = model(images)
            predictions = np.argmax(output.data.cpu().numpy(), axis=1)

            for sample_index, filename in enumerate(filenames):
                multilabel = _convert_segmentation_to_multilabel(
                    segmentation_prediction=predictions[sample_index],
                    num_classes=config.n_class,
                    ignore_class=ignore_class,
                )

                pseudo_labels[filename] = {
                    "pseudo_label": multilabel,
                    "iteration": iteration,
                    "source": "stage2",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }

                if logger is not None:
                    logger.info(f"Processing image {filename} -> {multilabel}")

    if logger is not None:
        logger.info("")
        logger.info(f"Pseudo weak labels generated for {len(pseudo_labels)} images.")
        logger.info("")

    return pseudo_labels


# ------------------------------------------------------------
# Saving
# ------------------------------------------------------------

def _save_pseudo_labels(pseudo_labels, output_path, logger=None):
    """
    Save the generated pseudo weak labels to a single JSON file.

    The original ground-truth weak labels are never touched -- pseudo
    labels are always written to a separate file/directory supplied by
    the caller.

    Parameters
    ----------
    pseudo_labels : dict
        Mapping of image filename -> label record, as produced by
        `_run_pseudo_label_inference`.
    output_path : str or pathlib.Path
        Destination JSON file path.
    logger : logging.Logger, optional

    Returns
    -------
    None
    """
    with open(output_path, "w") as f:
        json.dump(pseudo_labels, f, indent=4)

    if logger is not None:
        logger.info(f"Pseudo labels saved to: {output_path}")
        logger.info("")