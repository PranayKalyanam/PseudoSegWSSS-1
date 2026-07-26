"""
stages/pseudo_label_stage/pseudo_label_utils.py

Construction, weight-loading, dataset, and result-packaging helpers
for Stage-4 (Pseudo Weak Label Generation).

Each function has exactly one responsibility:

    _build_pseudo_label_model()   -> construct the segmentation model
    _load_stage2_checkpoint()     -> load this iteration's best Stage-2 weights
    _create_output_directory()    -> resolve/return this iteration's pseudo-label directory
    _build_inference_dataloader() -> lightweight image-only inference dataset/loader
    _create_generation_result()   -> package a PseudoLabelGenerationResult

Assumed config attributes
--------------------------
config.n_class            : int   number of real segmentation classes
config.cuda                : bool
config.stage2_backbone     : str
config.stage2_out_stride   : int
config.stage2_sync_bn      : bool
config.stage2_freeze_bn    : bool
config.dataset             : str
config.dataroot            : str
config.batch_size   : int
config.num_workers         : int

Rename to match your real CurriculumConfig if they differ.
"""

import os
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from tool import custom_transforms as tr
from stages.stage2.stage2_utils import _build_stage2_model


@dataclass
class PseudoLabelGenerationResult:
    """
    Packaged outcome of one call to `generate_pseudo_labels`.

    Attributes
    ----------
    dataset : str
    iteration : int
    checkpoint_path : str
        Stage-2 checkpoint used to generate these pseudo labels.
    output_directory : str
        Directory the pseudo-label JSON was written to.
    number_of_images : int
        Number of images inference was run over.
    number_of_labels_generated : int
        Number of pseudo-label entries written (== number_of_images
        under normal operation; tracked separately in case future
        strategies filter some images out).
    generation_time : str
        ISO-8601 timestamp marking when generation completed.
    runtime_seconds : float
    success : bool
    """

    dataset: str
    iteration: int
    checkpoint_path: str
    output_directory: str
    number_of_images: int
    number_of_labels_generated: int
    generation_time: str
    runtime_seconds: float
    success: bool



# ------------------------------------------------------------
# Model construction
# ------------------------------------------------------------

def _build_pseudo_label_model(config, logger=None):
    """
    Construct the segmentation model used for Stage-4 inference.

    Built exactly the same way as Stage-2 (same backbone, output
    stride, SyncBN/FreezeBN configuration, and device placement) by
    delegating directly to `_build_stage2_model` -- the architecture
    must not diverge from what Stage-2 was trained with. Checkpoint
    loading is intentionally NOT performed here; that is
    `_load_stage2_checkpoint`'s responsibility.

    Parameters
    ----------
    config : CurriculumConfig
    logger : logging.Logger, optional

    Returns
    -------
    torch.nn.Module
        The (untrained-weights) segmentation model, on the resolved device.
    """
    if logger is not None:
        logger.info("=" * 80)
        logger.info("Building Segmentation Model (Stage-4 Inference)")
        logger.info("=" * 80)
        logger.info(f"Backbone          : {config.backbone}")
        logger.info(f"Classes           : {config.n_class}")

    model = _build_stage2_model(config=config, logger=logger)

    if logger is not None:
        logger.info("Segmentation model created successfully.")
        logger.info("")

    return model


# ------------------------------------------------------------
# Checkpoint loading
# ------------------------------------------------------------

def _load_stage2_checkpoint(config, model, iteration_manager, iteration, logger=None):
    """
    Load this curriculum iteration's best Stage-2 checkpoint into the
    Stage-4 inference model.

    Note: unlike Stage-2's own `_load_stage2_checkpoint` (which loads
    the *previous* iteration's checkpoint for curriculum continuation
    at training time), Stage-4 always loads the *current* iteration's
    checkpoint, since pseudo-label generation runs immediately after
    that iteration's Stage-2 training completes.

    Parameters
    ----------
    config : CurriculumConfig
    model : torch.nn.Module
    iteration_manager : IterationManager
    iteration : int
    logger : logging.Logger, optional

    Returns
    -------
    str
        Path to the checkpoint that was loaded.
    """
    device = config.device
    # device = torch.device("cuda" if torch.cuda.is_available() and config.cuda else "cpu")
    # config.device = device

    checkpoint_path = config.stage2_checkpoint

    if logger is not None:
        logger.info(f"Checkpoint          : {checkpoint_path}")

    if not os.path.isfile(checkpoint_path):
        raise RuntimeError(f"=> no Stage-2 checkpoint found at '{checkpoint_path}'")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if hasattr(model, "module"):
        model.module.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
        model.load_state_dict(state_dict, strict=False)

    model.eval()

    if logger is not None:
        logger.info("Stage-2 checkpoint loaded successfully (eval mode).")
        logger.info("")

    return str(checkpoint_path)


# ------------------------------------------------------------
# Output directory
# ------------------------------------------------------------

def _create_output_directory(config, logger=None):
    """
    Resolve the flat, dataset-level pseudo-weak-label output directory.

    Unlike Stage-2's per-iteration checkpoints, pseudo weak labels are
    written to a single fixed location that Stage-1 always reads from
    for its "next iteration" supervision (ground-truth weak labels +
    pseudo weak labels). Each curriculum iteration OVERWRITES the same
    file rather than creating a new one -- traceability across
    iterations is preserved via the `"iteration"` field embedded in
    each label record (see `_run_pseudo_label_inference`), not via the
    directory structure.

    Parameters
    ----------
    config : CurriculumConfig
        Must have `config.dataroot` set. The resolved directory is
        `{config.dataroot}/train_PL`.
    logger : logging.Logger, optional

    Returns
    -------
    str
        The pseudo-weak-label output directory (created if missing).
    """
    output_directory = os.path.join(config.pseudo_labels_root)
    os.makedirs(output_directory, exist_ok=True)

    if logger is not None:
        logger.info(f"Output Directory    : {output_directory}")

    return output_directory


# ------------------------------------------------------------
# Inference dataset / dataloader
# ------------------------------------------------------------

class _PseudoLabelInferenceDataset(Dataset):
    """
    Lightweight, image-only dataset for Stage-4 inference.

    Isolated from `Stage2DatasetV2` because Stage-4 needs neither
    pixel-level labels nor the train/val/test split machinery -- only
    the training images and their filenames (used as keys in the
    saved pseudo-label JSON). Preprocessing is kept identical to
    Stage-2 by reusing the same `tr.Normalize()` / `tr.ToTensor()`
    transforms; a dummy zero label is passed through the transform
    pipeline (which expects an "image" + "label" pair) and discarded
    afterward.

    Parameters
    ----------
    image_dir : str
        Directory of images to run inference over.
    """

    def __init__(self, image_dir):
        super().__init__()

        self.image_dir = image_dir

        self.filenames = sorted(
            os.path.splitext(file)[0]
            for file in os.listdir(self.image_dir)
            if not file.startswith(".")
        )

        self.images = [
            os.path.join(self.image_dir, f"{name}.png")
            for name in self.filenames
        ]

        self.transform = transforms.Compose(
            [
                tr.Normalize(),
                tr.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        dummy_label = Image.new("L", image.size)

        sample = self.transform({"image": image, "label": dummy_label})

        return sample["image"], self.filenames[index]


def _build_inference_dataloader(config, logger=None):
    """
    Build the Stage-4 inference DataLoader over the training images.

    Parameters
    ----------
    config : CurriculumConfig
        Must have `config.stage4_input_image_dir` set by the caller
        (mirrors how `stage2_train` sets `config.stage2_train_image_dir`
        before building its dataloaders).
    logger : logging.Logger, optional

    Returns
    -------
    torch.utils.data.DataLoader
    """
    dataset = _PseudoLabelInferenceDataset(image_dir=config.stage4_input_image_dir)

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    if logger is not None:
        logger.info(f"Input Directory     : {config.stage4_input_image_dir}")
        logger.info(f"Images Found        : {len(dataset)}")
        logger.info(f"Batch Size          : {config.batch_size}")
        logger.info("")

    return dataloader


# ------------------------------------------------------------
# Result packaging
# ------------------------------------------------------------

def _create_generation_result(
    config,
    iteration,
    checkpoint_path,
    output_directory,
    number_of_images,
    number_of_labels_generated,
    generation_time,
    runtime_seconds,
    success,
    logger=None,
):
    """
    Package the outcome of a Stage-4 run into a
    `PseudoLabelGenerationResult`.

    Parameters
    ----------
    config : CurriculumConfig
    iteration : int
    checkpoint_path : str
    output_directory : pathlib.Path or str
    number_of_images : int
    number_of_labels_generated : int
    generation_time : str
        ISO-8601 timestamp marking when generation completed.
    runtime_seconds : float
    success : bool
    logger : logging.Logger, optional

    Returns
    -------
    PseudoLabelGenerationResult
    """
    result = PseudoLabelGenerationResult(
        dataset=config.dataset,
        iteration=iteration,
        checkpoint_path=str(checkpoint_path),
        output_directory=str(output_directory),
        number_of_images=number_of_images,
        number_of_labels_generated=number_of_labels_generated,
        generation_time=generation_time,
        runtime_seconds=runtime_seconds,
        success=success,
    )

    if logger is not None:
        logger.info(f"Images Processed    : {result.number_of_images}")
        logger.info(f"Pseudo Labels Generated : {result.number_of_labels_generated}")
        logger.info(f"Runtime             : {result.runtime_seconds:.2f}s")
        logger.info(f"Output Directory    : {result.output_directory}")
        logger.info(f"Checkpoint Used     : {result.checkpoint_path}")
        logger.info("")

    return result