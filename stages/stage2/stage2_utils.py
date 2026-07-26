from tool.GenDataset import Stage2DatasetV2, make_data_loader_v2

from torch.utils.data import DataLoader

from pathlib import Path


def _build_stage2_dataloader(
    config,
    logger=None,
):
    """
    Build Stage-2 dataloaders.

    Parameters
    ----------
    config : CurriculumConfig

    logger : logging.Logger, optional

    Returns
    -------
    dict
        Dictionary containing train, validation and test dataloaders.
    """

    if logger is not None:
        logger.info("=" * 80)
        logger.info("Building Stage-2 Dataloaders")
        logger.info("=" * 80)
        logger.info(f"Dataset Root : {config.dataroot}")
        logger.info(f"Batch Size   : {config.batch_size}")
        logger.info(f"Workers      : {config.num_workers}")
        logger.info("")

    # ------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------

    train_dataset = Stage2DatasetV2(
        config=config,
        split="train",
    )

    validation_dataset = Stage2DatasetV2(
        config=config,
        split="val",
    )

    test_dataset = Stage2DatasetV2(
        config=config,
        split="test",
    )

    # ------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    if logger is not None:

        logger.info("Dataset Summary")
        logger.info("-" * 80)
        logger.info(f"Training Images    : {len(train_dataset)}")
        logger.info(f"Validation Images  : {len(validation_dataset)}")
        logger.info(f"Testing Images     : {len(test_dataset)}")
        logger.info("")
        logger.info(f"Training Batches   : {len(train_loader)}")
        logger.info(f"Validation Batches : {len(validation_loader)}")
        logger.info(f"Testing Batches    : {len(test_loader)}")
        logger.info("=" * 80)
        logger.info("Stage-2 dataloaders created successfully.")
        logger.info("")

    return {
        "train": train_loader,
        "validation": validation_loader,
        "test": test_loader,
    }




"""
stage2_helpers.py

Stage-2 Semantic Segmentation Training - Helper Functions

Each function below has exactly one responsibility and mirrors a
specific piece of logic from the original monolithic `Trainer` class
baseline implementation. Refactoring here is purely architectural -
every numerical computation (loss weighting, the class-4 ignore
trick, checkpoint stripping rules, CAM-gating, metric definitions)
is reproduced exactly as the baseline computed it.

Two baseline quirks are intentionally preserved rather than "fixed",
per the instruction to preserve behavioral equivalence:

1. The checkpoint filename mismatch: the best checkpoint is *saved*
   as ``stage2_checkpoint_trained_on_v2_<dataset>.pth`` (see
   `_save_checkpoint`) but *reloaded* from
   ``stage2_checkpoint_trained_on_<dataset>.pth`` (no ``_v2_``) in
   both `_load_stage2_checkpoint` and `_test_model`. This matches the
   baseline's `Trainer.load_the_best_checkpoint()` exactly.
2. `_validate_one_epoch` never accumulates a real validation loss -
   the baseline initializes `test_loss = 0.0` and logs/reports it
   unchanged. No validation loss is invented here.
"""

import os
import time
import importlib

import numpy as np
import torch
from tqdm import tqdm

from tool.GenDataset import make_data_loader
from network.sync_batchnorm.replicate import patch_replication_callback
from network.deeplab import DeepLab
from tool.loss import SegmentationLosses
from tool.lr_scheduler import LR_Scheduler
from tool.saver import Saver
from tool.metrics import Evaluator

from data.stage2_result import Stage2TrainingResult


# ==================================================================
# Step 2: Dataloaders
# ==================================================================

def _build_stage2_dataloader(config, logger):
    """
    Build the Stage-2 train / validation / test dataloaders.

    Only constructs datasets and dataloaders via `make_data_loader`.
    Never builds transforms beyond what `make_data_loader` already
    does in the baseline, and never touches the model, optimizer, or
    scheduler.

    Parameters
    ----------
    config : CurriculumConfig
        Must expose `workers` plus all dataset-directory attributes
        consumed by `make_data_loader` (mirroring the baseline's
        `args` namespace): `dataroot`, `stage2_train_image_dir`,
        `stage2_train_label_dir`, `stage2_train_label_dir_a`,
        `stage2_train_label_dir_b`, `stage2_val_image_dir`,
        `stage2_val_label_dir`, `stage2_test_image_dir`,
        `stage2_test_label_dir`.
    logger : logging.Logger

    Returns
    -------
    dict
        {'train': train_loader, 'val': val_loader, 'test': test_loader}
    """
    logger.info("=" * 72)
    logger.info("Building Stage-2 Dataloaders")
    logger.info("=" * 72)
    logger.info(f"{'Data root':<25}: {config.dataroot}")
    logger.info(f"{'Train image dir':<25}: {config.stage2_train_image_dir}")
    logger.info(f"{'Train label dir':<25}: {config.stage2_train_label_dir}")
    logger.info(f"{'Train label dir A':<25}: {config.stage2_train_label_dir_a}")
    logger.info(f"{'Train label dir B':<25}: {config.stage2_train_label_dir_b}")
    logger.info(f"{'Val image dir':<25}: {config.stage2_val_image_dir}")
    logger.info(f"{'Test image dir':<25}: {config.stage2_test_image_dir}")
    logger.info(f"{'Workers':<25}: {config.num_workers}")

    start = time.time()

    # Matches baseline: kwargs = {'num_workers': args.workers, 'pin_memory': False}
    kwargs = {"num_workers": config.num_workers, "pin_memory": False}
    train_loader, val_loader, test_loader = make_data_loader_v2(config, **kwargs)

    elapsed = time.time() - start

    logger.info("")
    logger.info(f"{'Train batches':<25}: {len(train_loader)}")
    logger.info(f"{'Validation batches':<25}: {len(val_loader)}")
    logger.info(f"{'Test batches':<25}: {len(test_loader)}")
    logger.info(f"Dataloaders built in {elapsed:.2f} sec")
    logger.info("")

    return {"train": train_loader, "val": val_loader, "test": test_loader}


# ==================================================================
# Step 3: Model
# ==================================================================

def _build_stage2_model(config, logger):
    """
    Construct the Stage-2 DeepLab segmentation model.

    Builds DeepLab with the configured backbone / output stride /
    sync-BN / freeze-BN settings, resolves the compute device exactly
    as the baseline does (CUDA if available and requested, else CPU),
    and wraps with DataParallel + the sync-BN replication patch when
    running on CUDA - matching the baseline exactly. Does not load
    checkpoints, and does not build the optimizer or scheduler.

    Parameters
    ----------
    config : CurriculumConfig
        Must expose `backbone`, `out_stride`, `sync_bn`, `freeze_bn`,
        `n_class`, `cuda`, `gpu_ids`. This function sets
        `config.device` for downstream helpers to reuse.
    logger : logging.Logger

    Returns
    -------
    torch.nn.Module
        The (possibly DataParallel-wrapped) DeepLab model, already
        moved to the resolved device.
    """
    logger.info("=" * 72)
    logger.info("Building Stage-2 Model")
    logger.info("=" * 72)
    logger.info(f"{'Backbone':<20}: {config.backbone}")
    logger.info(f"{'Output stride':<20}: {config.stage2_out_stride}")
    logger.info(f"{'Sync BN':<20}: {config.stage2_sync_bn}")
    logger.info(f"{'Freeze BN':<20}: {config.stage2_freeze_bn}")
    logger.info(f"{'Classes':<20}: {config.n_class}")
    logger.info(f"{'CUDA requested':<20}: {config.cuda}")

    # Matches baseline: torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() and config.cuda else "cpu")
    config.device = device

    model = DeepLab(
        num_classes=config.n_class,
        backbone=config.backbone,
        output_stride=config.stage2_out_stride,
        sync_bn=config.stage2_sync_bn,
        freeze_bn=config.stage2_freeze_bn,
    )

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if config.cuda:
        model = torch.nn.DataParallel(model, device_ids=config.stage2_gpu_ids)
        patch_replication_callback(model)
        model = model.cuda()
    else:
        model = model.to(device)

    logger.info("")
    logger.info(f"{'Total parameters':<20}: {num_params:,}")
    logger.info(f"{'Trainable parameters':<20}: {num_trainable:,}")
    logger.info(f"{'Resolved device':<20}: {device}")
    logger.info("Model created successfully.")
    logger.info("")

    return model


# ==================================================================
# Step 4: Optimizer
# ==================================================================

def _build_optimizer(config, model, logger):
    """
    Build the Stage-2 SGD optimizer with the baseline's two
    learning-rate parameter groups (1x for backbone params, 10x for
    the newly-added segmentation head params).

    Reads parameter groups from the underlying (unwrapped) DeepLab
    module when `model` is DataParallel-wrapped - DataParallel does
    not copy parameters, so this yields the exact same tensors the
    baseline's optimizer would have referenced.

    Parameters
    ----------
    config : CurriculumConfig
        Must expose `lr`, `momentum`, `weight_decay`, `nesterov`.
    model : torch.nn.Module
        Model returned by `_build_stage2_model`.
    logger : logging.Logger

    Returns
    -------
    torch.optim.SGD
    """
    logger.info("=" * 72)
    logger.info("Building Optimizer")
    logger.info("=" * 72)

    base_model = model.module if hasattr(model, "module") else model

    # Matches baseline:
    # train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
    #                 {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]
    train_params = [
        {"params": base_model.get_1x_lr_params(), "lr": config.stage2_lr},
        {"params": base_model.get_10x_lr_params(), "lr": config.stage2_lr * 10},
    ]

    optimizer = torch.optim.SGD(
        train_params,
        momentum=config.stage2_momentum,
        weight_decay=config.stage2_weight_decay,
        nesterov=config.stage2_nesterov,
    )

    logger.info(f"{'Base LR (1x)':<20}: {config.stage2_lr}")
    logger.info(f"{'Head LR (10x)':<20}: {config.stage2_lr * 10}")
    logger.info(f"{'Momentum':<20}: {config.stage2_momentum}")
    logger.info(f"{'Weight decay':<20}: {config.stage2_weight_decay}")
    logger.info(f"{'Nesterov':<20}: {config.stage2_nesterov}")
    logger.info("Optimizer initialized.")
    logger.info("")

    return optimizer


# ==================================================================
# Step 5: Scheduler
# ==================================================================

def _build_scheduler(config, optimizer, logger):
    """
    Build the Stage-2 learning-rate scheduler.

    Mirrors the baseline's
    ``LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(train_loader))``
    call exactly.

    Parameters
    ----------
    config : CurriculumConfig
        Must expose `lr_scheduler`, `lr`, `stage2_epochs`, and
        `num_train_iterations` (the caller must set this to
        ``len(train_loader)`` after building the dataloaders, since
        the baseline scheduler needs iterations-per-epoch at
        construction time).
    optimizer : torch.optim.Optimizer
        Not read at construction time (the baseline scheduler takes
        the optimizer as an argument to its per-step `__call__`, not
        to `__init__`); kept as a parameter for interface symmetry.
    logger : logging.Logger

    Returns
    -------
    LR_Scheduler
    """
    logger.info("=" * 72)
    logger.info("Building Learning-Rate Scheduler")
    logger.info("=" * 72)
    logger.info(f"{'Scheduler type':<20}: {config.stage2_lr_scheduler}")
    logger.info(f"{'Base LR':<20}: {config.stage2_lr}")
    logger.info(f"{'Epochs':<20}: {config.stage2_epochs}")
    logger.info(f"{'Iterations/epoch':<20}: {config.num_train_iterations}")

    scheduler = LR_Scheduler(
        config.stage2_lr_scheduler,
        config.stage2_lr,
        config.stage2_epochs,
        config.num_train_iterations,
    )

    logger.info("Scheduler initialized.")
    logger.info("")

    return scheduler


# ==================================================================
# Step 6: Loss function
# ==================================================================

def _build_loss_function(config, logger):
    """
    Build the Stage-2 segmentation loss exactly as the baseline.

    Parameters
    ----------
    config : CurriculumConfig
        Must expose `loss_type` and `device` (set by
        `_build_stage2_model`).
    logger : logging.Logger

    Returns
    -------
    torch.nn.Module
    """
    logger.info("=" * 72)
    logger.info("Building Loss Function")
    logger.info("=" * 72)
    logger.info(f"{'Loss type':<20}: {config.stage2_loss_type}")

    # Matches the baseline's CPU-fallback patch:
    # self.criterion = SegmentationLosses(weight=None, cuda=self.device.type == "cuda").build_loss(mode=args.loss_type)
    criterion = SegmentationLosses(
        weight=None, cuda=config.device.type == "cuda"
    ).build_loss(mode=config.stage2_loss_type)

    logger.info("Loss function initialized.")
    logger.info("")

    return criterion


# ==================================================================
# Step 7: Evaluator
# ==================================================================

def _build_evaluator(config, logger):
    """
    Build the pixel-accuracy / mIoU evaluator.

    Parameters
    ----------
    config : CurriculumConfig
        Must expose `n_class`.
    logger : logging.Logger

    Returns
    -------
    Evaluator
    """
    logger.info("=" * 72)
    logger.info("Building Evaluator")
    logger.info("=" * 72)
    logger.info(f"{'Classes':<20}: {config.n_class}")

    evaluator = Evaluator(config.n_class)

    logger.info("Evaluator initialized.")
    logger.info("")

    return evaluator


# ==================================================================
# Step 8: Stage-1 classifier (used only for CAM-gating in testing)
# ==================================================================

def _load_stage1_classifier(config, logger):
    """
    Construct the Stage-1 ResNet38 classifier and load its checkpoint.

    This model is idle during Stage-2 training and validation - it is
    only used at test time, when `config.Is_GM` gates segmentation
    predictions with the classifier's image-level CAM output.

    Parameters
    ----------
    config : CurriculumConfig
        Must expose `dataset` and `device`.
    logger : logging.Logger

    Returns
    -------
    torch.nn.Module
        The Stage-1 classifier, in eval mode, on `config.device`.
    """
    logger.info("=" * 72)
    logger.info("Loading Stage-1 Classifier")
    logger.info("=" * 72)

    device = config.device

    # Matches baseline: importlib.import_module('network.resnet38_cls').Net_CAM(n_class=4)
    stage1_module = importlib.import_module("network.resnet38_cls")
    stage1_model = getattr(stage1_module, "Net_CAM")(n_class=config.n_class)

    # resume_stage1 = os.path.join(
    #     "checkpoints", f"stage1_checkpoint_trained_on_{config.dataset}.pth"
    # )
    resume_stage1 = config.stage1_checkpoint
    logger.info(f"{'Checkpoint':<20}: {resume_stage1}")

    checkpoint  = torch.load(resume_stage1, map_location=device)

    # unwrap it here rather than passing the whole dict to load_state_dict.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            weights_dict = checkpoint["model_state_dict"]
    
            logger.info(f"{'Network':<20}: {checkpoint.get('network')}")
            logger.info(f"{'Trained on dataset':<20}: {checkpoint.get('dataset')}")
            logger.info(f"{'Best epoch':<20}: {checkpoint.get('best_epoch')}")
            logger.info(f"{'Best accuracy':<20}: {checkpoint.get('best_accuracy')}")
    else:
        # Fall back to treating it as a raw state_dict, in case an
        # older/plain checkpoint is ever passed in.
        weights_dict = checkpoint

    stage1_model.load_state_dict(weights_dict)

    stage1_model = stage1_model.to(device)
    stage1_model.eval()

    logger.info("Stage-1 classifier loaded successfully.")
    logger.info("")

    return stage1_model


# ==================================================================
# Step 9: Stage-2 weight loading (curriculum-dependent)
# ==================================================================

def _load_pretrained_weights(config, model, logger, optimizer=None):
    """
    Load ImageNet/DeepLab pretrained weights into the Stage-2 model.

    Used for curriculum iteration 0 only. Reproduces the baseline's
    `args.resume` checkpoint-loading branch exactly, including
    stripping the final segmentation-head layer
    (`decoder.last_conv.8.{weight,bias}`) unless fine-tuning, and
    stripping `module.` prefixes when loading a DataParallel-saved
    checkpoint onto a non-DataParallel (CPU) model.

    Parameters
    ----------
    config : CurriculumConfig
        Must expose `resume`, `ft`, `cuda`, `device`.
    model : torch.nn.Module
        Model returned by `_build_stage2_model`; updated in place.
    logger : logging.Logger
    optimizer : torch.optim.Optimizer, optional
        When `config.ft` is True, the baseline also restores
        optimizer state from the checkpoint - pass the optimizer to
        reproduce that.

    Returns
    -------
    None
    """
    logger.info("=" * 72)
    logger.info("Loading Pretrained Segmentation Weights (Iteration 0)")
    logger.info("=" * 72)

    if config.stage2_resume is None:
        logger.info("No `resume` path configured - skipping pretrained load.")
        logger.info("")
        return

    if not os.path.isfile(config.stage2_resume):
        raise RuntimeError(f"=> no checkpoint found at '{config.stage2_resume}'")

    logger.info(f"{'Checkpoint':<20}: {config.stage2_resume}")
    logger.info(f"{'Fine-tune mode':<20}: {config.stage2_ft}")

    checkpoint = torch.load(config.stage2_resume, map_location=config.device, weights_only=False)
    state_dict = checkpoint["state_dict"]

    if config.cuda:
        if not config.stage2_ft:
            del state_dict["decoder.last_conv.8.weight"]
            del state_dict["decoder.last_conv.8.bias"]
        model.module.load_state_dict(state_dict, strict=False)
    else:
        if not config.stage2_ft:
            # Clean keys if the saved checkpoint used DataParallel but we're on CPU.
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            if "decoder.last_conv.8.weight" in state_dict:
                del state_dict["decoder.last_conv.8.weight"]
            if "decoder.last_conv.8.bias" in state_dict:
                del state_dict["decoder.last_conv.8.bias"]
        model.load_state_dict(state_dict, strict=False)

    if config.stage2_ft and optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        logger.info("Optimizer state restored (fine-tuning mode).")

    logger.info("Pretrained weights loaded successfully.")
    logger.info("")


def _load_stage2_checkpoint(config, model, logger):
    """
    Load the previous curriculum iteration's best Stage-2 checkpoint.

    Reproduces the baseline `Trainer.load_the_best_checkpoint()`
    method exactly, including its checkpoint filename - which does
    NOT include the `_v2_` suffix used when *saving* checkpoints (see
    `_save_checkpoint`). This mismatch exists in the baseline and is
    preserved here rather than corrected, per the instruction to keep
    behavior identical to the source implementation.

    Parameters
    ----------
    config : CurriculumConfig
        Must expose `dataset` and `device`.
    model : torch.nn.Module
        Updated in place.
    logger : logging.Logger

    Returns
    -------
    None
    """
    logger.info("=" * 72)
    logger.info("Loading Previous Stage-2 Checkpoint")
    logger.info("=" * 72)

    # NOTE: intentionally no "_v2_" here - matches baseline's
    # `load_the_best_checkpoint()`, which reloads a differently-named
    # file than `validation()` saves to.
    # checkpoint_path = os.path.join(
    #     "checkpoints", f"stage2_checkpoint_trained_on_{config.dataset}.pth"
    # )
    checkpoint_path = config.stage2_checkpoint
    logger.info(f"{'Checkpoint':<20}: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=config.device)

    if hasattr(model, "module"):
        model.module.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        state_dict = checkpoint["state_dict"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)

    logger.info("Previous Stage-2 checkpoint loaded successfully.")
    logger.info("")


# ==================================================================
# Step 10: Training / validation loop
# ==================================================================

def _train_one_epoch(
    config,
    epoch,
    model,
    stage1_model,
    dataloaders,
    optimizer,
    scheduler,
    criterion,
    evaluator,
    logger,
    state,
):
    """
    Run one Stage-2 training epoch.

    Reproduces `Trainer.training()` exactly: three pseudo-mask
    targets (`label`, `label_a`, `label_b`) are combined into a
    single weighted loss (0.6 / 0.2 / 0.2), and predictions for the
    ignored class (index 4) are forced to a very high logit via a
    synthetic extra output channel before computing the loss, so that
    class trivially wins wherever it's the ground truth.

    Parameters
    ----------
    config : CurriculumConfig
        Must expose `device`, `batch_size`.
    epoch : int
    model : torch.nn.Module
    stage1_model : torch.nn.Module
        Unused during training - kept for interface symmetry with the
        baseline, where the Stage-1 classifier is idle until test
        time.
    dataloaders : dict
        As returned by `_build_stage2_dataloader`.
    optimizer : torch.optim.Optimizer
    scheduler : LR_Scheduler
    criterion : torch.nn.Module
    evaluator : Evaluator
        Unused here (only used in validation/test); kept for
        interface symmetry.
    logger : logging.Logger
    state : dict
        Mutable dict carrying `best_pred` (read by the scheduler at
        every step, matching
        ``self.scheduler(self.optimizer, i, epoch, self.best_pred)``)
        and accumulating `loss_history` across epochs.

    Returns
    -------
    float
        Total training loss for the epoch (sum over batches, matching
        the baseline's `train_loss` accumulation).
    """
    device = config.device
    train_loader = dataloaders["train"]

    model.train()
    train_loss = 0.0
    tbar = tqdm(train_loader)
    num_img_tr = len(train_loader)

    image = None  # populated in the loop; referenced afterwards, as in baseline
    for i, sample in enumerate(tbar):
        image = sample["image"].to(device)
        target = sample["label"].to(device)
        target_a = sample["label_a"].to(device)
        target_b = sample["label_b"].to(device)

        scheduler(optimizer, i, epoch, state["best_pred"])
        optimizer.zero_grad()

        output = model(image)

        one = torch.ones((output.shape[0], 1, 224, 224), device=device)
        output = torch.cat(
            [output, (100 * one * (target == 4).unsqueeze(dim=1))], dim=1
        )

        loss_o = criterion(output, target)
        loss_a = criterion(output, target_a)
        loss_b = criterion(output, target_b)
        loss = 0.6 * loss_o + 0.2 * loss_a + 0.2 * loss_b

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        tbar.set_description("Train loss: %.3f" % (train_loss / (i + 1)))

    logger.info(
        f"[Epoch {epoch}] numImages: {i * config.batch_size + image.shape[0]}"
    )
    logger.info(f"{'Training Loss':<20}: {train_loss:.3f}")

    state.setdefault("loss_history", []).append(train_loss)

    return train_loss


def _validate_one_epoch(config, epoch, model, dataloaders, criterion, evaluator, logger, state):
    """
    Run one Stage-2 validation epoch, saving a new best checkpoint
    when mIoU improves.

    Reproduces `Trainer.validation()` exactly: predictions for the
    ignored class (index 4) are forced to match the ground truth
    before computing metrics, excluding that class from evaluation.
    As in the baseline, no real validation loss is computed (the
    baseline's `test_loss` stays `0.0` throughout).

    Parameters
    ----------
    config : CurriculumConfig
        Must expose `device`, `batch_size`.
    epoch : int
    model : torch.nn.Module
    dataloaders : dict
    criterion : torch.nn.Module
        Unused for metric computation - kept for interface symmetry
        with the baseline, which builds but never uses it inside
        `validation()`.
    evaluator : Evaluator
    logger : logging.Logger
    state : dict
        Mutable dict carrying/updated with `best_pred` and
        `optimizer` (needed by `_save_checkpoint`).

    Returns
    -------
    dict
        {'Acc', 'Acc_class', 'mIoU', 'FWIoU', 'ious'}
    """
    device = config.device
    val_loader = dataloaders["val"]

    model.eval()
    evaluator.reset()
    tbar = tqdm(val_loader, desc="\r")

    image = None
    for i, sample in enumerate(tbar):
        image = sample[0]["image"].to(device)
        target = sample[0]["label"].to(device)

        with torch.no_grad():
            output = model(image)

        pred = output.data.cpu().numpy()
        target_np = target.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        pred[target_np == 4] = 4  # class 4 is excluded from evaluation
        evaluator.add_batch(target_np, pred)

    acc = evaluator.Pixel_Accuracy()
    acc_class = evaluator.Pixel_Accuracy_Class()
    miou = evaluator.Mean_Intersection_over_Union()
    ious = evaluator.Intersection_over_Union()
    fwiou = evaluator.Frequency_Weighted_Intersection_over_Union()

    logger.info(f"[Validation][Epoch {epoch}] numImages: {i * config.batch_size + image.shape[0]}")
    logger.info(f"{'Pixel Accuracy':<20}: {acc:.4f}")
    logger.info(f"{'Mean Accuracy':<20}: {acc_class:.4f}")
    logger.info(f"{'Mean IoU':<20}: {miou:.4f}")
    logger.info(f"{'FWIoU':<20}: {fwiou:.4f}")
    logger.info(f"{'Per-class IoU':<20}: {ious}")

    if miou > state["best_pred"]:
        logger.info("")
        logger.info("Saving best checkpoint...")
        logger.info(f"{'Previous Best mIoU':<20}: {state['best_pred']:.4f}")
        logger.info(f"{'Current mIoU':<20}: {miou:.4f}")

        state["best_pred"] = miou
        _save_checkpoint(config, model, state, logger)

    logger.info("")

    return {
        "Acc": acc,
        "Acc_class": acc_class,
        "mIoU": miou,
        "FWIoU": fwiou,
        "ious": ious,
    }


def _save_checkpoint(config, model, state, logger):
    """
    Persist the current best Stage-2 model + optimizer state to disk.

    Reproduces the baseline's `Saver.save_checkpoint()` call inside
    `Trainer.validation()`, including the exact filename convention
    ``stage2_checkpoint_trained_on_v2_<dataset>.pth``.

    Parameters
    ----------
    config : CurriculumConfig
        Must expose `dataset` plus whatever `Saver` itself requires
        (matching the baseline's `Saver(args)` usage).
    model : torch.nn.Module
    state : dict
        Must expose `optimizer` so its state can be included in the
        checkpoint, matching the baseline.
    logger : logging.Logger

    Returns
    -------
    str
        The filename the checkpoint was saved under.
    """

    checkpoint_directory = Path(config.save_folder) / "stage2"
    checkpoint_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = checkpoint_directory / "stage2_best.pth"

    saver = Saver(config)

    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

    saver.save_checkpoint_v2(
        {
            "state_dict": state_dict,
            "optimizer": state["optimizer"].state_dict(),
        },
        checkpoint_path,
    )

    logger.info(f"Checkpoint saved to: {checkpoint_path}")
    logger.info("")

    return checkpoint_path

# ==================================================================
# Step 11: Testing
# ==================================================================

def _test_model(config, model, dataloaders, evaluator, logger, stage1_model=None):
    """
    Evaluate the best Stage-2 checkpoint on the held-out test set.

    Reproduces `Trainer.test()` exactly, including the optional
    CAM-gating mechanism (`config.Is_GM`): when enabled, the Stage-1
    classifier's image-level predictions (thresholded at 0.1) mask
    out segmentation classes that weren't detected at the image
    level, before the argmax is taken.

    As in the baseline's `test()`, this reloads the best checkpoint
    via the same (mismatched-filename) path used by
    `_load_stage2_checkpoint` before running inference.

    Parameters
    ----------
    config : CurriculumConfig
        Must expose `Is_GM`, `dataset`, `device`, `batch_size`.
    model : torch.nn.Module
    dataloaders : dict
    evaluator : Evaluator
    logger : logging.Logger
    stage1_model : torch.nn.Module, optional
        Required when `config.Is_GM` is True.

    Returns
    -------
    dict
        {'Acc', 'Acc_class', 'mIoU', 'FWIoU', 'ious', 'inference_time'}
    """
    device = config.device
    test_loader = dataloaders["test"]

    # Baseline always reloads the best-saved checkpoint before testing.
    _load_stage2_checkpoint(config, model, logger)

    model.eval()
    evaluator.reset()
    tbar = tqdm(test_loader, desc="\r")

    start = time.time()
    image = None
    for i, sample in enumerate(tbar):
        image = sample[0]["image"].to(device)
        target = sample[0]["label"].to(device)

        with torch.no_grad():
            output = model(image)
            if config.stage2_is_gm:
                _, y_cls = stage1_model.forward_cam(image)
                y_cls = y_cls.cpu().data
                pred_cls = y_cls > 0.1

        pred = output.data.cpu().numpy()
        if config.stage2_is_gm:
            pred = pred * pred_cls.unsqueeze(dim=2).unsqueeze(dim=3).numpy()

        target_np = target.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        pred[target_np == 4] = 4
        evaluator.add_batch(target_np, pred)

    inference_time = time.time() - start

    acc = evaluator.Pixel_Accuracy()
    acc_class = evaluator.Pixel_Accuracy_Class()
    miou = evaluator.Mean_Intersection_over_Union()
    ious = evaluator.Intersection_over_Union()
    fwiou = evaluator.Frequency_Weighted_Intersection_over_Union()

    logger.info(f"[Test] numImages: {i * config.batch_size + image.shape[0]}")
    logger.info("Testing Summary")
    logger.info(f"{'Pixel Accuracy':<20}: {acc:.4f}")
    logger.info(f"{'Mean IoU':<20}: {miou:.4f}")
    logger.info(f"{'FWIoU':<20}: {fwiou:.4f}")
    logger.info(f"{'Inference Time':<20}: {inference_time:.2f} sec")
    logger.info(f"{'Per-class IoU':<20}: {ious}")

    return {
        "Acc": acc,
        "Acc_class": acc_class,
        "mIoU": miou,
        "FWIoU": fwiou,
        "ious": ious,
        "inference_time": inference_time,
    }


# ==================================================================
# Step 12: Package results
# ==================================================================

def _create_stage2_result(config, iteration, model, test_results, runtime_seconds, logger, state):
    """
    Package the outcome of a Stage-2 training run into a
    `Stage2TrainingResult`.

    Parameters
    ----------
    config : CurriculumConfig
        Must expose `dataset`, `backbone`, `stage2_epochs`.
    iteration : int
    model : torch.nn.Module
        Not read directly - kept for interface symmetry / potential
        future use (e.g. reporting parameter counts alongside
        results).
    test_results : dict
        As returned by `_test_model`.
    runtime_seconds : float
    logger : logging.Logger
    state : dict
        Must expose `best_pred` and `loss_history`.

    Returns
    -------
    Stage2TrainingResult
    """
    # NOTE: matches the (mismatched) save-time filename; see
    # `_save_checkpoint` / `_load_stage2_checkpoint` docstrings.
    checkpoint_path = os.path.join(
        "checkpoints", f"stage2_checkpoint_trained_on_v2_{config.dataset}.pth"
    )

    result = Stage2TrainingResult(
        iteration=iteration,
        dataset=config.dataset,
        network=config.backbone,
        epochs=config.stage2_epochs,
        best_miou=state["best_pred"],
        loss_history=state.get("loss_history", []),
        checkpoint_path=checkpoint_path,
        test_results=test_results,
        runtime_seconds=runtime_seconds,
    )

    logger.info("Stage-2 Result Summary")
    logger.info(f"{'Best mIoU':<20}: {result.best_miou:.4f}")
    logger.info(f"{'Checkpoint':<20}: {result.checkpoint_path}")
    logger.info("")

    return result