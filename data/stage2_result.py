"""
stage2_result.py

Stage-2 Training Result Container

A small data container (not a behavioral class - no methods beyond
what @dataclass generates) used to package everything the curriculum
controller needs after one Stage-2 training run.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Stage2TrainingResult:
    """
    Container for the outputs of a single Stage-2 training run.

    Attributes
    ----------
    iteration : int
        Curriculum iteration index this result belongs to.
    dataset : str
        Dataset name (e.g. 'bcss').
    network : str
        Backbone name used by the DeepLab model (e.g. 'resnet').
    epochs : int
        Number of Stage-2 epochs trained this iteration.
    best_miou : float
        Best validation mIoU achieved during training, matching the
        baseline `Trainer.best_pred` value.
    loss_history : list of float
        Per-epoch total training loss (sum over batches), one entry
        per epoch, in the same units as the baseline's `train_loss`.
    checkpoint_path : str
        Path to the best-saved Stage-2 checkpoint
        (`stage2_checkpoint_trained_on_v2_<dataset>.pth`, matching
        the baseline's `Saver.save_checkpoint` filename).
    test_results : dict
        Metrics dict returned by `_test_model`
        (Acc, Acc_class, mIoU, FWIoU, ious, inference_time).
    runtime_seconds : float
        Wall-clock time for the full Stage-2 run (dataloader/model
        build through final testing).
    """

    iteration: int
    dataset: str
    network: str
    epochs: int
    best_miou: float
    loss_history: List[float] = field(default_factory=list)
    checkpoint_path: str = ""
    test_results: Dict[str, Any] = field(default_factory=dict)
    runtime_seconds: float = 0.0