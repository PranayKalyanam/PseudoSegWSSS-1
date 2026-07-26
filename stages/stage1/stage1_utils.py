from datetime import datetime
import logging
import os
from pathlib import Path

import torch

from data.stage1_results import Stage1Result
from data.stage1_test_result import Stage1TestResult
from tool import torchutils

logger = logging.getLogger(__name__)


def _build_optimizer(
    config,
    model,
    train_loader,
):

    max_steps = len(train_loader) * config.stage1_epochs
    #   max_step = (len(train_dataset) // args.batch_size) * args.max_epoches

    param_groups  = model.get_parameter_groups()

    optimizer = torchutils.PolyOptimizer(
        [
            
            {"params": param_groups[0], "lr": config.stage1_lr, "weight_decay": config.stage1_wt_dec, },
            {"params": param_groups[1], "lr": 2 * config.stage1_lr, "weight_decay": 0.0, },
            { "params": param_groups[2], "lr": 10 * config.stage1_lr, "weight_decay": config.stage1_wt_dec, },
            { "params": param_groups[3], "lr": 20 * config.stage1_lr, "weight_decay": 0.0, },
        ],
        lr=config.stage1_lr,
        weight_decay=config.stage1_wt_dec,
        max_step=max_steps,
        )

    return optimizer


def _load_pretrained_weights(
    config,
    model,
    logger=None,
):
    """
    Initialize the classifier weights.

    Responsibilities
    ----------------
    1. Load pretrained weights if provided.
    2. Otherwise use random initialization.

    Parameters
    ----------
    config : Stage1Config
        Stage-1 configuration.

    model : torch.nn.Module
        Classification model.

    Returns
    -------
    None
    """

    # ------------------------------------------------------------
    # No pretrained weights
    # ------------------------------------------------------------

    if config.stage1_weights is None:

        logger.info("No pretrained weights specified. Using random initialization.")

        return

    if not os.path.isfile(config.stage1_weights):

        logger.info(f"Weight file not found: {config.stage1_weights}")
        logger.info("Using random initialization.")

        return

    # ------------------------------------------------------------
    # Load PyTorch checkpoint
    # ------------------------------------------------------------

    if config.stage1_weights.endswith(".pth"):

        logger.info(f"Loading pretrained weights from:\n{config.stage1_weights}")

        weights = torch.load(
            config.stage1_weights,
            map_location="cpu",
            weights_only=False,
        )

        model.load_state_dict(
            weights,
            strict=False,
        )

        logger.info("Pretrained weights loaded successfully.")

        return

    # ------------------------------------------------------------
    # Unsupported format
    # ------------------------------------------------------------

    raise ValueError(
        f"Unsupported weight format: {config.stage1_weights}"
    )



def _load_stage1_checkpoint(
    config,
    model,
    logger=None,
):
    """
    Load the best Stage-1 checkpoint for continued curriculum training.

    This function is used at the beginning of Stage-1 training for
    curriculum iterations (> 0). If no checkpoint exists, the model
    remains initialized with its current pretrained weights.

    Parameters
    ----------
    config : CurriculumConfig
        Experiment configuration.

    model : torch.nn.Module
        Stage-1 classification model.

    logger : logging.Logger, optional
        Logger used to record loading information.

    Returns
    -------
    bool
        True if a checkpoint was loaded, otherwise False.
    """

    # ------------------------------------------------------------
    # Resolve checkpoint path
    # ------------------------------------------------------------

    checkpoint_path = Path(config.stage1_checkpoint)

    if not checkpoint_path.exists():

        if logger is not None:
            logger.info(
                "No previous Stage-1 checkpoint found. "
                "Training will start from pretrained weights."
            )

        return False

    # ------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint,
    )

    model.load_state_dict(
        state_dict,
        strict=False,
    )

    if logger is not None:
        logger.info(
            f"Loaded Stage-1 checkpoint: {checkpoint_path}"
        )

    return True


def _load_checkpoint(
    config,
    model,
    logger,
):
    """
    Load the trained Stage-1 checkpoint.

    Parameters
    ----------
    config : CurriculumConfig

    iteration_manager : IterationContext

    model : torch.nn.Module

    Returns
    -------
    pathlib.Path
        Path of the loaded checkpoint.
    """
    


    # ------------------------------------------------------------
    # Checkpoint Path
    # ------------------------------------------------------------

    checkpoint_path = Path(config.stage1_checkpoint)    
    
    logger.info("=" * 80)
    logger.info("Loading Stage-1 Checkpoint")
    logger.info("=" * 80)
    logger.info(f"Checkpoint : {checkpoint_path}")


    if not checkpoint_path.exists():
        logger.error("Checkpoint file not found.")

        raise FileNotFoundError(
            f"Stage-1 checkpoint not found:\n{checkpoint_path}"
        )

    # ------------------------------------------------------------
    # Load Checkpoint
    # ------------------------------------------------------------

    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        state_dict,
        strict=False,
    )

    model.eval()
    
    logger.info("Checkpoint loaded successfully.")
    logger.info("=" * 80)

    return checkpoint_path



    
def _save_checkpoint(
    config,
    training_history,
    logger=None,
):
    """
    Save the trained Stage-1 classifier checkpoint.

    Parameters
    ----------
    config : Stage1Config
        Stage-1 configuration.

    iteration_manager : IterationManager
        Provides iteration-specific output paths.

    model : torch.nn.Module
        Trained classification model.

    training_history : Stage1TrainingHistory
        Statistics collected during training.
        (Currently unused but kept for future extensions.)

    Returns
    -------
    str
        Absolute path of the saved checkpoint.
    """

    # ------------------------------------------------------------
    # Resolve checkpoint path
    # ------------------------------------------------------------

    # checkpoint_path = iteration_manager.get_stage1_checkpoint_path()

    # checkpoint_directory = os.path.dirname(checkpoint_path)
    # os.makedirs(checkpoint_directory, exist_ok=True)
    
    checkpoint_directory = Path(config.save_folder) / "stage1"
    checkpoint_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = checkpoint_directory / "stage1_best.pth"

    # ------------------------------------------------------------
    # Create checkpoint dictionary
    # ------------------------------------------------------------

    checkpoint = {
        "model_state_dict": training_history["best_model_state"],
        "network": config.stage1_network,
        "dataset": config.dataset,
        "num_classes": config.n_class,
        "epochs": config.stage1_epochs,
        
        # Training summary
        "best_epoch": training_history["best_epoch"],
        "best_loss": training_history["loss"],
        "best_exact_match": training_history["avg_ep_EM"],
        "best_accuracy": training_history["avg_ep_acc"],
        "total_epochs": training_history["epochs"],
        "total_iterations": training_history["iterations"],
    }

    # ------------------------------------------------------------
    # Save checkpoint
    # ------------------------------------------------------------

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    # ------------------------------------------------------------
    # Log information
    # ------------------------------------------------------------

    logger.info("=" * 80)
    logger.info("Best Stage-1 checkpoint saved")
    logger.info("=" * 80)
    logger.info(f"Checkpoint      : {checkpoint_path}")
    logger.info(f"Best Epoch      : {training_history['best_epoch']}")
    logger.info(f"Best Loss       : {training_history['loss']:.4f}")
    logger.info(f"Exact Match     : {training_history['avg_ep_EM']:.4f}")
    logger.info(f"Accuracy        : {training_history['avg_ep_acc']:.4f}")
    logger.info("=" * 80)

    return checkpoint_path    





def _create_stage1_result(
    config,
    iteration,
    iteration_manager,
    checkpoint_path,
    training_history,
    runtime,
    logger=None,
):
    """
    Create the Stage-1 result object.

    Parameters
    ----------
    config : Stage1Config
        Stage-1 configuration.

    iteration : int
        Current curriculum iteration.

    iteration_manager : IterationManager
        Provides iteration-specific output paths.

    checkpoint_path : str
        Path to the saved classifier checkpoint.

    training_history : Stage1TrainingHistory
        Statistics collected during training.

    runtime : RuntimeContext
        Runtime resources created during initialization.

    Returns
    -------
    Stage1Result
        Complete Stage-1 execution result.
    """

    result = Stage1Result(

        # --------------------------------------------------------
        # Experiment Information
        # --------------------------------------------------------

        iteration=iteration,

        dataset=config.dataset,

        network=config.stage1_network,

        # --------------------------------------------------------
        # Output Artifacts
        # --------------------------------------------------------

        checkpoint_path=checkpoint_path,

        output_directory=iteration_manager.get_iteration_directory(),

        tensorboard_directory=iteration_manager.get_tensorboard_directory(),

        log_directory=iteration_manager.get_log_directory,

        # --------------------------------------------------------
        # Training Information
        # --------------------------------------------------------

        training_history=training_history,
        
        inference_score=None,

        runtime_seconds= (datetime.now() - runtime.start_time).total_seconds(),
        
        start_time=runtime.start_time,
        
        end_time=datetime.now(),

    )

    return result


def _create_test_result(
    config,
    checkpoint_path,
    results,
    logger,
):
    """
    Package Stage-1 testing results.

    Parameters
    ----------
    config
        Curriculum configuration.

    iteration : int
        Current curriculum iteration.

    checkpoint_path : Path
        Checkpoint evaluated during testing.

    inference_score : dict | float | object
        Score returned by infer().

    runtime : RuntimeContext
        Runtime resources.

    Returns
    -------
    Stage1Result
        Stage-1 testing result.
    """
    
    evaluation_time = datetime.now()

    # ------------------------------------------------------------
    # Log Evaluation Summary
    # ------------------------------------------------------------

    logger.info("=" * 80)
    logger.info("Stage-1 Evaluation Summary")
    logger.info("=" * 80)

    logger.info(f"Checkpoint      : {checkpoint_path}")
    logger.info(f"Test Dataset    : {config.stage1_testroot}")
    logger.info(f"Evaluation Time : {evaluation_time}")

    if isinstance(results, dict):

        logger.info("-" * 80)

        for metric, value in results.items():

            logger.info(f"{metric:<25}: {value}")

    else:

        logger.info(f"Results : {results}")

    logger.info("=" * 80)


    return Stage1TestResult(
        checkpoint_path=checkpoint_path,
        test_dataset=config.stage1_testroot,
        inference_results=results,
        evaluation_time=datetime.now(),
    )
    
    
    