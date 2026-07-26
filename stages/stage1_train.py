import time

from stages.stage1.build_dataloader import _build_train_dataloader
from stages.stage1.build_model import _build_classifier
from stages.stage1.initialize import _initialize_runtime
from stages.stage1.stage1_utils import _build_optimizer, _create_stage1_result, _load_pretrained_weights, _load_stage1_checkpoint, _save_checkpoint
from stages.stage1.train_classifier import _train_classifier
from tool.GenDataset import _build_validation_dataloader
from utils import logger
from utils.logger import get_logger

def train_stage1(
    config,
    iteration,
    iteration_manager,
):
    """
    Train the Stage-1 classification model for a single curriculum iteration.

    Parameters
    ----------
    config : Stage1Config
        Complete Stage-1 training configuration.

    iteration : int
        Current curriculum iteration.

    iteration_manager : IterationManager
        Provides all iteration-specific paths and resources.

    Returns
    -------
    Stage1Result
        Information about the completed Stage-1 training.
    """

    logger = get_logger(
            name="stage1_train",
            log_directory="logs/stage1_train",
            log_level=config.log_level,
        )

    logger.info("=" * 80)
    logger.info("Stage-1 Classification Training")
    logger.info("=" * 80)
    logger.info(f"{'Curriculum Iteration':<25}: {iteration}")
    logger.info(f"{'Dataset':<25}: {getattr(config, 'dataset', 'N/A')}")
    logger.info(f"{'Backbone':<25}: {getattr(config, 'backbone', 'N/A')}")
    logger.info(f"{'Epochs':<25}: {getattr(config, 'stage1_epochs', getattr(config, 'epochs', 'N/A'))}")
    logger.info(f"{'Learning Rate':<25}: {getattr(config, 'stage1_lr', getattr(config, 'lr', 'N/A'))}")
    logger.info(f"{'Batch Size':<25}: {getattr(config, 'batch_size', 'N/A')}")
    logger.info("")

    start_time = time.time()

    # ------------------------------------------------------------------
    # Phase 1 : Initialization
    # ------------------------------------------------------------------
    logger.info("[1/7] Initializing Stage-1 runtime...")
    
    runtime = _initialize_runtime(
        config=config,
        iteration=iteration,
        iteration_manager=iteration_manager,
    )

    logger.info("Stage-1 runtime initialized successfully.")
    logger.info("")

    # ------------------------------------------------------------------
    # Phase 2 : Build Model
    # ------------------------------------------------------------------
    logger.info("[2/7] Building classification model...")

    model = _build_classifier(
        config=config,
        runtime=runtime,
        logger=logger,
    )

    logger.info("Classification model created successfully.")
    logger.info("")

    # ------------------------------------------------------------------
    # Phase 3 : Build Dataset
    # ------------------------------------------------------------------
    logger.info("[3/7] Building Stage-1 dataloaders...")

    train_loader = _build_train_dataloader(
            config=config,
            iteration=iteration,
            iteration_manager=iteration_manager,
            runtime=runtime,
            logger=logger,
        )

    logger.info(f"{'Dataset size':<25}: {len(train_loader.dataset)}")
    logger.info(f"{'Number of batches':<25}: {len(train_loader)}")

    logger.info("Stage-1 dataloaders created successfully.")
    logger.info("")

    # ------------------------------------------------------------------
    # Phase 4 : Optimizer & Scheduler
    # ------------------------------------------------------------------
    logger.info("[4/7] Building optimizer and scheduler...")

    optimizer = _build_optimizer(
        config=config,
        model=model,
        train_loader=train_loader,
    )

    logger.info("Optimizer and scheduler initialized.")
    logger.info("")


    # ------------------------------------------------------------------
    # Phase 5 : Load Pretrained Weights
    # ------------------------------------------------------------------
    logger.info("[5/7] Loading Stage-1 weights...")

    if iteration == 0:
        logger.info("Iteration 0 detected. Loading ImageNet pretrained weights.")
        # Start from ImageNet pretrained backbone
        _load_pretrained_weights(config=config, model=model, logger=logger)
    else:
        logger.info("Loading previous best Stage-1 checkpoint for curriculum learning.")
        # Continue from the best Stage-1 checkpoint
        _load_stage1_checkpoint(config=config, model=model, logger=logger)

    logger.info("Stage-1 weights loaded successfully.")
    logger.info("")

    # ------------------------------------------------------------------
    # Phase 6 : Training
    # ------------------------------------------------------------------
    logger.info("[6/7] Starting Stage-1 training...")
    logger.info("")

    training_history = _train_classifier(
        config=config,
        iteration=iteration,
        runtime=runtime,
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        logger=logger,
    )

    logger.info("Stage-1 training completed.")
    logger.info("")

    # ------------------------------------------------------------------
    # Phase 8 : Save Checkpoint
    # ------------------------------------------------------------------
        # checkpoint_path = _save_checkpoint(
        #     config=config,
        #     iteration_manager=iteration_manager,
        #     model=model,
        # )
    logger.info("[7/7] Saving checkpoint and packaging results...")
    checkpoint_path = _save_checkpoint(
        config=config,
        training_history=training_history,
        logger=logger,
    )

    # ------------------------------------------------------------------
    # Phase 9 : Package Results
    # ------------------------------------------------------------------
    result = _create_stage1_result(
        config=config,
        iteration=iteration,
        iteration_manager=iteration_manager,
        checkpoint_path=checkpoint_path,
        training_history=training_history,
        runtime=runtime,
        logger=logger,
    )

    runtime_seconds = time.time() - start_time

    logger.info("=" * 80)
    logger.info("Stage-1 Training Completed")
    logger.info("=" * 80)
    logger.info(f"Curriculum Iteration : {iteration}")
    logger.info(f"Total Runtime        : {runtime_seconds:.2f} seconds")
    logger.info("=" * 80)

    return result