"""
stages/stage1/run_inference.py

Runs Stage-1 classifier inference on the testing dataset.
"""



from tqdm import tqdm

from tool.infer_fun import infer



def _run_inference(
    config,
    model,
    logger,
):
    """
    Run Stage-1 inference.

    Parameters
    ----------
    config : CurriculumConfig

    runtime : RuntimeContext

    model : torch.nn.Module

    Returns
    -------
    dict | float | object
        Evaluation results returned by the baseline inference
        implementation.
    """

    logger.info("=" * 80)
    logger.info("Stage-1 Testing")
    logger.info("=" * 80)

    logger.info("Evaluation Configuration")
    logger.info("-" * 80)
    logger.info("Testing Dataset      : %s", config.stage1_testroot)
    logger.info("Number of Classes    : %d", config.n_class)
    logger.info("Checkpoint           : %s", config.stage1_checkpoint)
    logger.info("Device               : %s", next(model.parameters()).device)
    logger.info("-" * 80)

    logger.info("Starting inference on unseen test dataset...")

    # ------------------------------------------------------------
    # Run Inference
    # ------------------------------------------------------------

    with tqdm(
        total=1,
        desc="Running Stage-1 Inference",
        bar_format="{desc}: {elapsed}",
    ) as progress_bar:

        score = infer(
            model=model,
            dataroot=config.stage1_testroot,
            n_class=config.n_class,
        )

        progress_bar.update(1)
    
    logger.info("Inference completed successfully.")
    logger.info("-" * 80)
    logger.info("Evaluation Results")
    logger.info("-" * 80)
    logger.info("%s", score)
    logger.info("=" * 80)

    return score

