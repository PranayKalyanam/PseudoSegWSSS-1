from tqdm import tqdm

import numpy as np
import torch
from copy import deepcopy
from tool import pyutils

import torch.nn.functional as F


def compute_acc(pred_labels, gt_labels):
    pred_correct_count = 0
    for pred_label in pred_labels:
        if pred_label in gt_labels:
            pred_correct_count += 1
    union = len(gt_labels) + len(pred_labels) - pred_correct_count
    acc = round(pred_correct_count/union, 4)
    return acc

def _train_classifier(
    config,
    iteration,
    runtime,
    model,
    train_loader,
    optimizer,
    logger=None,
):
    """
    Train the Stage-1 classifier for one curriculum iteration.

    Parameters
    ----------
    config : Stage1Config

    iteration : int

    runtime : RuntimeContext

    model : torch.nn.Module

    train_loader : DataLoader

    optimizer : torch.optim.Optimizer

    Returns
    -------
    pyutils.AverageMeter
        Training statistics collected during optimization.
    """

    use_pseudo = iteration > config.use_pseudo_iteration

    logger.info("=" * 80)
    logger.info("Stage-1 Training")
    logger.info("=" * 80)

    avg_meter = pyutils.AverageMeter(
        "loss",
        "avg_ep_EM",
        "avg_ep_acc",
    )
    
   

    best_loss = float("inf")
    best_epoch = -1
    best_model_state = None



    max_step = len(train_loader) * config.stage1_epochs

    # for epoch in range(config.stage1_epochs):
    epoch_bar = tqdm(
        range(config.stage1_epochs),
        desc="Training",
        dynamic_ncols=True,
        leave=True,
    )

    for epoch in epoch_bar:

        model.train()

        config.ep_index = epoch
        ep_count = 0
        ep_EM = 0
        ep_acc = 0

        for batch_index, batch in enumerate(train_loader):

            # ----------------------------------------------------
            # Load Batch
            # ----------------------------------------------------

            if use_pseudo:
                filename, image, label, pseudo_label = batch
                pseudo_label = pseudo_label.to(runtime.device, non_blocking=True)
            else:
                filename, image, label = batch

            image = image.to(runtime.device)

            label = label.to(
                runtime.device,
                non_blocking=True,
            )

            # ----------------------------------------------------
            # Progressive Dropout Attention
            # ----------------------------------------------------

            enable_PDA = 1 if epoch > 2 else 0

            # ----------------------------------------------------
            # Forward Pass
            # ----------------------------------------------------

            logits, feature, prediction = model(
                image,
                enable_PDA,
            )

            # ----------------------------------------------------
            # Metrics
            # ----------------------------------------------------

            prob = prediction.detach().cpu().numpy()

            gt = label.detach().cpu().numpy()

            for sample_index, sample_prediction in enumerate(prob):

                ep_count += 1

                predicted_classes = np.where(sample_prediction > 0.5)[0]

                ground_truth_classes = np.where(
                    gt[sample_index] == 1
                )[0]

                if np.array_equal(
                    predicted_classes,
                    ground_truth_classes,
                ):
                    ep_EM += 1

                ep_acc += compute_acc(
                    predicted_classes,
                    ground_truth_classes,
                )

            avg_ep_EM = round(ep_EM / ep_count, 4)

            avg_ep_acc = round(ep_acc / ep_count, 4)

            # ----------------------------------------------------
            # Loss
            # ----------------------------------------------------

            gt_loss = F.multilabel_soft_margin_loss(
                logits,
                label,
            )

            if use_pseudo:
                # ----------------------------------------------------
                # Pseudo Label Loss
                # ----------------------------------------------------
                # Stage-1 is formulated as a multi-label image
                # classification problem. Since pseudo labels have the
                # same multi-hot representation as the ground-truth
                # image labels, the identical multilabel soft margin
                # loss is used for both supervision sources.
                #
                # Future work:
                # Confidence-aware weighting, soft pseudo labels,
                # label smoothing, or consistency regularization can
                # be incorporated here without modifying the remainder
                # of the training loop.
                # ----------------------------------------------------

                #
                # Curriculum loss will be implemented here.
                #
                pseudo_loss = F.multilabel_soft_margin_loss(
                    logits,
                    pseudo_label,
                    )
                
                total_loss = config.stage1_gt_loss_weight * gt_loss + config.stage1_pseudo_loss_weight * pseudo_loss
                
            else:

                total_loss = gt_loss


            avg_meter.add({

                "loss": total_loss.item(),

                "avg_ep_EM": avg_ep_EM,

                "avg_ep_acc": avg_ep_acc,

            })

            # ----------------------------------------------------
            # Optimization
            # ----------------------------------------------------

            optimizer.zero_grad()

            total_loss.backward()

            optimizer.step()
            
            epoch_bar.set_description(
                f"Epoch {epoch + 1}/{config.stage1_epochs}"
            )

            epoch_bar.set_postfix(
                Loss=f"{total_loss.item():.4f}",
                EM=f"{avg_ep_EM:.4f}",
                Acc=f"{avg_ep_acc:.4f}",
                LR=f"{optimizer.param_groups[0]['lr']:.2e}",
                PDA=f"{model.gama:.3f}",
            )

            if runtime.device.type == "cuda":
                torch.cuda.empty_cache()

            # ----------------------------------------------------
            # Logging
            # ----------------------------------------------------

            if (
                optimizer.global_step % 20 == 0
                and optimizer.global_step != 0
            ):

                runtime.timer.update_progress(
                    optimizer.global_step / max_step
                )

                if logger:
                    logger.info(

                    "Epoch:%2d "
                    "Iter:%5d/%5d "
                    "Loss:%.4f "
                    "avg_ep_EM:%.4f "
                    "avg_ep_acc:%.4f "
                    "lr: %.6f "
                    "Fin:%s"

                    % (

                        epoch,

                        optimizer.global_step,

                        max_step,

                        avg_meter.get("loss"),

                        avg_meter.get("avg_ep_EM"),

                        avg_meter.get("avg_ep_acc"),

                        optimizer.param_groups[0]["lr"],

                        runtime.timer.str_est_finish(),

                    )

                )

                runtime.writer.add_scalar(
                    "Loss/train",
                    avg_meter.get("loss"),
                    optimizer.global_step,
                )

                runtime.writer.add_scalar(
                    "Accuracy/exact",
                    avg_meter.get("avg_ep_EM"),
                    optimizer.global_step,
                )

                runtime.writer.add_scalar(
                    "Accuracy/standard",
                    avg_meter.get("avg_ep_acc"),
                    optimizer.global_step,
                )

        # --------------------------------------------------------
        # Progressive Dropout Gamma Update
        # --------------------------------------------------------

        if model.gama > 0.65:

            model.gama *= 0.98

        logger.info(
            f"PDA Gamma : {model.gama:.4f}"
        )
        
        # --------------------------------------------------------
        # Save Best Model (Training Criterion)
        # --------------------------------------------------------

        current_loss = avg_meter.get("loss")

        if current_loss < best_loss:

            best_loss = current_loss
            best_epoch = epoch + 1
            best_model_state = deepcopy(model.state_dict())

            logger.info(
                f"New best model saved "
                f"(Epoch {best_epoch}, Loss={best_loss:.4f})"
            )
        
    # --------------------------------------------------------
    # Final Training Summary
    # --------------------------------------------------------

    training_history = {
        "loss": avg_meter.get("loss"),
        "avg_ep_EM": avg_meter.get("avg_ep_EM"),
        "avg_ep_acc": avg_meter.get("avg_ep_acc"),
        "epochs": config.stage1_epochs,
        "iterations": optimizer.global_step,
        "best_epoch": best_epoch, 
        "best_model_state": best_model_state,
    }

    if logger:
        logger.info("=" * 80)
        logger.info("Stage-1 Training Completed")
        logger.info("=" * 80)
        logger.info(f"Final Loss        : {training_history['loss']:.4f}")
        logger.info(f"Final Exact Match : {training_history['avg_ep_EM']:.4f}")
        logger.info(f"Final Accuracy    : {training_history['avg_ep_acc']:.4f}")

    runtime.writer.flush()
    runtime.writer.close()

    return training_history


