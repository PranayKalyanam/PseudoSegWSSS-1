from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


# ============================================================================
# BCSS Semantic Classes
# ============================================================================

BACKGROUND_CLASS = 0

BCSS_LABELS = {
    1: "tumor",
    2: "stroma",
    3: "lymphocyte",
    4: "necrosis",
}



def positive_int(value: str) -> int:
    """
    Ensure integer is positive.
    """
    ivalue = int(value)

    if ivalue <= 0:
        raise argparse.ArgumentTypeError(
            f"{ivalue} is not a positive integer."
        )

    return ivalue

# ============================================================================
# Stage 1: HistoGraphWSL Pipeline Configuration
# ============================================================================

def build_stage1_parser() -> argparse.ArgumentParser:
    """
    Builds the argument parser for Stage 1 (HistoGraphWSL Pipeline).
    """
    
    parser = argparse.ArgumentParser(description="HistoGraphWSL Pipeline - Stage 1: Initial Training")

    # --- Dataset Path Options ---
    parser.add_argument("--dataset", type=str, default="luad", help="Dataset name.")
    parser.add_argument("--output_dir", type=str, default="outputs/")
    
    # New: Split dataset roots for train/test if required by your script
    parser.add_argument("--trainroot", type=str, default="datasets/LUAD-HistoSeg/train5/")
    parser.add_argument("--testroot", type=str, default="datasets/LUAD-HistoSeg/test/")
    parser.add_argument("--dataroot", default="datasets/LUAD-HistoSeg", type=str, help="Root directory of the dataset.")
    parser.add_argument("--save_folder", type=str, default="checkpoints/")

    # --- Optimization & Training Options ---
    parser.add_argument("--batch_size", type=positive_int, default=20)
    parser.add_argument("--max_epoches", type=positive_int, default=20)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--wt_dec", type=float, default=5e-4)
    parser.add_argument("--init_gama", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    # --- Network Options ---
    parser.add_argument("--network", type=str, default="network.resnet38_cls")
    parser.add_argument("--n_class", type=positive_int, default=4, help="Number of classes for segmentation. Ex: [Tumor, Stroma, Lymphocyte, Necrosis] = 4 classes") 
    
    # ===================================== Model Weights (Toggle stages as needed) =============================================================
    # --- stage 1: intial training weights --- --- comment this line when you run other stages ---
    # parser.add_argument("--weights", type=str, default="init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.pth")
    
    # --- stage 2: pseudo mask generation: stage 1 final weights ---  --- comment this line when you run other stages ---
    parser.add_argument("--weights", default='checkpoints/stage1_checkpoint_trained_on_luad.pth', type=str)
    #=============================================================================================================================================
    
    # --- Runtime Environments ---
    parser.add_argument("--session_name", type=str, default="Stage 1")
    parser.add_argument("--env_name", type=str, default="PDA")
    parser.add_argument("--model_name", type=str, default="PDA")
    parser.add_argument("--num_workers", type=positive_int, default=2, help="Set to 2 for Colab environment stability.")
    parser.add_argument("--dry-run", action="store_true", default=False)
   
    # --- Logging ---
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=[
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
        ]
    )

    return parser





def get_stage1_config(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Parses arguments for Stage 1, creates necessary directories, and returns parsed configurations.
    """

    parser = build_stage1_parser()

    args = parser.parse_args(argv)

    # create_output_dirs(Path(args.output_dir))

    return args


# ============================================================================
# Stage 2: Weakly Supervised Semantic Segmentation (WSSS) Configuration
# ============================================================================
def build_stage2_parser() -> argparse.ArgumentParser:
    """
    Builds the argument parser for Stage 2 (WSSS Stage 2 using DeepLab/Backbones).
    """
    parser = argparse.ArgumentParser(description="WSSS Stage2 Pipeline")
    
    # --- Execution Controls ---
    parser.add_argument("--test-only", "-t", action="store_true", default=False, help="Bypass training and validation to run testing directly.")
    
    # --- Network Architecture Options ---
    parser.add_argument("--backbone", type=str, default="resnet", choices=["resnet", "xception", "drn", "mobilenet"], help="Feature extractor backbone.")
    parser.add_argument("--out-stride", type=int, default=16, help="Network output stride.")
    parser.add_argument("--Is_GM", type=bool, default=True, help="Enable the Gate Mechanism in the testing phase.")
    
    # --- Dataset Options ---
    parser.add_argument("--dataroot", type=str, default="datasets/LUAD-HistoSeg", help="Root directory of the dataset.")
    parser.add_argument("--dataset", type=str, default="luad", help="Dataset name.")
    parser.add_argument("--savepath", type=str, default="checkpoints/", help="Directory to save training checkpoints.")
    parser.add_argument("--workers", type=int, default=2, metavar="N", help="Number of dataloader worker threads.")
    
    # --- Batch Normalization & Loss ---
    parser.add_argument("--sync-bn", type=bool, default=None, help="Use synchronous batch normalization.")
    parser.add_argument("--freeze-bn", type=bool, default=False, help="Freeze batch normalization layers during training.")
    parser.add_argument("--loss-type", type=str, default="ce", choices=["ce", "focal"], help="Loss function type.")
    parser.add_argument("--n_class", type=int, default=4, help="Number of segmentation classes.")
    
    # --- Training Hyperparameters ---
    parser.add_argument("--epochs", type=int, default=20, metavar="N", help="Number of epochs to train.")
    parser.add_argument("--batch-size", type=int, default=16, metavar="N", help="Input batch size for training.")
    
    # --- Optimizer Parameters ---
    parser.add_argument("--lr", type=float, default=0.01, metavar="LR", help="Initial learning rate.")
    parser.add_argument("--lr-scheduler", type=str, default="poly", choices=["poly", "step", "cos"], help="Learning rate schedule approach.")
    parser.add_argument("--momentum", type=float, default=0.9, metavar="M", help="SGD momentum factor.")
    parser.add_argument("--weight-decay", type=float, default=5e-4, metavar="M", help="SGD weight decay factor.")
    parser.add_argument("--nesterov", action="store_true", default=False, help="Enable Nesterov momentum.")
    
    # --- Environment & CUDA Settings ---
    parser.add_argument("--no-cuda", action="store_true", default=False, help="Disables CUDA (GPU) training.")
    parser.add_argument("--gpu-ids", type=str, default="0", help="Comma-separated GPU IDs to use (e.g. '0,1').")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="Random seed for reproducibility.")
    
    # --- Checkpointing & Fine-Tuning ---
    parser.add_argument("--resume", type=str, default="init_weights/deeplab-resnet.pth.tar", help="Path to checkpoint model to resume from.")
    parser.add_argument("--checkname", type=str, default="deeplab-resnet", help="Identifier name for saving checkpoints.")
    parser.add_argument("--ft", action="store_true", default=False, help="Enable fine-tuning on a pre-trained model.")
    parser.add_argument("--eval-interval", type=int, default=1, help="Validation interval (in epochs).")
   
    return parser
  
def get_stage2_config(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Parses arguments for Stage 2 and returns parsed configurations.
    """
    parser = build_stage2_parser()
    args = parser.parse_args(argv)
    
    return args




# ============================================================================
# Curriculum Learning Configuration
# ============================================================================
def build_curriculum_parser() -> argparse.ArgumentParser:
    """
    Build the argument parser for the curriculum learning framework.

    The curriculum configuration serves as the central configuration for the
    entire pipeline. Stage-specific parameters are grouped using stage-specific
    prefixes to avoid naming conflicts and improve maintainability.
    """

    parser = argparse.ArgumentParser(
        description="Curriculum-Based Weakly Supervised Semantic Segmentation"
    )

    # ==========================================================================
    # Dataset Configuration
    """Top-level dataset location and output directory options."""
    # ==========================================================================
    parser.add_argument("--dataset", type=str, default="luad", help="Dataset name.")
    parser.add_argument("--dataroot", type=str, default="datasets/LUAD-HistoSeg", help="Root directory of the dataset.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Root directory for curriculum outputs.")


    # ==========================================================================
    # Stage-1 Configuration
    """Stage-1 classification network: data paths, weights, and training hyperparameters."""
    # ==========================================================================
    # --- Data paths / checkpoints ---
    parser.add_argument("--stage1_trainroot", type=str, default="datasets/LUAD-HistoSeg/train5/", help="Training dataset.")
    parser.add_argument("--pseudo_labels_root", type=str, default="datasets/LUAD-HistoSeg/train_PL/", help="Training dataset.")
    parser.add_argument("--stage1_testroot", type=str, default="datasets/LUAD-HistoSeg/test/", help="Testing dataset.")
    parser.add_argument("--stage1_network", type=str, default="network.resnet38_cls", help="Classification network.")
    parser.add_argument("--stage1_weights", type=str, default="init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.pth", help="Initial classification weights.")
    parser.add_argument( "--stage1_checkpoint", type=str, default="checkpoints/stage1/stage1_best.pth", help="Checkpoint of the trained Stage-1 classifier used for evaluation.", )
    
    # --- Training hyperparameters ---
    parser.add_argument("--stage1_epochs", type=positive_int, default=20, help="Number of training epochs.")
    parser.add_argument("--stage1_lr", type=float, default=0.01, help="Initial learning rate.")
    parser.add_argument("--stage1_wt_dec", type=float, default=5e-4, help="Weight decay.")
    parser.add_argument("--stage1_init_gama", type=float, default=1.0, help="Initial Progressive Dropout Attention coefficient.")
    parser.add_argument("--num_workers", type=positive_int, default=4, help="Number of dataloader workers.")

    # --- Logging / naming ---
    parser.add_argument("--stage1_session_name", type=str, default="Stage 1", help="Training session name.")
    parser.add_argument("--stage1_env_name", type=str, default="PDA", help="TensorBoard environment.")
    parser.add_argument("--stage1_model_name", type=str, default="PDA", help="Model identifier.")
    
    # --- Loss weighting ---
    parser.add_argument("--stage1_gt_loss_weight", type=float, default=0.8, help="Stage 1 ground truth loss weight.")
    parser.add_argument("--stage1_pseudo_loss_weight", type=float, default=0.2, help="Stage 1 pseudo-label loss weight.")



    # ==========================================================================
    # Stage-2 Configuration
    """Stage-2 segmentation model: data paths, backbone/model, training, and inference options. """
    # ==========================================================================
 
    # --- Data roots / sub-directories ------------------------------------------
    parser.add_argument("--stage2_train_image_subdir", type=str, default="train5", help="Sub-directory (under dataroot) containing Stage-2 training images.")
    parser.add_argument("--stage2_pseudo_mask_root", type=str, default="train_PM", help="Sub-directory (under dataroot) containing generated pseudo masks.")
    parser.add_argument("--stage2_pseudo_mask_dir", type=str, default="PM_bn7", help="Pseudo-mask sub-directory used as the main training target.")
    parser.add_argument("--stage2_pseudo_mask_dir_a", type=str, default="PM_b5_2", help="Pseudo-mask sub-directory used as the secondary (A) training target.")
    parser.add_argument("--stage2_pseudo_mask_dir_b", type=str, default="PM_b4_5", help="Pseudo-mask sub-directory used as the tertiary (B) training target.")
    parser.add_argument("--stage2_val_subdir", type=str, default="val", help="Sub-directory (under dataroot) containing validation images/masks.")
    parser.add_argument("--stage2_test_subdir", type=str, default="test", help="Sub-directory (under dataroot) containing test images/masks.")
 
    # --- Checkpoints ---------------------------------------------------------
    parser.add_argument("--save_folder", type=str, default="checkpoints/", help="Directory Stage-2 checkpoints are saved to.")
    parser.add_argument("--stage2_checkname", type=str, default="deeplab-resnet", help="Checkpoint/experiment name used by the Saver.")
    parser.add_argument("--stage2_checkpoint", type=str, default="checkpoints/stage2/stage2_best.pth", help="Checkpoint of the trained Stage-2 model used for evaluation/testing.")
    parser.add_argument("--stage2_resume", type=str, default="init_weights/deeplab-resnet.pth.tar", help="Checkpoint to resume/initialize Stage-2 weights from at curriculum iteration 0.")
    parser.add_argument("--stage2_ft", action="store_true", default=False, help="Fine-tuning mode: keep the segmentation head and restore optimizer state from `--stage2_resume`.")
    parser.add_argument("--stage2_eval_interval", type=positive_int, default=1, help="Run validation every N epochs.")
 
    # --- Model / backbone ----------------------------------------------------
    parser.add_argument("--backbone", type=str, default="resnet", choices=["resnet", "xception", "drn", "mobilenet"], help="Feature extractor backbone.")
    parser.add_argument("--stage2_out_stride", type=positive_int, default=16, help="Output stride of the backbone.")
    parser.add_argument("--n_class", type=positive_int, default=4, help="Number of segmentation classes (including the ignore class).")
    parser.add_argument("--stage2_sync_bn", type=bool, default=None, help="Use synchronized batch norm. Defaults to True when running on >1 GPU, else False.")
    parser.add_argument("--stage2_freeze_bn", type=bool, default=False, help="Freeze batch-norm statistics during training.")
 
    # --- Training hyperparameters ---------------------------------------------
    parser.add_argument("--batch_size", type=positive_int, default=20, help="Training batch size.")
    parser.add_argument("--stage2_epochs", type=positive_int, default=3, help="Number of training epochs.")
    parser.add_argument("--stage2_lr", type=float, default=0.01, help="Initial (1x) learning rate for backbone parameters; the segmentation head uses 10x this value.")
    parser.add_argument("--stage2_lr_scheduler", type=str, default="poly", choices=["poly", "step", "cos"], help="Learning-rate schedule.")
    parser.add_argument("--stage2_momentum", type=float, default=0.9, help="SGD momentum.")
    parser.add_argument("--stage2_weight_decay", type=float, default=5e-4, help="SGD weight decay.")
    parser.add_argument("--stage2_nesterov", action="store_true", default=False, help="Use Nesterov momentum.")
    parser.add_argument("--stage2_loss_type", type=str, default="ce", choices=["ce", "focal"], help="Segmentation loss function.")
   
    # --- CUDA / devices --------------------------------------------------------
    # Whether Stage-2 actually runs on CUDA is still governed by the shared
    # top-level `--device` flag (see Runtime Configuration below); this only
    # controls *which* GPUs are used once CUDA is selected.
    parser.add_argument("--cuda", action="store_true", default=False, help="Enables CUDA (GPU) training.")
    parser.add_argument("--stage2_gpu_ids", type=str, default="0", help="Comma-separated GPU ids to use when running on CUDA.")
 
    # --- Testing / inference -----------------------------------------------
    parser.add_argument("--stage2_is_gm", type=bool, default=True, help="Enable the CAM-based Gated Mechanism (using the Stage-1 classifier) during Stage-2 testing.")
    parser.add_argument("--stage2_test_only", action="store_true", default=False, help="Skip training and evaluate `--stage2_checkpoint` directly.")
 
    parser.add_argument("--stage2_gt_loss_weight", type=float, default=0.6, help="Loss weight for the primary pseudo-mask target.")
    parser.add_argument("--stage2_pseudo_loss_weight_a", type=float, default=0.2, help="Loss weight for the secondary (A) pseudo-mask target.")
    parser.add_argument("--stage2_pseudo_loss_weight_b", type=float, default=0.2, help="Loss weight for the tertiary (B) pseudo-mask target.")
 
    
    # ==========================================================================
    # Curriculum Configuration
    # ==========================================================================

    parser.add_argument("--num_iterations", type=positive_int, default=2, help="Maximum number of curriculum iterations.")
    parser.add_argument("--start_iteration", type=int, default=0, help="Iteration index to start from.")
    parser.add_argument("--use_pseudo_iteration", type=int, default=4, help="Iteration index to start using pseudo labels.")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume an existing curriculum experiment.")
    

    # ==========================================================================
    # Runtime Configuration
    # ==========================================================================

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Execution device.")
    parser.add_argument("--dry_run", action="store_true", default=False, help="Run without saving outputs.")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level.")

    return parser


def get_curriculum_config(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """
    Parse curriculum configuration.

    Returns
    -------
    argparse.Namespace
        Curriculum configuration.
    """

    parser = build_curriculum_parser()

    args = parser.parse_args(argv)

    args.output_dir = Path(args.output_dir)

    return args