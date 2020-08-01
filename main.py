import os
import logging
import argparse

import torch
from torch.optim import SGD
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import *
from run_manager import RunManager
from vggface2_data_manager import VGGFace2DataManager


parser = argparse.ArgumentParser("CR-FR")
# Generic usage
parser.add_argument('-s', '--seed', type=int, default=41, help='Set random seed (default: 41)')
# Dataset and run mode
parser.add_argument('-dn', '--dataset', choices=['tinyface', 'vggface2', 'vggface2-500', 'ijbb', 'ijbc', 'qmul'],
                default='tinyface', help='Dataset name (default: tinyface)')
parser.add_argument('-rm', '--run-mode', choices=['train', 'test', 'extr_feat'],
                default='extr_feat', help='Run mode (default: extr_feat)')
# Model related options
parser.add_argument('-bp', '--model-base-path', help='Path to base model checkpoint')
parser.add_argument('-ckp', '--model-ckp', help='Path to fine tuned model checkpoint')
# Use super resolved images for features extraction
parser.add_argument('-sr', '--super-resolved-images', action='store_true',
                help='Extract features from SR images. It is only valid for QMUL and Tinyface (default: False)')
# Training Options
parser.add_argument('-dp', '--dset-base-path', help='Base path to datasets')
parser.add_argument('-l', '--lambda_', default=0.2, type=float,
                help='Lambda for features regression loss (default: 0.2)')
parser.add_argument('-lr', '--learning-rate', default=0.001, type=float, help='Learning rate (default: 1.e-3)')
parser.add_argument('-m', '--momentum', default=0.9, type=float, help='Optimizer momentum (default: 0.9)')
parser.add_argument('-lp', '--lower-resolution-prob', default=0.5, type=float,
                help='Lowering resoltion probability (default: 0.5)')
parser.add_argument('-e', '--epochs', type=int, default=1, help='Training epochs (default: 1)')
parser.add_argument('-rs', '--train-steps', type=int, default=2,
                help='Set number of training iterations before each validation run (default: 2)')
parser.add_argument('-c', '--curriculum', action='store_true', help='Use curriculum learning (default: False)')
parser.add_argument('-cs', '--curr-step-iterations', type=int, default=35000, help='Number of images for each curriculum step (default: 35000)')
parser.add_argument('-sp', '--scheduler-patience', type=int, default=10, help='Scheduler patience (default: 10)')
parser.add_argument('-bs', '--batch-accumulation', type=int, default=8, help='Batch accumulation iterations (default: 8)')
# Added ontly to downsample scface images at 64 pixels
parser.add_argument('-ds', '--downsample', action='store_true')

args = parser.parse_args()


# ----------------------------- GENERAL ----------------------------------------
tmp = (
    f"{args.run_mode}-{args.super_resolved_images}-{args.lambda_}-{args.learning_rate}-"
    f"{args.lower_resolution_prob}-{args.run_steps}-{args.curriculum}-{args.curr_step_iterations}"
)

out_dir = create_folder(args.experimental_path, os.path.join(args.dataset, tmp))

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(out_dir, 'training.log')),
        logging.StreamHandler()
    ])
logger = logging.getLogger()

logging.info(f"Training outputs will be saved here: {out_dir}")

logging.info("Start training with params:")
# ------------------------------------------------------------------------------


# --------------------------- CUDA SET UP --------------------------------------
cudnn.benchmark = True

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

cuda_available = torch.cuda.is_available()
device = torch.device('cuda' if cuda_available else 'cpu')
# ------------------------------------------------------------------------------


# ---------------- LOAD MODEL & OPTIMIZER & SCHEDULER --------------------------
sm, tm = load_models(args.model_base_path, device, args.model_ckp)
optimizer = SGD(
            parameters=sm.parameters(), 
            lr=args.learning_rate, 
            momentum=args.optimizer_momentum, 
            weight_decay=1e-05, 
            nesterov=True
        )
scheduler = ReduceLROnPlateau(
                        optimizer=optimizer, 
                        mode='min', 
                        factor=0.5,
                        patience=args.scheduler_patience, 
                        verbose=True,
                        min_lr=1.e-7, 
                        threshold=0.1
                    )
# ------------------------------------------------------------------------------


# ---------------------------- LOAD DATA ---------------------------------------
kwargs = {
    'run_mode': args.run_mode, 
    'batch_size': args.batch_size,
    'lowering_resolution_prob': args.lowering_resolution_prob,
    'curriculum': args.curriculum,
    'curr_step_iterations': args.curr_step_iterations, 
    'algo_name': 'bilinear',
    'algo_val': PIL.Image.BILINEAR,
    'valid_fix_resolution': args.valid_fix_resolution,
}
data_manager = VGGFace2DataManager(
                            dataset_path=args.dset_base_path,
                            img_folders=['train_copied', 'validation'],
                            transforms=[get_transforms(mode='train'), get_transforms(mode='eval')],
                            device=device,
                            logging=logging,
                            **kwargs
                        )
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    run_manager = RunManager(
                        student=sm, 
                        teacher=tm, 
                        optimizer=optimizer,
                        scheduler=scheduler,
                        data_manager=data_manager.get_dataset_manager(),
                        dataset=args.dataset,
                        lowering_resolution_prob=args.lowering_resolution_prob,
                        device=device,
                        curriculum=args.curriculum,
                        epochs=args.epochs,
                        lambda_=args.lambda_,
                        train_steps=args.train_steps,
                        run_mode=args.run_mode,
                        super_resolved_images=args.super_resolved_images,
                        out_dir=out_dir,
                        logging=logging
                    )
    run_manager.extract_features() if args.run_mode == "extr_feat" else run_manager.run()
    
