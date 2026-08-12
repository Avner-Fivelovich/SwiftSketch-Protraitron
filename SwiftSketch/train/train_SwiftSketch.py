# This code is based on https://github.com/openai/guided-diffusion

import os
import json
import logging
from torch.utils.data import DataLoader
import wandb

from utils.fixseed import fixseed
from utils.parser_util import train_args, get_wandb_name
from utils import dist_util
from train.training_loop import TrainLoop
from utils.model_util import create_model_and_diffusion
from utils.get_data import create_data_set


def main():
    args = train_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting SwiftSketch Training Session")

    # Handle num_strokes CLI parameterization and dataset routing
    if args.num_strokes is not None:
        args.num_paths = args.num_strokes
        args.target_key_name = f"svg_{args.num_strokes}s"
        # Auto-route to the correct dataset folder if not provided
        if not args.train_data_dir:
            args.train_data_dir = [f"data/controlsketch_{args.num_strokes}/train"]
        # Update data_name to avoid cache collision
        args.data_name = f"train_data_{args.num_strokes}s"
    else:
        args.num_strokes = args.num_paths

    # Enable GPU execution for the differentiable rasterizer (pydiffvg) to speed up training
    import pydiffvg
    import torch
    pydiffvg.set_use_gpu(True)
    
    # We will set the device after dist_util is initialized
    # pydiffvg.set_device(torch.device(dist_util.dev()))

    fixseed(args.seed)

    wandb_name= get_wandb_name(args)
    if args.use_wandb:
        wandb.init(project=args.wandb_project_name, entity=args.wandb_user,
                    config=args, name=wandb_name,id=wandb.util.generate_id()
                    )

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    
    if not args.cache_path_dir:
        args.cache_path_dir= args.save_dir

    args.save_dir = os.path.join(args.save_dir, wandb_name)


    if os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        logger.info(f"Created save directory: {args.save_dir}")
        
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)
    logger.debug(f"Saved training arguments to {args_path}")

    logger.info("Setting up distributed training (if any)")
    dist_util.setup_dist(args.device)
    
    # Set pydiffvg to use the correct GPU device
    pydiffvg.set_device(dist_util.dev())

    logger.info(f"Starting data creation (target_key: {args.target_key_name}, features: {args.image_features_type})")
    try:
        train_dataset = create_data_set(args.train_data_dir, args.target_key_name, args.image_features_type, args.canvas_width, args.canvas_height, dist_util.dev(), args.scaling_factor, args.cat_data_size, args.sort_by, args.use_data_cache, args.cache_path_dir, args.data_name)
        data = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)
        logger.info(f"Successfully loaded dataset with {len(train_dataset)} examples")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    logger.info("Creating model and diffusion schedule...")
    try:
        model, diffusion = create_model_and_diffusion(args)
        model.to(dist_util.dev())
        total_params = sum(p.numel() for p in model.parameters()) / 1000000.0
        logger.info(f"Model created successfully. Total params: {total_params:.2f}M")
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        raise

    logger.info("Initializing Training Loop...")
    loop = TrainLoop(args, model, diffusion, data)
    
    logger.info("Training started.")
    loop.run_loop()
    logger.info("Training completed successfully.")
    

    if args.use_wandb:
        wandb.finish()
    
if __name__ == "__main__":
    main()
