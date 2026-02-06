from causvid.video_data import VideoRegressionDataset
from causvid.utils.latent_precompute import precompute_causvid_latents, PrecomputedCausVidLatentDataset
from causvid.video_regression import VideoRegression
from collections import defaultdict
from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed
from omegaconf import OmegaConf
import argparse
import torch
import time
import os
import logging
from tqdm import tqdm
import sys
# project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.experiment_utils import setup_experiment

def generate_causvid_exp_name(args):
    components = [args.experiment_name]

    components.append(f"lr{args.lr:.0e}".replace('-0', '-'))
    components.append(f"ep{args.num_epochs}")

    return [c for c in components if c]


class VideoTrainer:
    def __init__(self, config, exp_dir=None, wandb_run=None):
        self.config = config
        self.exp_dir = exp_dir
        self.wandb_run = wandb_run

        # Step 1: Initialize Accelerator
        self.accelerator = Accelerator(
            mixed_precision="bf16" if config.mixed_precision else "no",
            gradient_accumulation_steps=getattr(config, "gradient_accumulation_steps", 1),
            project_dir=self.exp_dir
        )

        self.device = self.accelerator.device
        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.is_main_process = self.accelerator.is_main_process

        # Set seed
        accelerate_set_seed(config.seed)

        if self.is_main_process:
            os.makedirs(self.exp_dir, exist_ok=True)

        # Step 2: Initialize the model
        self.regression_model = VideoRegression(config, device=self.device)

        # Step 3: Initialize the dataloader
        dataset_repeat = config.dataset_repeat
        num_workers = config.num_workers

        dataset_args = {
            "dataset_base_path": config.dataset_base_path,
            "metadata_path": config.metadata_path,
            "num_frames": getattr(config, "num_frames", 81),
            "height": getattr(config, "height", 480),
            "width": getattr(config, "width", 832),
            "time_division_factor": getattr(config, "time_division_factor", 4),
            "time_division_remainder": getattr(config, "time_division_remainder", 1),
        }


        metadata_name = os.path.basename(config.metadata_path)
        cache_save_path = os.path.join(config.dataset_base_path, f"latent_cache_{metadata_name}.pt")
        match_dataset_cache=False
        if os.path.exists(cache_save_path):
            dataset_cache = torch.load(cache_save_path)
            if dataset_cache["dataset_args"] == dataset_args:
                cached_samples = dataset_cache["cached_samples"]
                dataset = PrecomputedCausVidLatentDataset(cached_samples, repeat=dataset_repeat)
                match_dataset_cache=True

        if not match_dataset_cache:
            dataset = VideoRegressionDataset(
                **dataset_args,
                repeat=1
            )
            dataset = precompute_causvid_latents(
                dataset=dataset,
                vae=self.regression_model.vae,
                device=self.device,
                dtype=self.dtype,
                dataset_repeat=dataset_repeat,
                num_workers=num_workers
            )
            cached_samples = dataset.samples
            dataset_cache = {
                "dataset_args": dataset_args,
                "cached_samples": cached_samples
            }
            torch.save(dataset_cache, cache_save_path)


        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=config.batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            drop_last=True
        )

        # Step 4: Initialize optimizer
        self.optimizer_unwrap = torch.optim.AdamW(
            [param for param in self.regression_model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(getattr(config, "beta1", 0.9), getattr(config, "beta2", 0.999))
        )
        self.start_epoch = 0
        self.initial_global_step = 0
        self.resume_checkpoint = getattr(config, "resume", None)

        (
            self.regression_model,
            self.dataloader,
            self.optimizer
        ) = self.accelerator.prepare(
            self.regression_model,
            dataloader,
            self.optimizer_unwrap
        )

        if self.resume_checkpoint:
            self._resume_training_state(self.resume_checkpoint)

        self.regression_model = self.accelerator.unwrap_model(self.regression_model)

        self.accelerator = Accelerator(
            mixed_precision="bf16" if config.mixed_precision else "no",
            gradient_accumulation_steps=getattr(config, "gradient_accumulation_steps", 1),
            project_dir=self.exp_dir
        )

        trainable_params = [param for param in self.regression_model.generator.parameters()
                            if param.requires_grad]
        logging.info(f"parameters before training: {trainable_params[0].flatten()[:10]}")

        self.optimizer_unwrap = torch.optim.AdamW(
            trainable_params,
            lr=config.lr,
            betas=(getattr(config, "beta1", 0.9), getattr(config, "beta2", 0.999))
        ) # have to re-initialize optimizer to avoid issues after unwrap, for zero stage 1
        logging.info("Re-initialized optimizer after unwrapping model for DeepSpeed compatibility")
        # Step 5: Prepare everything with Accelerator
        (
            self.regression_model,
            self.optimizer,
            self.dataloader
        ) = self.accelerator.prepare(
            self.regression_model,
            self.optimizer_unwrap,
            dataloader
        )

        # Keep VAE on CPU/original device (not trained)
        # self.regression_model.vae remains unprepared

        self.step = self.initial_global_step
        self.max_grad_norm = getattr(config, "max_grad_norm", 10.0)
        self.previous_time = None

        if self.is_main_process:
            original_size = len(dataset) // dataset_repeat
            logging.info(f"Initialized VideoTrainer with {original_size} videos (repeated {dataset_repeat}x = {len(dataset)} samples)")
            logging.info(f"Model: {config.model_name}")
            logging.info(f"Batch size: {config.batch_size}")
            logging.info(f"Learning rate: {config.lr}")
            logging.info(f"Device: {self.device}")
            logging.info(f"Mixed precision: {config.mixed_precision}")
            logging.info(f"Experiment directory: {self.exp_dir}")

    def save(self, epoch=None):
        if self.is_main_process:
            if "no_save" in self.config and self.config.no_save:
                logging.info("Skipping checkpoint save (no_save=True)")
                return
            logging.info("Saving model checkpoint...")
            # Unwrap model for saving
            unwrapped_generator = self.accelerator.unwrap_model(self.regression_model.generator)
            
            # Use experiment directory if available
            output_folder = self.exp_dir
            
            # Save with epoch info only
            if epoch is not None:
                checkpoint_dir = os.path.join(output_folder, f"checkpoint_epoch_{epoch:03d}")
                filename = f"model_epoch_{epoch:03d}.pt"
            else:
                checkpoint_dir = os.path.join(output_folder, "checkpoint_final")
                filename = "model.pt"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save state dict
            # Handle DeepSpeed optimizer separately
            optimizer_state = None
            logging.info(f"no save optimizer state now")
            model_state_dict = unwrapped_generator.state_dict()
            logging.info(f"parameters saving at epoch {epoch}: {model_state_dict[list(model_state_dict.keys())[0]].flatten()[:10]}")
            state_dict = {
                "generator": model_state_dict,
                "epoch": epoch,
                "config": dict(self.config)
            }
            if optimizer_state is not None:
                state_dict["optimizer"] = optimizer_state
            
            torch.save(state_dict, os.path.join(checkpoint_dir, filename))
            logging.info(f"Model saved to {os.path.join(checkpoint_dir, filename)}")

    def _resume_training_state(self, checkpoint_path):
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu",weights_only=False)
        unwrapped_generator = self.accelerator.unwrap_model(self.regression_model.generator)
        missing = unwrapped_generator.load_state_dict(checkpoint["generator"], strict=False)
        print(f"Loading checkpoint from {checkpoint_path} completed with missing: {missing}")
        if "optimizer" in checkpoint:
            optimizer_state = checkpoint["optimizer"]
            # try:
            self._load_optimizer_state(optimizer_state)
            # except Exception as e:
            #     logging.warning("Failed to load optimizer, skipping. Error: " + str(e))
        else:
            logging.info("No optimizer state found in checkpoint, skipping optimizer load.")
        saved_epoch = checkpoint.get("epoch")
        self.start_epoch = 0 if saved_epoch is None else saved_epoch + 1
        steps_per_epoch = len(self.dataloader)
        self.initial_global_step = self.start_epoch * steps_per_epoch
        self.step = self.initial_global_step
        logging.info(f"Resumed training from epoch {self.start_epoch} (0-based)")

    def _load_optimizer_state(self, optimizer_state):
        """Best-effort load for standard and DeepSpeed-style optimizer states."""
        if isinstance(optimizer_state, dict) and "param_groups" in optimizer_state and "state" in optimizer_state:
            try:
                self.optimizer.load_state_dict(optimizer_state)
                logging.info("Loaded optimizer state into Accelerator-wrapped optimizer")
            except:
                logging.warning("Failed to load optimizer state into wrapped optimizer)")
            try:
                self.optimizer_unwrap.load_state_dict(optimizer_state)
                logging.info("Loaded optimizer state into unwrapped optimizer")
            except:
                logging.warning("Failed to load optimizer state into unwrapped optimizer)")
            return True

        if isinstance(optimizer_state, dict) and "base_optimizer_state" in optimizer_state:
            base_state = optimizer_state["base_optimizer_state"]
            if isinstance(base_state, dict) and "param_groups" in base_state and "state" in base_state:
                target_groups = len(self.optimizer.param_groups)
                source_groups = len(base_state["param_groups"])
                if source_groups == target_groups:
                    try:
                        self.optimizer.load_state_dict(base_state)
                        logging.info("Loaded DeepSpeed optimizer state into Accelerator-wrapped optimizer")
                    except:
                        logging.warning("Failed to load DeepSpeed optimizer state into wrapped optimizer)")
                    try:
                        self.optimizer_unwrap.load_state_dict(base_state)
                        logging.info("Loaded DeepSpeed optimizer state into unwrapped optimizer")
                    except:
                        logging.warning("Failed to load DeepSpeed optimizer state into unwrapped optimizer)")
                    return True
        return False

    def train_one_step(self, batch):
        """Train on a single batch"""
        self.regression_model.generator.eval()  # prevent any randomness (e.g. dropout)
        self.regression_model.text_encoder.eval()

        text_prompts = batch["prompts"]
        target_latent = batch["video_latent"]

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.regression_model.text_encoder(
                text_prompts=text_prompts)

        # Step 3: Train the generator using video regression loss
        with self.accelerator.accumulate(self.regression_model.generator):
            generator_loss, log_dict = self.regression_model.video_loss(
                conditional_dict=conditional_dict,
                target_latent=target_latent
            )

            # Backpropagation
            self.accelerator.backward(generator_loss)
            
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(
                    self.regression_model.generator.parameters(), 
                    self.max_grad_norm
                )
            
            self.optimizer.step()
            self.optimizer.zero_grad()

        # Store loss for progress bar display
        loss_value = generator_loss.item()

        # Step 4: Logging
        if self.is_main_process:
            log_data = {
                "generator_loss": loss_value,
                "target_latent_mean": log_dict["target_latent_mean"].item(),
                "target_latent_std": log_dict["target_latent_std"].item(),
                "pred_latent_mean": log_dict["pred_latent_mean"].item(),
                "pred_latent_std": log_dict["pred_latent_std"].item(),
                "step": self.step,
                "epoch": self.step // len(self.dataloader)
            }
            
            # Add timestep breakdown
            unnormalized_loss = log_dict["unnormalized_loss"]
            timestep = log_dict["timestep"]
            
            loss_breakdown = defaultdict(list)
            for index, t in enumerate(timestep):
                loss_breakdown[str(int(t.item()) // 250 * 250)].append(
                    unnormalized_loss[index].item())

            for key_t in loss_breakdown.keys():
                log_data[f"loss_at_time_{key_t}"] = sum(loss_breakdown[key_t]) / len(loss_breakdown[key_t])

            if self.wandb_run is not None:
                self.wandb_run.log(log_data, step=self.step)
            
            # Reduced logging frequency for cleaner output with tqdm
            if self.step % 50 == 0:
                logging.info(f"Step {self.step}: Loss = {loss_value:.6f}")
        
        return loss_value

    def train(self):
        num_epochs = self.config["num_epochs"]
        steps_per_epoch = len(self.dataloader)
        global_step = self.initial_global_step
        
        if self.start_epoch >= num_epochs:
            if self.is_main_process:
                logging.info("Checkpoint epoch is greater than or equal to total epochs. Nothing to train.")
            return
        
        if self.is_main_process:
            logging.info(f"Starting training for {num_epochs} epochs")
            if self.start_epoch > 0:
                logging.info(f"Resuming from epoch {self.start_epoch+1}")
            
        # Create overall training progress bar for main process only
        if self.is_main_process:
            total_steps = num_epochs * steps_per_epoch
            overall_pbar = tqdm(
                total=total_steps, 
                desc="Training Progress", 
                position=0,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
            if global_step > 0:
                overall_pbar.update(global_step)

        self.save(self.start_epoch-1)  # Save initial checkpoint if resuming
        for epoch in range(self.start_epoch, num_epochs):
            if self.is_main_process:
                logging.info(f"Starting epoch {epoch+1}/{num_epochs}")
                
                # Create epoch progress bar for main process only
                epoch_pbar = tqdm(
                    self.dataloader, 
                    desc=f"Epoch {epoch+1}/{num_epochs}",
                    position=1,
                    leave=False,
                    bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}'
                )
                dataloader_iter = epoch_pbar
            else:
                dataloader_iter = self.dataloader
                
            epoch_losses = []
            epoch_start_time = time.time()
            
            for step, batch in enumerate(dataloader_iter):
                self.step = global_step
                step_start_time = time.time()
                
                with self.accelerator.autocast():
                    loss_value = self.train_one_step(batch)
                
                step_end_time = time.time()
                step_time = step_end_time - step_start_time
                
                if self.is_main_process:
                    # Track epoch losses
                    epoch_losses.append(loss_value)
                    avg_loss = sum(epoch_losses) / len(epoch_losses)
                    
                    # Update epoch progress bar with metrics
                    epoch_pbar.set_postfix({
                        'Loss': f'{loss_value:.6f}',
                        'Avg': f'{avg_loss:.6f}',
                        'Time': f'{step_time:.2f}s'
                    })
                    
                    # Update overall progress bar
                    overall_pbar.update(1)
                    overall_pbar.set_postfix({
                        'Epoch': f'{epoch+1}/{num_epochs}',
                        'Loss': f'{loss_value:.6f}',
                        'Step': f'{step+1}/{steps_per_epoch}'
                    })
                    
                    # Log timing
                    current_time = time.time()
                    if self.previous_time is None:
                        self.previous_time = current_time
                    else:
                        time_per_step = current_time - self.previous_time
                        log_data = {"time_per_step": time_per_step}
                        
                        if self.wandb_run is not None:
                            self.wandb_run.log(log_data, step=self.step)
                        self.previous_time = current_time
                        
                global_step += 1
                
            # Close epoch progress bar and log summary
            if self.is_main_process:
                epoch_pbar.close()
                
                # Log epoch summary
                epoch_end_time = time.time()
                epoch_duration = epoch_end_time - epoch_start_time
                avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
                
                tqdm.write(f"✅ Epoch {epoch+1} completed in {epoch_duration:.2f}s | Avg Loss: {avg_epoch_loss:.6f}")
                logging.info(f"Epoch {epoch+1} completed in {epoch_duration:.2f}s, Average Loss: {avg_epoch_loss:.6f}")
                
            # Save checkpoint at the end of each epoch
            self.save(epoch=epoch)
            
        # Close overall progress bar and log completion
        if self.is_main_process:
            overall_pbar.close()
            tqdm.write("🎉 Training completed successfully!")
            logging.info("Training completed successfully!")
            
        if self.wandb_run is not None:
            self.wandb_run.finish()


def main():
    parser = argparse.ArgumentParser(description="CausVid Video Regression Training")
    parser.add_argument("--config_path", type=str, default="configs/wan_causal_ode_finetune.yaml",
                        help="Path to configuration file")
    parser.add_argument("--no_save", action="store_true", help="Disable model saving")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with reduced dataset size")
    
    # Experiment management
    parser.add_argument("--experiment_dir", type=str, default="experiments",
                        help="Base directory for experiments")
    parser.add_argument("--experiment_name", type=str, default="causvid_finetune")
    
    # Commonly changed parameters
    parser.add_argument("--dataset_base_path", type=str, 
                        default="data/custom_sketch0618/trunc_compress81_sample",
                        help="Override dataset base path")
    parser.add_argument("--metadata_path", type=str, 
                        default="data/custom_sketch0618/metadata_detailed.csv",
                        help="Override metadata CSV path")
    parser.add_argument("--num_frames", type=int, default=81, help="Override number of frames per video")
    parser.add_argument("--height", type=int, default=480, help="Override video height")
    parser.add_argument("--width", type=int, default=832, help="Override video width")
    parser.add_argument("--batch_size", type=int, default=1, help="Override batch size")
    parser.add_argument("--lr", type=float, default=2.0e-06, help="Override learning rate")
    parser.add_argument("--wandb_name", type=str, default="causvid_finetune", help="Override wandb run name")
    parser.add_argument("--seed", type=int, default=0, help="Override random seed")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--dataset_repeat", type=int, default=100, help="Number of times to repeat the dataset per epoch")
    parser.add_argument("--generator_ckpt", type=str,
                        default="../pretrained/VideoSketcher/VideoSketcher-models/VideoSketcher_AR/AR_1.3B.pt")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data loader workers")
    args = parser.parse_args()

    if args.debug:
        args.no_wandb = True

    # Load config and override with command-line arguments
    config = OmegaConf.load(args.config_path)
    config.update({k: v for k, v in args.__dict__.items() if v is not None})

    # Setup experiment with enhanced logging and management
    exp_dir, wandb_run = setup_experiment(
        args, 
        base_output_dir=args.experiment_dir,
        project_name=args.experiment_name if not args.no_wandb else None,
        exp_name_func=generate_causvid_exp_name
    )
    
    # If WandB is disabled, set wandb_run to None
    if args.no_wandb:
        wandb_run = None
        
    # Update config with experiment directory
    config.output_folder = exp_dir
    
    logging.info("Starting CausVid video regression training...")
    logging.info(f"Experiment directory: {exp_dir}")
    logging.info(f"Dataset: {args.dataset_base_path}")
    logging.info(f"Dataset repeat: {args.dataset_repeat}")
    logging.info(f"Learning rate: {args.lr}")
    logging.info(f"Epochs: {args.num_epochs}")
    logging.info(f"Frames: {args.num_frames}")
    logging.info(f"Resolution: {args.height}x{args.width}")
        
    # Create trainer and start training
    trainer = VideoTrainer(config, exp_dir=exp_dir, wandb_run=wandb_run)
    
    trainer.train()
    
    # Log training completion
    logging.info("Training completed successfully")


if __name__ == "__main__":
    main() 