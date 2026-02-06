import torch, os, sys, logging

from pipeline.wan_video import WanVideoPipeline
from utils.trainer_utils import DiffusionTrainingModule
from utils.trainer_utils import VideoDataset, launch_training_task, wan_parser
from utils.experiment_utils import setup_experiment, save_code_snapshot
from utils.trainer_utils import EnhancedModelLogger
from utils.latent_precompute import precompute_wan_latents

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def generate_wan_exp_name(args):
    components = []

    # Model name
    if args.model_id_with_origin_paths:
        model_id = args.model_id_with_origin_paths.split(":")[0].split("/")[-1]
        components.append(model_id)

    # LoRA info
    if args.lora_base_model:
        components.append(f"lora_r{args.lora_rank}")
        if args.lora_checkpoint:
            components.append("ft")

    # Training info
    components.append(f"lr{args.learning_rate:.0e}".replace('-0', '-'))
    components.append(f"ep{args.num_epochs}")

    return [c for c in components if c]



class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        diffusion_loss_weight=1.0,
    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=model_configs,
        )
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
        self.device = device
        self.pipe.device = device
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )
        self.diffusion_loss_weight = diffusion_loss_weight
        self.pipe.diffusion_loss_weight = diffusion_loss_weight

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

    def forward_preprocess(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        
        precompute_latents = data.get("precompute_latents")
        input_video = data.get("video")
        if input_video is not None:
            height = input_video[0].size[1]
            width = input_video[0].size[0]
            num_frames = len(input_video)
        elif precompute_latents is not None:
            latent_length = precompute_latents.shape[2]
            height = precompute_latents.shape[3] * self.pipe.vae.upsampling_factor
            width = precompute_latents.shape[4] * self.pipe.vae.upsampling_factor
            num_frames = (latent_length - 1) * 4 + 1
        else:
            raise ValueError("Either input_video or precompute_latents must be provided.")

        inputs_shared = {
            "input_video": input_video,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }

        # Extra inputs
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                if input_video is not None:
                    inputs_shared["input_image"] = input_video[0]
                else:
                    inputs_shared["input_image"] = data["input_image"]
            elif extra_input == "end_image":
                if input_video is not None:
                    inputs_shared["end_image"] = input_video[-1]
                else:
                    inputs_shared["end_image"] = data["end_image"]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                ref_data = data[extra_input]
                inputs_shared[extra_input] = ref_data[0] if isinstance(ref_data, list) else ref_data
            else:
                inputs_shared[extra_input] = data[extra_input]
        if precompute_latents is not None:
            inputs_shared["precompute_latents"] = precompute_latents
        
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        outputs = {**inputs_shared, **inputs_posi}
        return outputs
    
    
    def forward(self, data, inputs=None, forced_timestep_idx=None):
        if inputs is None:
            inputs = self.forward_preprocess(data)
        if forced_timestep_idx is not None:
            inputs["forced_timestep_idx"] = int(forced_timestep_idx)
        inputs["sample_id"] = data["sample_id"]
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    def subdir_name_func(args):
        if args.max_timestep_boundary != 1.0 or args.min_timestep_boundary != 0.0:
            return f"ts{int(args.min_timestep_boundary*1000)}-{int(args.max_timestep_boundary*1000)}"
        return ""
    
    exp_dir, wandb_run = setup_experiment(
        args, 
        base_output_dir=args.experiment_dir,
        project_name=args.wandb_project if not args.no_wandb else None,
        exp_name_func=generate_wan_exp_name,
        subdir_name_func=subdir_name_func
    )

    project_root = os.path.dirname(__file__)
    key_code_files = [
        os.path.abspath(__file__),
        os.path.join(project_root, "utils", "experiment_utils.py"),
        os.path.join(project_root, "utils", "trainer_utils.py"),
        os.path.join(project_root, "utils", "latent_precompute.py"),
        os.path.join(project_root, "pipeline", "wan_video.py"),
    ]
    save_code_snapshot(exp_dir, key_code_files, source_root=project_root)
    if wandb_run is not None:
        wandb_run.config.update({"log_diffusion_loss": True}, allow_val_change=True)
    
    logging.info("Starting WAN video training...")
    logging.info(f"Experiment directory: {exp_dir}")
    logging.info(f"Dataset: {args.dataset_base_path}")
    logging.info(f"LoRA rank: {args.lora_rank}")
    logging.info(f"Learning rate: {args.learning_rate}")
    logging.info(f"Epochs: {args.num_epochs}")
    if args.resume:
        logging.info(f"Resume from checkpoint: {args.resume}")
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        diffusion_loss_weight=args.diffusion_loss_weight,
    )
    trainable_names = model.trainable_param_names()
    logging.info(f"Trainable modules: {trainable_names}")
    # Create dataset and precompute latents
    dataset = VideoDataset(args=args, repeat=1)
    dataset = precompute_wan_latents(dataset, model, args)
    print(dataset[0])
    logging.info(f"Dataset size: {len(dataset)} samples")

    model_logger = EnhancedModelLogger(
        os.path.join(args.output_path, "ckpt"),
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        wandb_run=wandb_run,
        visual_log_interval=args.visual_log_interval,
    )

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    launch_training_task(
        dataset, model, model_logger,
        optimizer=optimizer, scheduler=scheduler, num_workers=args.dataset_num_workers,
        save_steps=args.save_steps,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        find_unused_parameters=args.find_unused_parameters,
        resume_from_checkpoint=args.resume,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    
    summary = model_logger.get_training_summary()
    logging.info(f"Training completed. Summary: {summary}")
    
    if wandb_run is not None:
        wandb_run.log({"training/completed": True})
        wandb_run.finish()
