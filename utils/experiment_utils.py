import os
import sys
import logging
import json
import socket
import shutil
from pathlib import Path
from datetime import datetime
import torch
import atexit
import time
import wandb

class StreamToLogger:
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.ERROR):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        # Only log non-empty lines
        stripped_buf = buf.rstrip()
        if stripped_buf:
            for line in stripped_buf.splitlines():
                self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        # Necessary for file-like object interface.
        pass

    def isatty(self):
        # Always return False since this is not a TTY
        return False


def init_logging(rank=0, exp_dir="output"):
    """
    Initializes logging to file and console (rank 0 only).
    Redirects stderr to the logger.

    Args:
        rank (int): The rank of the current process.
        exp_dir (str): The experiment directory where the log file will be saved.
    """
    log_file = os.path.join(exp_dir, f'run_rank{rank}.log')
    log_format = "[%(asctime)s] %(levelname)s: %(message)s"
    # Get root logger BEFORE setting basicConfig if handlers are added manually
    logger = logging.getLogger()
    # Set level for the logger itself (messages below this level are ignored)
    logger.setLevel(logging.INFO) # Set root logger level (can be overridden by handlers)

    formatter = logging.Formatter(log_format)

    # File handler for all ranks
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO) # Log INFO and above to file
    logger.addHandler(file_handler)

    if rank == 0:
        # set format for console
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.INFO) # Log INFO and above to console for rank 0
        logger.addHandler(stream_handler)
        # Root logger level is already INFO
    else:
        # Other ranks log less verbosely to console (only ERROR and above)
        # We don't add a StreamHandler for non-rank 0 to keep console clean
        # File handler already added above logs INFO+
        pass # No console handler for non-rank 0

    # Redirect stderr to logger
    # Ensure stderr_logger is only created once per process, not per call to init_logging
    if not isinstance(sys.stderr, StreamToLogger):
        stderr_logger = StreamToLogger(logger, logging.ERROR)
        sys.stderr = stderr_logger
    elif sys.stderr.logger is not logger:  # If stderr is already redirected, make sure it points to the current logger
        sys.stderr.logger = logger


def generate_exp_name_generic(time_format="%Y-%m-%d_%H-%M-%S", custom_name_func=None, args=None):
    """
    Generate experiment name with optional custom naming function.
    
    Args:
        time_format (str): Time format string for datetime.strftime
        custom_name_func (callable): Optional function to generate custom name components
        args: Optional arguments to pass to custom_name_func
        
    Returns:
        str: Generated experiment name
    """

    components = []
    # Use custom naming function if provided
    if custom_name_func is not None and args is not None:
        custom_components = custom_name_func(args)
        if isinstance(custom_components, list):
            components.extend(custom_components)
        elif isinstance(custom_components, str):
            components.append(custom_components)
    formatted_time = datetime.now().strftime(time_format)
    components.append(formatted_time)
    
    return "_".join(components)


def setup_experiment_directory(base_output_dir: str, exp_name: str = None, name_func=None, args=None):
    """
    Sets up the experiment directory with automatic naming.

    Args:
        base_output_dir (str): The base directory for outputs.
        exp_name (str): Optional experiment name. If None, will generate using name_func or default.
        name_func (callable): Optional function to generate experiment name from args.
        args (argparse.Namespace): Optional parsed command-line arguments.

    Returns:
        str: The path to the created experiment directory.
    """
    if exp_name is None:
        if name_func is not None and args is not None:
            exp_name = generate_exp_name_generic(custom_name_func=name_func, args=args)
        else:
            exp_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    exp_dir = os.path.join(base_output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    logging.info(f"Created experiment directory: {exp_dir}")
    return exp_dir


def save_args(args, exp_dir):
    """
    Saves command-line arguments to a JSON file in the experiment directory.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        exp_dir (str): The experiment directory.
    """
    args_save_path = os.path.join(exp_dir, 'args.json')
    args_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    with open(args_save_path, 'w') as f:
        json.dump(args_dict, f, indent=4, ensure_ascii=False)
    logging.info(f"Saved arguments to {args_save_path}")


def save_env_info(exp_dir, device=None):
    """
    Saves environment information (hostname, GPU, SLURM ID) to a JSON file.

    Args:
        exp_dir (str): The experiment directory.
        device (optional): The torch device (e.g., 'cuda:0') or device index (e.g., 0)
                           to get GPU info for. Defaults to None.
    """
    env_info = {
        "hostname": socket.gethostname(),
    }
    if torch.cuda.is_available():
        gpu_info_device = device
        env_info['gpu_model'] = torch.cuda.get_device_name(gpu_info_device)
        env_info['gpu_memory'] = f"{torch.cuda.get_device_properties(gpu_info_device).total_memory // 1024**3}GB"
    else:
        env_info['gpu_model'] = "CUDA not available"

    slurm_job_id = os.getenv('SLURM_JOB_ID')
    if slurm_job_id:
        env_info['slurm_job_id'] = slurm_job_id
        # save a file called slurm id
        with open(os.path.join(exp_dir, f'slurm_id-{slurm_job_id}'), 'w') as f:
            f.write("")
    
    env_info['launch_command'] = ' '.join(sys.argv)

    env_info_save_path = os.path.join(exp_dir, 'env_info.json')
    with open(env_info_save_path, 'w') as f:
        json.dump(env_info, f, indent=4)
    logging.info(f"Saved environment info to {env_info_save_path}")


def save_code_snapshot(exp_dir, source_files, source_root=None):
    """
    Save selected source files into a code snapshot directory under the experiment.
    """
    if not source_files:
        return
    code_dir = os.path.join(exp_dir, "code")
    os.makedirs(code_dir, exist_ok=True)
    normalized_root = os.path.abspath(source_root) if source_root is not None else None
    copied = 0
    seen = set()
    for file_path in source_files:
        if file_path is None:
            continue
        abs_path = os.path.abspath(file_path)
        if abs_path in seen:
            continue
        seen.add(abs_path)
        if normalized_root is not None and abs_path.startswith(normalized_root):
            rel_path = os.path.relpath(abs_path, normalized_root)
        else:
            rel_path = os.path.basename(abs_path)
        dest_path = os.path.join(code_dir, rel_path)
        dest_dir = os.path.dirname(dest_path)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(abs_path, dest_path)
        copied += 1
    logging.info(f"Saved {copied} code files to {code_dir}")


def copy_slurm_logs(exp_dir):
    """
    Copies SLURM output and error logs (if found) to the experiment directory.

    Args:
        exp_dir (str): The experiment directory.
    """
    slurm_log_output = os.getenv('SLURM_LOG_OUTPUT')
    slurm_log_error = os.getenv('SLURM_LOG_ERROR')

    copied_files = []

    if slurm_log_output and os.path.exists(slurm_log_output):
        dest_log = os.path.join(exp_dir, os.path.basename(slurm_log_output))
        if os.path.exists(dest_log):
            os.remove(dest_log)
        shutil.copy2(slurm_log_output, dest_log)
        logging.info(f"Copied SLURM output log to {dest_log}")
        copied_files.append(dest_log)
    elif slurm_log_output:
        logging.warning(f"SLURM output log not found at {slurm_log_output}")

    if slurm_log_error and os.path.exists(slurm_log_error):
        if slurm_log_output and os.path.abspath(slurm_log_error) == os.path.abspath(slurm_log_output):
            logging.info("SLURM error log is the same as output log, skipping duplicate copy.")
        else:
            dest_err = os.path.join(exp_dir, os.path.basename(slurm_log_error))
            if os.path.exists(dest_err):
                os.remove(dest_err)
            shutil.copy2(slurm_log_error, dest_err)
            logging.info(f"Copied SLURM error log to {dest_err}")
            copied_files.append(dest_err)
    elif slurm_log_error:
        logging.info(f"SLURM error log not found at {slurm_log_error} (this might be normal)")

    return copied_files # Return list of successfully copied files (optional) 


def link_slurm_logs(exp_dir):
    """
    Links SLURM output and error logs (if found) to the experiment directory.

    Args:
        exp_dir (str): The experiment directory.
    """
    slurm_log_output = os.getenv('SLURM_LOG_OUTPUT')
    slurm_log_error = os.getenv('SLURM_LOG_ERROR')

    linked_files = []

    if slurm_log_output and os.path.exists(slurm_log_output):
        dest_log = os.path.join(exp_dir, os.path.basename(slurm_log_output))
        if os.path.islink(dest_log):
            os.unlink(dest_log)
        elif os.path.exists(dest_log):
            os.remove(dest_log)
        os.symlink(os.path.abspath(slurm_log_output), dest_log)
        linked_files.append(dest_log)
        logging.info(f"Linked SLURM output log to {dest_log}")

    if slurm_log_error and os.path.exists(slurm_log_error):
        if not (slurm_log_output and os.path.abspath(slurm_log_error) == os.path.abspath(slurm_log_output)):
            dest_err = os.path.join(exp_dir, os.path.basename(slurm_log_error))
            if os.path.islink(dest_err):
                os.unlink(dest_err)
            elif os.path.exists(dest_err):
                os.remove(dest_err)
            os.symlink(os.path.abspath(slurm_log_error), dest_err)
            linked_files.append(dest_err)
            logging.info(f"Linked SLURM error log to {dest_err}")

    return linked_files


def setup_slurm_log_copy_on_exit(exp_dir):
    """
    Sets up automatic copying of SLURM logs when the script exits.
    
    Args:
        exp_dir (str): The experiment directory.
    """
    def copy_logs_on_exit():
        logging.info("Copying SLURM logs on exit...")
        copy_slurm_logs(exp_dir)
    
    atexit.register(copy_logs_on_exit)


def init_wandb(args, exp_dir, project_name="wan_video_training"):
    """
    Initialize Weights & Biases for experiment tracking.
    
    Args:
        args: Parsed command-line arguments
        exp_dir (str): Experiment directory path
        project_name (str): WandB project name
        
    Returns:
        wandb run object or None if WandB is not available
    """
    if project_name is None:
        return None
    
    # Only initialize WandB on the main process (rank 0) in distributed training
    rank = int(os.getenv('LOCAL_RANK', 0))
    if rank != 0:
        logging.info(f"Rank {rank}: Skipping WandB initialization (only rank 0 should initialize WandB)")
        return None
        
    # Generate run name from experiment directory
    run_name = os.path.basename(exp_dir)

    # Prepare config from args
    config = {}
    for key, value in vars(args).items():
        # Convert Path objects and other non-serializable types
        if isinstance(value, Path):
            config[key] = str(value)
        elif isinstance(value, (str, int, float, bool, type(None))):
            config[key] = value
        else:
            config[key] = str(value)

    # Add some derived config
    config['exp_dir'] = exp_dir
    config['exp_name'] = run_name
    config['world_size'] = int(os.getenv('WORLD_SIZE', 1))
    config['local_rank'] = rank

    # Initialize wandb
    run = wandb.init(
        entity="video-sketch",
        project=project_name,
        name=run_name,
        config=config,
        dir=exp_dir,  # Save wandb files in experiment directory
        resume="allow",
    )

    logging.info(f"Initialized WandB run: {run.name}")
    return run

def connect_name(components):
    """
    Connect name components into a single string with appropriate separators.

    Args:
        components (list): List of name components (str or int)

    Returns:
        str: Connected name string
    """
    exp_name=""
    for component in components:
        if exp_name!="" and exp_name[-1]!="/":
            exp_name += f"_{component}"
        else:
            exp_name += f"{component}"
    return exp_name

def setup_experiment(args, base_output_dir="experiments", project_name=None, exp_name_func=None, subdir_name_func=None):
    """
    Complete experiment setup including directory creation, logging, SLURM handling, and WandB.
    
    Args:
        args: Parsed command-line arguments
        base_output_dir (str): Base directory for experiments
        project_name (str): WandB project name (None to disable WandB)
        exp_name_func (callable): Optional function to generate experiment name from args
        
    Returns:
        tuple: (exp_dir, wandb_run)
    """
    rank = int(os.getenv('LOCAL_RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    
    slurm_job_id = os.getenv('SLURM_JOB_ID')
    if slurm_job_id:
        # Use SLURM job ID as the deterministic component
        if exp_name_func is not None and args is not None:
            custom_components = exp_name_func(args)
            if isinstance(custom_components, list):
                exp_name=connect_name(custom_components)
                exp_name += f"_slurm{slurm_job_id}"
            elif isinstance(custom_components, str):
                exp_name = f"{custom_components}_slurm{slurm_job_id}"
            else:
                exp_name = f"slurm{slurm_job_id}"
        else:
            exp_name = f"slurm{slurm_job_id}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    else:
        deterministic_factors = []
        
        slurm_job_name = os.getenv('SLURM_JOB_NAME')
        if slurm_job_name and slurm_job_name != 'bash':
            deterministic_factors.append(slurm_job_name)
            
        # Use current date/hour as stable component (changes only every hour)
        stable_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
        deterministic_factors.append(stable_time)
        
        # Create deterministic name
        if exp_name_func is not None and args is not None:
            custom_components = exp_name_func(args)
            if isinstance(custom_components, list):
                components = custom_components + deterministic_factors
            elif isinstance(custom_components, str):
                components = [custom_components] + deterministic_factors
            else:
                components = deterministic_factors
        else:
            components = ['distributed'] + deterministic_factors

        exp_name = connect_name(components)
    
    exp_dir = os.path.join(base_output_dir, exp_name)

    subdir_name = subdir_name_func(args) if subdir_name_func is not None else ""
    if subdir_name != "":
        exp_dir = os.path.join(exp_dir, subdir_name)
        logging.info(f"Created experiment subdirectory: {exp_dir}")

    if rank == 0:
        os.makedirs(exp_dir, exist_ok=True)
        logging.info(f"Rank 0 created experiment directory: {exp_dir}")
    else:
        time.sleep(0.5)  # 500ms delay
        os.makedirs(exp_dir, exist_ok=True)
    
    # Initialize logging for all processes
    init_logging(rank=rank, exp_dir=exp_dir)
    
    # Only rank 0 saves experiment info and handles SLURM logs
    if rank == 0:
        save_args(args, exp_dir)
        save_env_info(exp_dir, device=0 if torch.cuda.is_available() else None)
        
        # Handle SLURM logs
        link_slurm_logs(exp_dir)  # Link logs at start
        setup_slurm_log_copy_on_exit(exp_dir)  # Copy logs on exit
    
    # Initialize WandB (only on rank 0, as handled in init_wandb)
    if args.debug or args.no_wandb:
        wandb_run = None
    else:
        wandb_run = init_wandb(args, exp_dir, project_name)
    
    # Update args.output_path to use the new experiment directory
    args.output_path = exp_dir
    
    logging.info(f"Experiment setup complete. Directory: {exp_dir} (rank {rank}/{world_size})")
    
    return exp_dir, wandb_run
