# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import sys
import gc
import logging
import math
import torch
import errno
from datetime import datetime
from pathlib import Path
from typing import Union, Tuple, List, Dict, Optional
from collections import defaultdict

from src import dist_utils

Number = Union[float, int]

logger = logging.getLogger(__name__)

DEVICE = dist_utils.DEVICE


def setup_logging(
    log_dir: str = "logs",
    log_prefix: str = "",
    log_filename: Optional[str] = None,
    level: Optional[int] = None,
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
    stdout_only: bool = False,
    distributed: bool = False,
    use_timestamp: bool = True,
) -> logging.Logger:
    """Setup comprehensive logging with optional distributed training support.

    This function provides flexible logging configuration for both standalone
    applications and distributed training scenarios.

    Args:
        log_dir: Directory to store log files. Will be created if it doesn't exist.
        log_prefix: Prefix for the log filename. Used when `use_timestamp=True`.
            The final filename will be `{log_prefix}_{timestamp}.log`.
        log_filename: Fixed log filename (e.g., "run.log"). If provided, overrides
            `log_prefix` and `use_timestamp`. Useful for distributed training.
        level: Logging level (e.g., logging.INFO, logging.DEBUG). If None and
            `distributed=True`, uses INFO for main process and WARN for others.
            If None and `distributed=False`, defaults to INFO.
        log_format: Custom log format string. If None, uses default format.
            For distributed training, defaults to:
            '[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s'
            Otherwise defaults to:
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        date_format: Custom date format string. If None, defaults to
            '%m/%d/%Y %H:%M:%S' for distributed mode, or standard format otherwise.
        stdout_only: If True, only log to stdout without creating a log file.
        distributed: If True, enables distributed training support:
            - Synchronizes with torch.distributed.barrier() if initialized
            - Uses dist_utils.is_main() to determine log level
        use_timestamp: If True, appends timestamp to log filename. Ignored if
            `log_filename` is provided.

    Returns:
        Logger instance configured with the specified handlers.

    Examples:
        # Standalone benchmark logging with timestamped files
        >>> logger = setup_logging(log_dir="logs", log_prefix="benchmark")
        >>> logger.info("Starting benchmark...")

        # Distributed training logging (replaces init_logger)
        >>> logger = setup_logging(
        ...     log_dir=args.output_dir,
        ...     log_filename="run.log",
        ...     distributed=True
        ... )
    """
    # Handle distributed training synchronization
    if distributed and torch.distributed.is_initialized():
        torch.distributed.barrier()

    # Determine log level
    if level is None:
        if distributed:
            level = logging.INFO if dist_utils.is_main() else logging.WARN
        else:
            level = logging.INFO

    # Set default format based on mode
    if log_format is None:
        if distributed:
            log_format = "[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s"
        else:
            log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Set default date format for distributed mode
    if date_format is None and distributed:
        date_format = "%m/%d/%Y %H:%M:%S"

    # Determine log file path
    if not stdout_only:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        if log_filename:
            log_file = Path(log_dir) / log_filename
        elif use_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = Path(log_dir) / f"{log_prefix}_{timestamp}.log"
        else:
            log_file = Path(log_dir) / f"{log_prefix}.log"

    # Configure handlers
    handlers = [logging.StreamHandler(sys.stdout)]
    if not stdout_only:
        handlers.append(logging.FileHandler(log_file))

    # Configure logging
    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
        force=True,  # Override any existing configuration
    )

    # Get logger and log initial message
    module_logger = logging.getLogger(__name__)
    if not stdout_only and (not distributed or dist_utils.is_main()):
        module_logger.info(f"Logging initialized - Log file: {log_file}")

    return module_logger


def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e


def save(model, optimizer, scheduler, step, opt, dir_path, name):
    model_to_save = model.module if hasattr(model, "module") else model
    path = os.path.join(dir_path, "checkpoint")
    epoch_path = os.path.join(path, name)  # "step-%s" % step)
    os.makedirs(epoch_path, exist_ok=True)
    cp = os.path.join(path, "latest")
    fp = os.path.join(epoch_path, "checkpoint.pt")
    checkpoint = {
        "step": step,
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "opt": opt,
    }
    torch.save(checkpoint, fp)
    symlink_force(epoch_path, cp)
    if not name == "lastlog":
        logger.info(f"Saving model to {epoch_path}")


def load(model_class, dir_path, opt, reset_params=False):
    epoch_path = os.path.realpath(dir_path)
    checkpoint_path = os.path.join(epoch_path, "checkpoint.pt")
    logger.info(f"loading checkpoint {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    opt_checkpoint = checkpoint["opt"]
    state_dict = checkpoint["model"]

    model = model_class(opt_checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(DEVICE)
    step = checkpoint["step"]
    if not reset_params:
        optimizer, scheduler = set_optim(opt_checkpoint, model)
        scheduler.load_state_dict(checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        optimizer, scheduler = set_optim(opt, model)

    return model, optimizer, scheduler, opt_checkpoint, step


############ OPTIM


class WarmupLinearScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup, total, ratio, last_epoch=-1):
        self.warmup = warmup
        self.total = total
        self.ratio = ratio
        super(WarmupLinearScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup:
            return (1 - self.ratio) * step / float(max(1, self.warmup))

        return max(
            0.0,
            1.0 + (self.ratio - 1) * (step - self.warmup) / float(max(1.0, self.total - self.warmup)),
        )


class CosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup, total, ratio=0.1, last_epoch=-1):
        self.warmup = warmup
        self.total = total
        self.ratio = ratio
        super(CosineScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup:
            return float(step) / self.warmup
        s = float(step - self.warmup) / (self.total - self.warmup)
        return self.ratio + (1.0 - self.ratio) * math.cos(0.5 * math.pi * s)


def set_optim(opt, model):
    if opt.optim == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps, weight_decay=opt.weight_decay
        )
    else:
        raise NotImplementedError("optimizer class not implemented")

    scheduler_args = {
        "warmup": opt.warmup_steps,
        "total": opt.total_steps,
        "ratio": opt.lr_min_ratio,
    }
    if opt.scheduler == "linear":
        scheduler_class = WarmupLinearScheduler
    elif opt.scheduler == "cosine":
        scheduler_class = CosineScheduler
    else:
        raise ValueError
    scheduler = scheduler_class(optimizer, **scheduler_args)
    return optimizer, scheduler


def get_parameters(net, verbose=False):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    message = "[Network] Total number of parameters : %.6f M" % (num_params / 1e6)
    return message


class WeightedAvgStats:
    """provides an average over a bunch of stats"""

    def __init__(self):
        self.raw_stats: Dict[str, float] = defaultdict(float)
        self.total_weights: Dict[str, float] = defaultdict(float)

    def update(self, vals: Dict[str, Tuple[Number, Number]]) -> None:
        for key, (value, weight) in vals.items():
            self.raw_stats[key] += value * weight
            self.total_weights[key] += weight

    @property
    def stats(self) -> Dict[str, float]:
        return {x: self.raw_stats[x] / self.total_weights[x] for x in self.raw_stats.keys()}

    @property
    def tuple_stats(self) -> Dict[str, Tuple[float, float]]:
        return {x: (self.raw_stats[x] / self.total_weights[x], self.total_weights[x]) for x in self.raw_stats.keys()}

    def reset(self) -> None:
        self.raw_stats = defaultdict(float)
        self.total_weights = defaultdict(float)

    @property
    def average_stats(self) -> Dict[str, float]:
        keys = sorted(self.raw_stats.keys())
        if torch.distributed.is_initialized():
            torch.distributed.broadcast_object_list(keys, src=0)
        global_dict = {}
        for k in keys:
            if not k in self.total_weights:
                v = 0.0
            else:
                v = self.raw_stats[k] / self.total_weights[k]
            v, _ = dist_utils.weighted_average(v, self.total_weights[k])
            global_dict[k] = v
        return global_dict


def load_hf(object_class, model_name):
    try:
        obj = object_class.from_pretrained(model_name, local_files_only=True)
    except:
        obj = object_class.from_pretrained(model_name, local_files_only=False)
    return obj


def init_tb_logger(output_dir):
    try:
        from torch.utils import tensorboard

        if dist_utils.is_main():
            tb_logger = tensorboard.SummaryWriter(output_dir)
        else:
            tb_logger = None
    except:
        logger.warning("Tensorboard is not available.")
        tb_logger = None

    return tb_logger


def clear_cache(device: str | torch.device = DEVICE):
    gc.collect()
    if isinstance(device, str) and device != "cpu":
        getattr(torch, device).empty_cache()
    elif isinstance(device, torch.device) and device.type != "cpu":
        getattr(torch, device.type).empty_cache()