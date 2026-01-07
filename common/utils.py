import os
from pathlib import Path

import torch
from loguru import logger
from torch import Tensor, nn
from torch.types import Device
from typing import cast

# 移除 loguru 默认的 stderr handler
logger.remove()

_logger_initialized = False


def setup_logger(
    log_level: str = "INFO",
    log_dir: str = "logs",
    console_enabled: bool = True,
    file_rotation: str = "10 MB",
    file_retention: int = 5,
    compress: bool = True,
) -> None:
    """
    配置 Loguru 日志系统,支持多级日志输出和文件轮转。

    Args:
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: 日志目录路径 (相对于项目根目录)
        console_enabled: 是否启用控制台输出
        file_rotation: 文件轮转大小 (例如: "10 MB", "500 MB")
        file_retention: 保留的备份文件数量 (例如: 5 表示保留最近 5 个文件)
        compress: 是否压缩旧日志文件 (zip 格式)

    日志文件结构:
        logs/
        ├── app.log          # INFO 及以上级别
        ├── errors.log       # ERROR 及以上级别
        └── debug.log        # DEBUG 及以上级别 (开发环境)
    """
    global _logger_initialized

    if _logger_initialized:
        return

    # 创建日志目录
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # 控制台输出格式 (彩色)
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    # 文件输出格式 (无彩色)
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function}:{line} - "
        "{message}"
    )

    # 1. 控制台 Sink (INFO 及以上,彩色输出)
    if console_enabled:
        logger.add(
            sink=lambda msg: print(msg, end=""),
            format=console_format,
            level=log_level,
            colorize=True,
            enqueue=True,
        )

    # 2. 应用日志 Sink (INFO 及以上)
    logger.add(
        sink=log_path / "app.log",
        format=file_format,
        level="INFO",
        rotation=file_rotation,
        retention=file_retention,
        compression="zip" if compress else None,
        encoding="utf-8",
        enqueue=True,
    )

    # 3. 错误日志 Sink (ERROR 及以上)
    logger.add(
        sink=log_path / "errors.log",
        format=file_format,
        level="ERROR",
        rotation=file_rotation,
        retention=file_retention,
        compression="zip" if compress else None,
        encoding="utf-8",
        enqueue=True,
    )

    # 4. 调试日志 Sink (DEBUG 及以上,仅开发环境)
    if os.getenv("DEBUG", "0").lower() in ("1", "true", "yes"):
        logger.add(
            sink=log_path / "debug.log",
            format=file_format,
            level="DEBUG",
            rotation=file_rotation,
            retention=file_retention,
            compression="zip" if compress else None,
            encoding="utf-8",
            enqueue=True,
        )

    _logger_initialized = True


def get_device(cuda_num: int = 0) -> Device:
    """自动检测并返回最优计算设备"""
    device = "cpu"

    if torch.mps.is_available():
        device = "mps"

    if torch.cuda.is_available():
        device = f"cuda:{cuda_num}"

    # 使用 logger 替代 print
    logger.info(f"Using device: {device}")

    return device


def unwrap_state_dict(model: nn.Module) -> dict[str, Tensor]:
    # NOTE: this is to return plain or unwrapped state_dict even if Opacus wrapped the model.
    if hasattr(model, "_module"):
        module = cast(nn.Module, model._module)
        return cast(dict[str, Tensor], module.state_dict())
    return cast(dict[str, Tensor], model.state_dict())
