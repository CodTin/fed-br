"""Fed-BR 公共工具模块"""

__version__ = "0.1.0"

from .utils import get_device, setup_logger, unwrap_state_dict

__all__ = ["get_device", "setup_logger", "unwrap_state_dict"]
