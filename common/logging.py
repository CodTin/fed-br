import logging


def configure_flwr_logging() -> None:
    """避免 Flower 日志向根 logger 传播导致的重复输出。"""
    flwr_logger = logging.getLogger("flwr")
    flwr_logger.propagate = False
