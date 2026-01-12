import logging


def configure_flwr_logging() -> None:
    """
    配置 Flower 日志系统，避免重复输出。

    Flower 框架默认会将日志同时输出到根 logger 和应用 logger，
    导致日志重复。此函数通过设置 flwr logger 的 propagate 为 False
    来避免这一问题。

    Note:
        此函数应在应用初始化时调用一次即可生效。

    Example:
        >>> configure_flwr_logging()
        >>> # 后续 Flower 相关日志将不会重复输出
    """
    flwr_logger = logging.getLogger("flwr")
    flwr_logger.propagate = False
