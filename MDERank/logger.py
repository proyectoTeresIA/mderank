# logger.py
import logging
import sys

def create_logger(name: str = "mderank", level: str = "INFO") -> logging.Logger:
    """
    Crea un logger que imprime solo por pantalla.
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        formatter = logging.Formatter(
            fmt="%(asctime)s — %(levelname)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
