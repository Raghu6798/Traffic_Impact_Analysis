import sys
from loguru import logger

logger.remove()

logger.add(
    sys.stdout, 
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{function}</cyan> - <level>{message}</level>"
)
