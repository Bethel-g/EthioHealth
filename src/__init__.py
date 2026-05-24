"""
EthioHealth-AI Package
Localized Clinical Decision Support System for ED Overcrowding
"""

__version__ = "1.0.0"
__author__ = "EthioHealth-AI Team"
__description__ = "ML-based LOS prediction for Ethiopian referral hospitals"

from . import data_generator
from . import preprocessing
from . import eda
from . import model_training

__all__ = ["data_generator", "preprocessing", "eda", "model_training"]
