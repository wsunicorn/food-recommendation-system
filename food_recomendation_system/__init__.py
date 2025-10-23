"""food_recomendation_system package"""

from .data_loader import load_data
from .content_based import ContentBasedRecommender

__all__ = ["load_data", "ContentBasedRecommender"]