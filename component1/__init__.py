"""
Source Package Initialization

This file makes the 'src' directory a Python package.
It allows us to import modules from the src folder using statements like:
from src.data_loader import DataLoader

For this project, we keep it simple - just basic package information.
"""

# Package metadata
__version__ = "1.0.0"
__author__ = "Student Name"
__description__ = "Document Classification System - Source Package"

# Optional: You can import key classes here to make them easier to access
# This allows users to do: from src import DataLoader, DocumentClassifier
# Instead of: from src.data_loader import DataLoader

# Uncomment these lines if you want easier imports (optional):
# from .data_loader import DataLoader
# from .text_processor import TextProcessor  
# from .models import DocumentClassifier
# from .evaluator import ModelEvaluator

# For this beginner project, we'll keep imports explicit in main.py
# so students can clearly see where each class comes from
