"""
Key Phrase Extraction Engine for Research Documents
Extracts important phrases, entities, and concepts from academic text
"""

import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter, defaultdict
import re
import string

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
    NLTK_AVAILABLE = True
except:
    NLTK_AVAILABLE = False
    print("NLTK components not available. Some features may not work.")

class PhraseExtractor:
    def __init__(self):
        """Initialize the phrase extraction system"""
        self.tfidf_vectorizer = None
        self.phrase_patterns = self._define_phrase_patterns()
        self.domain_vocabularies = {}
        self.stop_phrases = set()
        
    def _define_phrase_patterns(self):
        """
        Define POS patterns for
