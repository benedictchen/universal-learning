"""
Solomonoff Induction Modular Components

Based on: Ray Solomonoff (1964) "A Formal Theory of Inductive Inference"
"""

from .solomonoff_config import *
from .solomonoff_inductor import SolomonoffInductor

__all__ = ['ComplexityMethod', 'CompressionAlgorithm', 'SolomonoffConfig', 'SolomonoffInductor']
