"""
AI components for MIC AI: environment wrapper, adaptive agent, curiosity, and plotting helpers.
"""

from .ai_env import AiEnvConfig, MicAiAIEnv
from .curiosity import SimpleCuriosityModule
from .simple_agent import SimpleAdaptiveAgent
from .plots_ai import plot_ident_and_learning

__all__ = ["AiEnvConfig", "MicAiAIEnv", "SimpleCuriosityModule", "SimpleAdaptiveAgent", "plot_ident_and_learning"]
