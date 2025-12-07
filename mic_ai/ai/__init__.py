"""
AI components for MIC AI: environment wrapper, adaptive agent, curiosity, and plotting helpers.
"""

from .ai_env import AiEnvConfig, MicAiAIEnv
from .curiosity import WorldModelCuriosity
from .simple_agent import ActorCriticAgent, SimpleAdaptiveAgent
from .ai_agent_ai_only import AiCurrentAgent
from .world_model import SimpleWorldModel
from .plots_ai import plot_ident_and_learning

__all__ = [
    "AiEnvConfig",
    "MicAiAIEnv",
    "WorldModelCuriosity",
    "ActorCriticAgent",
    "AiCurrentAgent",
    "SimpleAdaptiveAgent",
    "SimpleWorldModel",
    "plot_ident_and_learning",
]
