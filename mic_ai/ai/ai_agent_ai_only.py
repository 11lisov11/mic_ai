from __future__ import annotations

from mic_ai.ai.simple_agent import ActorCriticAgent


class AiCurrentAgent(ActorCriticAgent):
    """
    Thin alias over the generic ActorCriticAgent for the ai_current regime.
    Kept separate for clarity/extension in full AI-only control experiments.
    """


__all__ = ["AiCurrentAgent"]
