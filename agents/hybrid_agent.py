"""Hybrid agent pattern.

This pattern plans first, then uses a recovery/replan node when a tool step
fails.
"""

from flight_agent import build_plan_graph


def build_hybrid_agent(ctx):
    """Build the hybrid graph with local recovery enabled."""
    return build_plan_graph(ctx, replan=True)


__all__ = ["build_hybrid_agent"]
