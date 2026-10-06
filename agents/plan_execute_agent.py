"""Plan-then-Execute agent pattern."""

from flight_agent import build_plan_graph


def build_plan_execute_agent(ctx):
    """Build the plan-then-execute graph without local replanning."""
    return build_plan_graph(ctx, replan=False)


__all__ = ["build_plan_execute_agent"]
