"""Token cost and context-window arithmetic. Author: Sandeep Belgavi.

Pure functions, no tokenizer needed: you pass token counts in. Prices are *inputs*. This lesson
never ships a vendor price list because prices change; use your provider's current price page.
"""

from __future__ import annotations

from dataclasses import dataclass, field

MILLION = 1_000_000


@dataclass(frozen=True)
class Pricing:
    """USD per one million tokens."""

    input_per_m: float
    output_per_m: float

    def cost(self, input_tokens: int, output_tokens: int = 0) -> float:
        return (input_tokens * self.input_per_m + output_tokens * self.output_per_m) / MILLION


# Clearly fake numbers so examples are reproducible. Not any vendor's price.
EXAMPLE_PRICING = Pricing(input_per_m=1.00, output_per_m=4.00)


@dataclass(frozen=True)
class ContextWindow:
    size: int
    reserved_for_output: int = 0

    @property
    def usable_for_input(self) -> int:
        return max(0, self.size - self.reserved_for_output)

    def fits(self, input_tokens: int) -> bool:
        return input_tokens <= self.usable_for_input

    def percent_used(self, input_tokens: int) -> float:
        return 100.0 * input_tokens / self.size


def turns_until_full(
    window: ContextWindow, fixed_prefix: int, tokens_per_turn: int
) -> int:
    """How many whole chat turns fit after ``fixed_prefix`` tokens (system prompt, tools).

    A turn is one user message plus the model's reply, both kept in the history.
    """
    if tokens_per_turn <= 0:
        raise ValueError("tokens_per_turn must be positive")
    room = window.usable_for_input - fixed_prefix
    return max(0, room // tokens_per_turn)


@dataclass(frozen=True)
class Step:
    """One model call inside an agent run.

    ``output_tokens`` is what the model writes (a tool call or the final answer);
    ``tool_result_tokens`` is what the tool returns and is appended before the next call.
    """

    output_tokens: int
    tool_result_tokens: int = 0


@dataclass(frozen=True)
class RunCost:
    calls: int
    input_tokens: int  # summed over all calls: every call resends the whole history
    output_tokens: int
    peak_context_tokens: int
    cost: float
    fits_window: bool
    per_call_input: list[int] = field(default_factory=list)


def agent_run_cost(
    prefix_tokens: int,
    steps: list[Step],
    pricing: Pricing,
    window: ContextWindow | None = None,
) -> RunCost:
    """Cost of one agent run.

    ``prefix_tokens`` = system prompt + tool schemas + the user message. Call *n* is billed for the
    prefix plus every output and tool result of calls 1..n-1, so input grows with each step
    (the sum grows roughly with the square of the number of steps).
    """
    if prefix_tokens < 0 or not steps:
        raise ValueError("need a non-negative prefix and at least one step")
    history = prefix_tokens
    per_call: list[int] = []
    out_total = 0
    for step in steps:
        per_call.append(history)
        out_total += step.output_tokens
        history += step.output_tokens + step.tool_result_tokens
    peak = per_call[-1] + steps[-1].output_tokens
    in_total = sum(per_call)
    fits = True if window is None else window.fits(peak)
    return RunCost(
        calls=len(steps),
        input_tokens=in_total,
        output_tokens=out_total,
        peak_context_tokens=peak,
        cost=pricing.cost(in_total, out_total),
        fits_window=fits,
        per_call_input=per_call,
    )
