"""Full-on risk model — paper-only experimental aggressive trading lab.

Three lanes:
  1. Council Lane (`council_trader.py`): 5+ NIM models vote on daily action from
     a constrained menu. Aggressive sizing. No defensive gates.
  2. Pure-API Lane (`pure_api_trader.py`): EVERY parameter from LLM each day.
     No human-set thresholds. Maximum agentic.
  3. All-In Lane (`all_in_trader.py`): Stack multiple strategy sleeves
     simultaneously, sized to consume the whole account at high concurrency.

Paper trading only. The framework deliberately removes most of the safety rails
from the main `turbo-carnival/experiments/` framework — the user has explicitly
waived the conservative-quant constraints for this lab.

Hard rules that remain (physics, not policy):
  - Cannot trade more contracts than account NAV can collateralize
  - Cannot have negative cash (would be a margin call IRL)
  - Cannot trade after account goes to zero
"""
