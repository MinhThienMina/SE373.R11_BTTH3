# BTVN#3 - Flight Booking Agent

This project demonstrates a mock flight-booking agent with LangGraph-style
agent workflows and a safety harness.

## Structure

- `flight_agent.py`: complete original implementation.
- `tools/flight_tools.py`: mock backend and LangChain tools.
- `harness/`: constraints, completion sensor, permission guard, loop detection,
  and handoff modules.
- `agents/`: ReAct, Plan-then-Execute, and Hybrid graph builders.
- `main.py`: quick test/demo CLI.
- `benchmark.py`: evaluates all three patterns and writes `results.json`.

## Run

```bash
python main.py test
python main.py demo hybrid
python benchmark.py --n 40
```

The three evaluated patterns are:

- `react`
- `plan_execute`
- `hybrid`
