# Power Grid Contingency Analysis System

## Overview
A complete, self-contained system for identifying critical failures in the IEEE 118-bus power grid. It performs both N-1 (Exhaustive) and N-2 (Heuristic) analysis.

## Features
* **Voltage Stability Cost Function:** $S(x) = 1.0 - V_{min}$.
* **Simulated Annealing:** Finds critical N-2 faults without brute force.
* **Auto-Calibration:** Generates a `benchmark_results.csv` file automatically.
* **High-Contrast Visualization:** Highlights critical failures in Red/Orange against a Silver grid.

## How to Run
1.  Run the build script to generate the system.
2.  `cd Grid_Contingency_Project`
3.  `python setup_and_run.py`
