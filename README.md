# Self-Driving Car (2D) — NEAT Neural Network

<img src="assets/demo.gif" width="700" alt="Demo" />

A 2D self-driving car simulation trained using **NEAT (NeuroEvolution of Augmenting Topologies)**.

---

## NEAT Overview

**NEAT**, introduced by *Kenneth O. Stanley*, evolves neural networks by:
- optimizing **connection weights**, and
- increasing **network complexity** over time by adding **nodes** and **connections**.

To stabilize learning, NEAT groups similar genomes into **species** using a genetic distance metric. This reduces destructive competition between structurally different networks and allows promising structural mutations to mature before being outcompeted.

This project uses the Python implementation **neat-python**.

---

## Getting Started
This project uses `uv python package manager`
Install `uv` using the official guide: [uv installation](https://docs.astral.sh/uv/getting-started/installation/#pypi)

---

## Run the Project

##### Windows (PowerShell)

```powershell
git clone https://github.com/AbhijeetRoy2107/Self-Driving-Car-2d.git
cd Self-Driving-Car-2d

uv venv .venv
.\.venv\Scripts\Activate.ps1

uv sync
python main.py #to train your model for the car
python play.py #(optional) to let the best trained model run the car
```
##### macOS / Linux
```
git clone https://github.com/AbhijeetRoy2107/Self-Driving-Car-2d.git
cd Self-Driving-Car-2d

uv venv .venv
source .venv/bin/activate

uv sync
python main.py
python play.py
```
---
## Dependencies
Dependencies are managed via `pyproject.toml` and locked in `uv.lock`.
The project uses (minimum versions):
```
requires-python = ">=3.13"
dependencies = [
  "neat-python>=1.1.0",
  "numpy>=2.4.1",
  "pygame>=2.6.1",
  "scipy>=1.17.0",
]

```
---

## Configuration
- **Simulation settings**: `config_variables.py`
(window size, sensor distance, camera behavior, scoring constants, etc.)

- **NEAT hyperparameters**: `config_file.txt`
(population size, mutation rates, species settings, compatibility threshold, etc.)