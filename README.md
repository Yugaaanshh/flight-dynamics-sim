# Roskam Flight Dynamics Simulator

A Python-based aircraft flight dynamics simulator implementing the theory from:
- **Roskam, "Airplane Flight Dynamics & Automatic Flight Controls"**
- **Zipfel, "Modeling and Simulation of Aerospace Vehicle Dynamics"**

## Features

- **Phase 1**: Linear 4-state longitudinal model (u, alpha, q, theta)
- **Phase 2**: Nonlinear 6-DOF rigid body model (12 states)
- **Phase 3**: General trim (level, turn, pull-up) + numerical linearization + modal analysis
- **Phase 4**: Control surface influence experiments (elevator, aileron, rudder power)
- **Multi-Plane**: YAML configuration for easy aircraft addition

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Run Phase 4 demo (control surface experiments)
python examples/phase4_demo.py

# Run Phase 3 demo (trim + linearization + modes)
python examples/phase3_demo.py
```

## Available Aircraft

Edit `planes.yaml` to add your own aircraft:

| Aircraft | Source | V_trim | Mass |
|----------|--------|--------|------|
| business_jet | Roskam App.B | 206 m/s | 7,257 kg |
| boeing_747 | Zipfel Ch10 | 250 m/s | 288,000 kg |
| cessna_172 | Estimated | 60 m/s | 1,043 kg |

## Project Structure

```
flight_dynamics_sim/
├── config.py              # Dataclasses for derivatives
├── config_loader.py       # YAML loader for multi-plane
├── planes.yaml            # Aircraft database
├── aero/coefficients.py   # CL, CD, Cm, CY, Cl, Cn
├── forces_moments.py      # Body-axis forces
├── eom/
│   ├── longitudinal.py    # Linear 4-state model
│   └── six_dof.py         # Nonlinear 12-state model
├── sim/trim_maneuver.py   # General trim solver
├── analysis/linearize.py  # Numerical Jacobians + modes
├── experiments/           # Control surface sweeps
└── examples/              # Demo scripts
```

## References

1. Roskam, J. "Airplane Flight Dynamics and Automatic Flight Controls" Parts I & II
2. Zipfel, P. "Modeling and Simulation of Aerospace Vehicle Dynamics" 3rd Ed
