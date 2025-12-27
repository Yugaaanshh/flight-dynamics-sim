# Roskam Flight Dynamics Simulator

**Complete 6-DOF aircraft simulator**: Linear → Nonlinear → Trim/Modes/Sensitivity → FCS PID.  
Multi-aircraft support (BizJet, B747, C172). Interactive Streamlit UI.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Quick Start

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py  # Opens localhost:8501
```

## 📋 Features

| Phase | Description |
|-------|-------------|
| **Phase 1** | Longitudinal linear 4-state model + aero coefficients |
| **Phase 2** | Full 6-DOF nonlinear rigid-body dynamics |
| **Phase 3** | General trim (level/turn/pullup) + modal analysis + sensitivity |
| **Phase 4** | Control power sweeps (elevator/aileron/rudder effectiveness) |
| **Phase 5** | PID Flight Control System (q-damper, roll-hold SAS) |

**Aircraft Database** (`planes.yaml`):
- Roskam Business Jet (400 kts, 35,000 ft)
- Boeing 747-400 (M=0.8, 40,000 ft)
- Cessna 172 Skyhawk (104 kts, 5,000 ft)

## 🖥️ Streamlit UI

Interactive dashboard with sliders for V/altitude/load factor:
- **Trim Tab**: Compute steady-state trim conditions
- **Modes Tab**: Eigenvalue analysis with damping/frequency charts
- **Sensitivity Tab**: Derivative scaling effects on modes
- **Response Tab**: Step response simulations

## 📊 Examples

```bash
# FCS open-loop vs closed-loop comparison
python examples/phase5_demo.py

# Validate all aircraft against Roskam expected ranges
python examples/verify_aircraft.py

# Multi-plane mode comparison
python examples/multiplane_demo.py
```

## 🛠️ Project Structure

```
flight_dynamics_sim/
├── config_loader.py      # YAML aircraft loader (Imperial→SI)
├── planes.yaml           # BizJet/B747/C172 database
├── streamlit_app.py      # Interactive UI
├── aero/
│   └── coefficients.py   # CL, CD, Cm, CY, Cl, Cn models
├── eom/
│   ├── longitudinal.py   # Linear 4-state model
│   ├── six_dof.py        # Nonlinear 12-state model
│   └── closed_loop.py    # FCS simulation wrapper
├── sim/
│   └── trim_maneuver.py  # General trim solver
├── analysis/
│   └── linearize.py      # Numerical Jacobians + modes
├── control/
│   └── fcs.py            # PID controllers + SAS
└── examples/
    ├── phase1_demo.py
    ├── phase2_demo.py
    ├── phase3_demo.py
    ├── phase4_demo.py
    ├── phase5_demo.py
    ├── multiplane_demo.py
    └── verify_aircraft.py
```

## 📚 References

1. Roskam, J. *Airplane Flight Dynamics and Automatic Flight Controls*, Parts I & II
2. Zipfel, P. *Modeling and Simulation of Aerospace Vehicle Dynamics*, 3rd Ed

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

**Contributions welcome!** ⭐ Star if useful.
