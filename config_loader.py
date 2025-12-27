"""
YAML Configuration Loader for Multi-Plane Support

Loads aircraft configurations from YAML files enabling easy addition
of new aircraft without code changes.

Features:
- Load aircraft by name from planes.yaml
- Validate required derivative keys
- Convert to internal dataclass format for all phases
- Support for both longitudinal and lateral-directional derivatives

Usage:
    from config_loader import load_aircraft, get_available_aircraft
    
    config = load_aircraft('boeing_747')
    model = SixDoFModel(config)

References:
    Roskam, "Airplane Flight Dynamics" - derivative definitions
    Zipfel, "Modeling and Simulation" Ch10 - Boeing 747 data
"""

import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    LongitudinalDerivatives, LateralDerivatives,
    AircraftParameters, FullAircraftConfig
)


# Required derivative keys for validation
REQUIRED_LONG_KEYS = ['Xu', 'Xa', 'Zu', 'Za', 'Zq', 'Mu', 'Ma', 'Mq', 'Mde', 'Zde', 'Xde']
REQUIRED_LAT_KEYS = ['Yb', 'Yp', 'Yr', 'Lb', 'Lp', 'Lr', 'Lda', 'Ldr', 'Nb', 'Np', 'Nr', 'Nda', 'Ndr', 'Yda', 'Ydr']


def get_yaml_path() -> Path:
    """Get path to planes.yaml in same directory as this module."""
    return Path(__file__).parent / 'planes.yaml'


def get_available_aircraft() -> List[str]:
    """Return list of available aircraft names from planes.yaml."""
    yaml_path = get_yaml_path()
    if not yaml_path.exists():
        return []
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    return list(data.keys()) if data else []


def validate_config(name: str, config: Dict) -> List[str]:
    """
    Validate aircraft configuration has required keys.
    
    Returns list of missing keys (empty if valid).
    """
    missing = []
    
    # Check basic parameters
    required_params = ['mass', 'Ixx', 'Iyy', 'Izz', 'S', 'cbar', 'b']
    for key in required_params:
        if key not in config:
            missing.append(f"params.{key}")
    
    # Check longitudinal derivatives
    if 'derivs_long' in config:
        for key in REQUIRED_LONG_KEYS:
            if key not in config['derivs_long']:
                # Some keys have aliases
                aliases = {
                    'Xa': ['X_alpha', 'Xalpha'],
                    'Za': ['Z_alpha', 'Zalpha'],
                    'Ma': ['M_alpha', 'Malpha'],
                    'Mde': ['M_delta_e', 'Mde'],
                    'Zde': ['Z_delta_e', 'Zde'],
                    'Xde': ['X_delta_e', 'Xde']
                }
                found = False
                if key in aliases:
                    for alias in aliases[key]:
                        if alias in config['derivs_long']:
                            found = True
                            break
                if not found and key not in config['derivs_long']:
                    missing.append(f"derivs_long.{key}")
    else:
        missing.append("derivs_long")
    
    # Lateral derivatives optional but check if present
    if 'derivs_lat' in config:
        for key in REQUIRED_LAT_KEYS:
            if key not in config['derivs_lat']:
                # Allow missing with default 0
                pass
    
    return missing


def load_aircraft(name: str, yaml_path: Optional[Path] = None) -> FullAircraftConfig:
    """
    Load aircraft configuration from YAML.
    
    Args:
        name: Aircraft name (key in YAML file)
        yaml_path: Optional custom path to YAML file
        
    Returns:
        FullAircraftConfig ready for use in any phase
        
    Raises:
        ValueError: If aircraft not found or config invalid
    """
    if yaml_path is None:
        yaml_path = get_yaml_path()
    
    if not yaml_path.exists():
        raise ValueError(f"YAML file not found: {yaml_path}")
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    if name not in data:
        available = list(data.keys())
        raise ValueError(f"Aircraft '{name}' not found. Available: {available}")
    
    config = data[name]
    
    # Validate
    missing = validate_config(name, config)
    if missing:
        print(f"Warning: Missing keys for '{name}': {missing}")
    
    # Build AircraftParameters
    params = AircraftParameters(
        mass=config.get('mass', 1000),
        Ixx=config.get('Ixx', 1000),
        Iyy=config.get('Iyy', 1000),
        Izz=config.get('Izz', 1000),
        Ixz=config.get('Ixz', 0),
        S=config.get('S', 20),
        c_bar=config.get('cbar', 2),
        b=config.get('b', 10)
    )
    
    # Build LongitudinalDerivatives
    ld = config.get('derivs_long', {})
    V_trim = config.get('V_trim', 200)
    theta0 = config.get('theta0', 0)
    rho = config.get('rho', 0.4)  # ~10km altitude
    
    long_derivs = LongitudinalDerivatives(
        X_u=ld.get('Xu', 0),
        X_alpha=ld.get('Xa', ld.get('X_alpha', 0)),
        Z_u=ld.get('Zu', 0),
        Z_alpha=ld.get('Za', ld.get('Z_alpha', -100)),
        Z_q=ld.get('Zq', ld.get('Z_q', 0)),
        M_u=ld.get('Mu', 0),
        M_alpha=ld.get('Ma', ld.get('M_alpha', -5)),
        M_q=ld.get('Mq', ld.get('M_q', -2)),
        X_delta_e=ld.get('Xde', ld.get('X_delta_e', 0)),
        Z_delta_e=ld.get('Zde', ld.get('Z_delta_e', -10)),
        M_delta_e=ld.get('Mde', ld.get('M_delta_e', -5)),
        U0=V_trim,
        theta0=theta0,
        g=9.81,
        rho=rho,
        S=params.S,
        c_bar=params.c_bar,
        m=params.mass,
        Iyy=params.Iyy
    )
    
    # Build LateralDerivatives
    latd = config.get('derivs_lat', {})
    
    lat_derivs = LateralDerivatives(
        Y_beta=latd.get('Yb', latd.get('Y_beta', -50)),
        Y_p=latd.get('Yp', latd.get('Y_p', 0)),
        Y_r=latd.get('Yr', latd.get('Y_r', 0)),
        L_beta=latd.get('Lb', latd.get('L_beta', -5)),
        L_p=latd.get('Lp', latd.get('L_p', -1)),
        L_r=latd.get('Lr', latd.get('L_r', 0.1)),
        N_beta=latd.get('Nb', latd.get('N_beta', 2)),
        N_p=latd.get('Np', latd.get('N_p', 0)),
        N_r=latd.get('Nr', latd.get('N_r', -0.5)),
        Y_delta_a=latd.get('Yda', latd.get('Y_delta_a', 0)),
        Y_delta_r=latd.get('Ydr', latd.get('Y_delta_r', 10)),
        L_delta_a=latd.get('Lda', latd.get('L_delta_a', -0.5)),
        L_delta_r=latd.get('Ldr', latd.get('L_delta_r', 0.1)),
        N_delta_a=latd.get('Nda', latd.get('N_delta_a', 0.05)),
        N_delta_r=latd.get('Ndr', latd.get('N_delta_r', -0.3))
    )
    
    # Build FullAircraftConfig
    alpha_trim = config.get('alpha_trim', np.radians(2))
    
    return FullAircraftConfig(
        params=params,
        long_derivs=long_derivs,
        lat_derivs=lat_derivs,
        V_trim=V_trim,
        alpha_trim=alpha_trim,
        rho=rho
    )


def print_aircraft_summary(name: str, config: FullAircraftConfig) -> str:
    """Format aircraft config as summary string."""
    lines = [
        f"\n{name.upper()} Configuration",
        "=" * 40,
        f"  Mass: {config.params.mass:.0f} kg",
        f"  Wing: S={config.params.S:.1f} m2, b={config.params.b:.1f} m, c={config.params.c_bar:.2f} m",
        f"  Inertia: Ixx={config.params.Ixx:.0f}, Iyy={config.params.Iyy:.0f}, Izz={config.params.Izz:.0f}",
        f"  Trim: V={config.V_trim:.1f} m/s, alpha={np.degrees(config.alpha_trim):.1f} deg",
        f"  Key derivs: M_alpha={config.long_derivs.M_alpha:.2f}, M_q={config.long_derivs.M_q:.2f}",
        "=" * 40
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    # Test loading
    print("Available aircraft:", get_available_aircraft())
    
    for name in get_available_aircraft():
        try:
            config = load_aircraft(name)
            print(print_aircraft_summary(name, config))
        except Exception as e:
            print(f"Error loading {name}: {e}")
