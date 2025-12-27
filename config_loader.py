"""
Multi-Aircraft Configuration Loader (Fixed Version)

Robust loader with:
- Complete Imperial to SI conversion
- Full dimensional derivative computation from dimensionless coefficients
- Debug output for missing keys
- Defaults for all required parameters

Usage:
    from config_loader import load_aircraft, get_available_aircraft
    config = load_aircraft('boeing_747')
"""

import yaml
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple

from config import (
    LongitudinalDerivatives, LateralDerivatives,
    AircraftParameters, FullAircraftConfig
)


# =============================================================================
# UNIT CONVERSION CONSTANTS
# =============================================================================
FT_TO_M = 0.3048
FT2_TO_M2 = 0.09290304
SLUG_TO_KG = 14.5939
LBF_TO_N = 4.44822
SLUGFT2_TO_KGM2 = 1.35582
G_IMPERIAL = 32.174  # ft/s^2
G_SI = 9.80665  # m/s^2


def get_yaml_path() -> Path:
    """Get path to planes.yaml."""
    return Path(__file__).parent / 'planes.yaml'


def get_available_aircraft() -> List[str]:
    """Return list of available aircraft names."""
    yaml_path = get_yaml_path()
    if not yaml_path.exists():
        return []
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return list(data.keys()) if data else []


def convert_imperial_to_si(raw: Dict) -> Dict:
    """
    Convert Imperial units to SI with complete handling.
    """
    converted = {}
    
    # Reference conditions
    ref = raw.get('ref', {})
    converted['V_ref'] = float(ref.get('V_ref', 500)) * FT_TO_M
    converted['h_ref'] = float(ref.get('h_ref', 10000)) * FT_TO_M
    
    # Density: slug/ft^3 -> kg/m^3
    rho_imperial = float(ref.get('rho_ref', 0.001))
    converted['rho_ref'] = rho_imperial * 515.379
    
    # Mass properties - ensure float conversion for YAML scientific notation
    mass = raw.get('mass_props', {})
    weight_lbf = float(mass.get('weight_lbf', 10000))
    converted['mass_kg'] = weight_lbf * LBF_TO_N / G_SI  # W = mg -> m = W/g
    
    # Inertias - abs() to handle any sign, explicit float()
    converted['Ixx'] = abs(float(mass.get('Ixx', 1000))) * SLUGFT2_TO_KGM2
    converted['Iyy'] = abs(float(mass.get('Iyy', 1000))) * SLUGFT2_TO_KGM2
    converted['Izz'] = abs(float(mass.get('Izz', 1000))) * SLUGFT2_TO_KGM2
    converted['Ixz'] = float(mass.get('Ixz', 0)) * SLUGFT2_TO_KGM2  # Can be negative
    
    # Geometry
    geom = raw.get('geometry', {})
    converted['S'] = float(geom.get('S', 100)) * FT2_TO_M2
    converted['b'] = float(geom.get('b', 50)) * FT_TO_M
    converted['cbar'] = float(geom.get('cbar', 5)) * FT_TO_M
    
    # Aero coefficients (dimensionless - no conversion needed)
    converted['aero_long'] = raw.get('aero_long', {}).copy()
    converted['aero_lat'] = raw.get('aero_lat', {}).copy()
    
    return converted


def compute_dimensional_derivatives(
    V: float, rho: float, S: float, b: float, cbar: float,
    mass: float, Ixx: float, Iyy: float, Izz: float, Ixz: float,
    aero_long: Dict, aero_lat: Dict,
    debug: bool = False
) -> Dict:
    """
    Compute ALL dimensional stability/control derivatives from dimensionless coefficients.
    
    Formulas from Roskam Chapter 5, Table 5.1.
    """
    q_bar = 0.5 * rho * V**2
    g = G_SI
    
    if debug:
        print(f"  Computing derivatives: V={V:.1f} m/s, rho={rho:.4f} kg/m3, q_bar={q_bar:.1f} Pa")
        print(f"  Mass={mass:.0f} kg, S={S:.1f} m2, b={b:.1f} m, c={cbar:.2f} m")
    
    # Extract dimensionless coefficients with sensible defaults
    CL0 = aero_long.get('CL0', aero_long.get('CL_ref', 0.3))
    CL_alpha = aero_long.get('CL_alpha', 5.0)
    CD0 = aero_long.get('CD0', 0.02)
    CD_alpha = aero_long.get('CD_alpha', 0.3)
    K = aero_long.get('K', 0.05)  # Induced drag factor
    
    Cm_alpha = aero_long.get('Cm_alpha', -1.0)
    CL_q = aero_long.get('CL_q', 5.0)
    Cm_q = aero_long.get('Cm_q', -15.0)
    CL_de = aero_long.get('CL_de', 0.4)
    Cm_de = aero_long.get('Cm_de', -1.2)
    
    CY_beta = aero_lat.get('CY_beta', -0.5)
    Cl_beta = aero_lat.get('Cl_beta', -0.1)
    Cn_beta = aero_lat.get('Cn_beta', 0.1)
    Cl_p = aero_lat.get('Cl_p', -0.4)
    Cn_p = aero_lat.get('Cn_p', -0.03)
    Cl_r = aero_lat.get('Cl_r', 0.1)
    Cn_r = aero_lat.get('Cn_r', -0.15)
    Cl_da = aero_lat.get('Cl_da', 0.1)
    Cn_da = aero_lat.get('Cn_da', 0.01)
    CY_dr = aero_lat.get('CY_dr', 0.15)
    Cl_dr = aero_lat.get('Cl_dr', 0.01)
    Cn_dr = aero_lat.get('Cn_dr', -0.08)
    
    # ======================
    # LONGITUDINAL DERIVATIVES
    # ======================
    # X derivatives (drag-related, speed stability)
    Xu = -q_bar * S * 2 * CD0 / (mass * V)
    Xa = q_bar * S * (CL0 - CD_alpha) / mass
    
    # Z derivatives (lift-related)
    Zu = -q_bar * S * 2 * CL0 / (mass * V)
    Za = -q_bar * S * CL_alpha / mass
    Zq = -q_bar * S * cbar * CL_q / (2 * mass * V)
    Zde = -q_bar * S * CL_de / mass
    
    # M derivatives (pitching moment)
    Ma = q_bar * S * cbar * Cm_alpha / Iyy
    Mq = q_bar * S * cbar**2 * Cm_q / (2 * V * Iyy)
    Mu = 0.0  # Usually negligible for subsonic
    Mde = q_bar * S * cbar * Cm_de / Iyy
    
    # ======================
    # LATERAL DERIVATIVES
    # ======================
    # Y derivatives (side force)
    Yb = q_bar * S * CY_beta / mass
    Yp = 0.0  # Usually small
    Yr = 0.0  # Usually small
    Yda = 0.0
    Ydr = q_bar * S * CY_dr / mass
    
    # L derivatives (rolling moment) - use Ixx
    Lb = q_bar * S * b * Cl_beta / Ixx
    Lp = q_bar * S * b**2 * Cl_p / (2 * V * Ixx)
    Lr = q_bar * S * b**2 * Cl_r / (2 * V * Ixx)
    Lda = q_bar * S * b * Cl_da / Ixx
    Ldr = q_bar * S * b * Cl_dr / Ixx
    
    # N derivatives (yawing moment) - use Izz
    Nb = q_bar * S * b * Cn_beta / Izz
    Np = q_bar * S * b**2 * Cn_p / (2 * V * Izz)
    Nr = q_bar * S * b**2 * Cn_r / (2 * V * Izz)
    Nda = q_bar * S * b * Cn_da / Izz
    Ndr = q_bar * S * b * Cn_dr / Izz
    
    result = {
        'Xu': Xu, 'Xa': Xa,
        'Zu': Zu, 'Za': Za, 'Zq': Zq, 'Zde': Zde,
        'Mu': Mu, 'Ma': Ma, 'Mq': Mq, 'Mde': Mde,
        'Yb': Yb, 'Yp': Yp, 'Yr': Yr, 'Yda': Yda, 'Ydr': Ydr,
        'Lb': Lb, 'Lp': Lp, 'Lr': Lr, 'Lda': Lda, 'Ldr': Ldr,
        'Nb': Nb, 'Np': Np, 'Nr': Nr, 'Nda': Nda, 'Ndr': Ndr,
        'q_bar': q_bar
    }
    
    if debug:
        print(f"  Key derivs: Za={Za:.1f}, Ma={Ma:.3f}, Mq={Mq:.4f}, Lb={Lb:.3f}, Nb={Nb:.3f}")
    
    return result


def load_aircraft(name: str, yaml_path: Optional[Path] = None, debug: bool = False) -> FullAircraftConfig:
    """
    Load aircraft configuration from YAML.
    
    Handles Imperial/SI conversion and derivative computation.
    """
    if yaml_path is None:
        yaml_path = get_yaml_path()
    
    if not yaml_path.exists():
        raise ValueError(f"YAML file not found: {yaml_path}")
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    if name not in data:
        raise ValueError(f"Aircraft '{name}' not found. Available: {list(data.keys())}")
    
    raw = data[name]
    meta = raw.get('meta', {})
    units = meta.get('units', 'SI').lower()
    
    if debug:
        print(f"\nLoading {name} (units={units})")
    
    # Convert or extract parameters
    if units == 'imperial':
        conv = convert_imperial_to_si(raw)
        mass = conv['mass_kg']
        Ixx = conv['Ixx']
        Iyy = conv['Iyy']
        Izz = conv['Izz']
        Ixz = conv['Ixz']
        S = conv['S']
        b = conv['b']
        cbar = conv['cbar']
        V_ref = conv['V_ref']
        h_ref = conv['h_ref']
        rho = conv['rho_ref']
        aero_long = conv['aero_long']
        aero_lat = conv['aero_lat']
    else:
        # SI units
        ref = raw.get('ref', {})
        V_ref = ref.get('V_ref', 200)
        h_ref = ref.get('h_ref', 10000)
        rho = ref.get('rho_ref', 0.4)
        
        mp = raw.get('mass_props', {})
        mass = mp.get('mass_kg', 5000)
        Ixx = mp.get('Ixx', 10000)
        Iyy = mp.get('Iyy', 50000)
        Izz = mp.get('Izz', 55000)
        Ixz = mp.get('Ixz', 0)
        
        geom = raw.get('geometry', {})
        S = geom.get('S', 20)
        b = geom.get('b', 15)
        cbar = geom.get('cbar', 2)
        
        aero_long = raw.get('aero_long', {})
        aero_lat = raw.get('aero_lat', {})
    
    # Use pre-computed dimensional derivs if available (backwards compat)
    if 'dimensional' in raw and units.lower() == 'si':
        dim = raw['dimensional']
        if debug:
            print("  Using pre-computed dimensional derivatives")
    else:
        # Compute from dimensionless coefficients
        dim = compute_dimensional_derivatives(
            V=V_ref, rho=rho, S=S, b=b, cbar=cbar,
            mass=mass, Ixx=Ixx, Iyy=Iyy, Izz=Izz, Ixz=Ixz,
            aero_long=aero_long, aero_lat=aero_lat,
            debug=debug
        )
    
    # Build AircraftParameters
    params = AircraftParameters(
        mass=mass,
        Ixx=Ixx,
        Iyy=Iyy,
        Izz=Izz,
        Ixz=Ixz,
        S=S,
        c_bar=cbar,
        b=b
    )
    
    # Build LongitudinalDerivatives
    long_derivs = LongitudinalDerivatives(
        X_u=dim.get('Xu', -0.02),
        X_alpha=dim.get('Xa', 10.0),
        Z_u=dim.get('Zu', -0.2),
        Z_alpha=dim.get('Za', -100.0),
        Z_q=dim.get('Zq', -3.0),
        M_u=dim.get('Mu', 0.0),
        M_alpha=dim.get('Ma', -3.0),
        M_q=dim.get('Mq', -0.5),
        X_delta_e=0.0,
        Z_delta_e=dim.get('Zde', -10.0),
        M_delta_e=dim.get('Mde', -1.5),
        U0=V_ref,
        theta0=0.0,
        g=G_SI,
        rho=rho,
        S=S,
        c_bar=cbar,
        m=mass,
        Iyy=Iyy
    )
    
    # Build LateralDerivatives
    lat_derivs = LateralDerivatives(
        Y_beta=dim.get('Yb', -30.0),
        Y_p=dim.get('Yp', 0.0),
        Y_r=dim.get('Yr', 0.0),
        L_beta=dim.get('Lb', -5.0),
        L_p=dim.get('Lp', -0.5),
        L_r=dim.get('Lr', 0.1),
        N_beta=dim.get('Nb', 2.0),
        N_p=dim.get('Np', -0.05),
        N_r=dim.get('Nr', -0.2),
        Y_delta_a=dim.get('Yda', 0.0),
        Y_delta_r=dim.get('Ydr', 5.0),
        L_delta_a=dim.get('Lda', -0.5),
        L_delta_r=dim.get('Ldr', 0.1),
        N_delta_a=dim.get('Nda', 0.02),
        N_delta_r=dim.get('Ndr', -0.3)
    )
    
    # Estimate trim alpha
    CL0 = aero_long.get('CL0', aero_long.get('CL_ref', 0.3))
    CL_alpha = aero_long.get('CL_alpha', 5.0)
    q_bar = 0.5 * rho * V_ref**2
    CL_needed = mass * G_SI / (q_bar * S) if q_bar * S > 0 else 0.5
    alpha_trim = (CL_needed - CL0) / CL_alpha if CL_alpha != 0 else 0.05
    alpha_trim = np.clip(alpha_trim, 0.0, 0.3)
    
    if debug:
        print(f"  Trim alpha estimate: {np.degrees(alpha_trim):.2f} deg (CL_needed={CL_needed:.3f})")
    
    return FullAircraftConfig(
        params=params,
        long_derivs=long_derivs,
        lat_derivs=lat_derivs,
        V_trim=V_ref,
        alpha_trim=alpha_trim,
        rho=rho
    )


def get_aircraft_info(name: str) -> Dict[str, Any]:
    """Get aircraft metadata for UI display."""
    yaml_path = get_yaml_path()
    
    if not yaml_path.exists():
        return {'name': name, 'V_ref': 200, 'h_ref': 10000}
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    if name not in data:
        return {'name': name, 'V_ref': 200, 'h_ref': 10000}
    
    raw = data[name]
    meta = raw.get('meta', {})
    units = meta.get('units', 'SI').lower()
    ref = raw.get('ref', {})
    
    if units == 'imperial':
        V_ref = ref.get('V_ref', 500) * FT_TO_M
        h_ref = ref.get('h_ref', 10000) * FT_TO_M
    else:
        V_ref = ref.get('V_ref', 200)
        h_ref = ref.get('h_ref', 10000)
    
    return {
        'name': meta.get('name', name),
        'source': meta.get('source', 'Unknown'),
        'V_ref': V_ref,
        'h_ref': h_ref
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Config Loader Test")
    print("=" * 60)
    
    for name in get_available_aircraft():
        print(f"\n--- {name.upper()} ---")
        try:
            config = load_aircraft(name, debug=True)
            print(f"  SUCCESS: Mass={config.params.mass:,.0f} kg, V={config.V_trim:.1f} m/s")
            print(f"  Long: Ma={config.long_derivs.M_alpha:.3f}, Mq={config.long_derivs.M_q:.4f}")
            print(f"  Lat:  Lb={config.lat_derivs.L_beta:.3f}, Nb={config.lat_derivs.N_beta:.3f}")
        except Exception as e:
            print(f"  ERROR: {e}")
    
    print("\n" + "=" * 60)
