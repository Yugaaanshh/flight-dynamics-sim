"""
Force and Moment Computation

Converts non-dimensional aerodynamic coefficients to body-axis forces and moments.

Body-axis forces (Roskam Ch1):
    X = q̄ S C_X = q̄ S (-C_A) = -q̄ S (C_D cos α - C_L sin α)   [forward]
    Y = q̄ S C_Y                                                 [right]
    Z = q̄ S C_Z = q̄ S (-C_N) = -q̄ S (C_D sin α + C_L cos α)   [down]

Body-axis moments (Roskam Ch1):
    L = q̄ S b C_l    [rolling, right wing down positive]
    M = q̄ S c̄ C_m    [pitching, nose up positive]
    N = q̄ S b C_n    [yawing, nose right positive]

where q̄ = ½ρV²

Assumptions:
- Standard atmosphere or specified density
- Coefficients computed in stability axes, converted to body
- Sign conventions per Roskam Chapter 1

References:
    Roskam, "Airplane Flight Dynamics", Chapter 1
"""

import numpy as np
from typing import Tuple, Dict
from dataclasses import dataclass


@dataclass
class ForceMoment:
    """Body-axis forces and moments."""
    X: float = 0.0    # Forward force (N)
    Y: float = 0.0    # Side force (N)
    Z: float = 0.0    # Down force (N)
    L: float = 0.0    # Rolling moment (N·m)
    M: float = 0.0    # Pitching moment (N·m)
    N: float = 0.0    # Yawing moment (N·m)


def compute_dynamic_pressure(rho: float, V: float) -> float:
    """
    Compute dynamic pressure.
    
    Args:
        rho: Air density (kg/m³)
        V: True airspeed (m/s)
        
    Returns:
        q_bar: Dynamic pressure (Pa)
    """
    return 0.5 * rho * V**2


def coefficients_to_forces_moments(
    coeffs: Dict[str, float],
    alpha: float,
    q_bar: float,
    S: float,
    b: float,
    c_bar: float
) -> ForceMoment:
    """
    Convert aerodynamic coefficients to body-axis forces and moments.
    
    Args:
        coeffs: Dict with CL, CD, Cm, CY, Cl, Cn
        alpha: Angle of attack (rad)
        q_bar: Dynamic pressure (Pa)
        S: Reference area (m²)
        b: Wing span (m)
        c_bar: Mean chord (m)
        
    Returns:
        ForceMoment dataclass with X, Y, Z, L, M, N
        
    References:
        Roskam Ch1, Equations 1.1-1.5
    """
    CL = coeffs.get('CL', 0.0)
    CD = coeffs.get('CD', 0.0)
    Cm = coeffs.get('Cm', 0.0)
    CY = coeffs.get('CY', 0.0)
    Cl = coeffs.get('Cl', 0.0)
    Cn = coeffs.get('Cn', 0.0)
    
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    
    # Convert stability-axis CL, CD to body-axis CX, CZ
    # CX = CL sin α - CD cos α  (positive forward)
    # CZ = -CL cos α - CD sin α (positive down)
    CX = CL * sin_alpha - CD * cos_alpha
    CZ = -CL * cos_alpha - CD * sin_alpha
    
    # Body-axis forces
    X = q_bar * S * CX      # Forward (N)
    Y = q_bar * S * CY      # Right (N)
    Z = q_bar * S * CZ      # Down (N)
    
    # Body-axis moments
    L = q_bar * S * b * Cl      # Rolling (N·m)
    M = q_bar * S * c_bar * Cm  # Pitching (N·m)
    N = q_bar * S * b * Cn      # Yawing (N·m)
    
    return ForceMoment(X=X, Y=Y, Z=Z, L=L, M=M, N=N)


def compute_gravity_body(
    mass: float,
    phi: float,
    theta: float,
    g: float = 9.81
) -> Tuple[float, float, float]:
    """
    Compute gravity components in body axes.
    
    Per Roskam Ch1, gravity in body axes for Euler angles (φ, θ, ψ):
        g_x = -g sin θ
        g_y = g cos θ sin φ
        g_z = g cos θ cos φ
    
    Args:
        mass: Aircraft mass (kg)
        phi: Roll angle (rad)
        theta: Pitch angle (rad)
        g: Gravitational acceleration (m/s²)
        
    Returns:
        (Fx_g, Fy_g, Fz_g): Gravity force components in body axes (N)
        
    References:
        Roskam Ch1, Equation 1.19
    """
    Fx_g = -mass * g * np.sin(theta)
    Fy_g = mass * g * np.cos(theta) * np.sin(phi)
    Fz_g = mass * g * np.cos(theta) * np.cos(phi)
    
    return Fx_g, Fy_g, Fz_g


def compute_total_forces(
    aero_fm: ForceMoment,
    mass: float,
    phi: float,
    theta: float,
    thrust: float = 0.0,
    g: float = 9.81
) -> ForceMoment:
    """
    Compute total forces including aerodynamic, gravity, and thrust.
    
    Thrust assumed along body x-axis.
    
    Args:
        aero_fm: Aerodynamic forces and moments
        mass: Aircraft mass (kg)
        phi: Roll angle (rad)
        theta: Pitch angle (rad)
        thrust: Thrust force (N), along +x body axis
        g: Gravitational acceleration (m/s²)
        
    Returns:
        ForceMoment with total forces and moments
        
    References:
        Roskam Ch1, Section 1.3
    """
    Fx_g, Fy_g, Fz_g = compute_gravity_body(mass, phi, theta, g)
    
    return ForceMoment(
        X=aero_fm.X + Fx_g + thrust,
        Y=aero_fm.Y + Fy_g,
        Z=aero_fm.Z + Fz_g,
        L=aero_fm.L,
        M=aero_fm.M,
        N=aero_fm.N
    )


def compute_airspeed_from_body_velocities(u: float, v: float, w: float) -> float:
    """Compute total airspeed from body velocity components."""
    return np.sqrt(u**2 + v**2 + w**2)


def compute_alpha_beta(u: float, v: float, w: float) -> Tuple[float, float]:
    """
    Compute angle of attack and sideslip from body velocities.
    
    Wind axes definition (Roskam Ch1):
        α = arctan(w/u)  angle of attack
        β = arcsin(v/V)  sideslip angle
    
    Args:
        u, v, w: Body-axis velocities (m/s)
        
    Returns:
        (alpha, beta): Angles (rad)
        
    References:
        Roskam Ch1, Section 1.2
    """
    V = compute_airspeed_from_body_velocities(u, v, w)
    
    # Protect against zero velocity
    if V < 1e-6:
        return 0.0, 0.0
    
    alpha = np.arctan2(w, u)
    beta = np.arcsin(np.clip(v / V, -1.0, 1.0))
    
    return alpha, beta
