"""
Trim Solver for Level Flight

Computes trim conditions for steady, level, unaccelerated flight
using scipy.optimize.fsolve.

Trim conditions satisfy:
    - All accelerations zero (u̇ = v̇ = ẇ = ṗ = q̇ = ṙ = 0)
    - Level flight (γ = 0, φ = 0)
    - Constant heading (ψ̇ = 0)

Solve for:
    - Angle of attack (α)
    - Elevator deflection (δe)
    - Thrust (T)

Given:
    - Airspeed (V)
    - Altitude (h) → density (ρ)

Assumptions:
    - Symmetric flight (β = 0, p = r = 0)
    - Wings level (φ = 0)
    - No wind

References:
    Roskam, "Airplane Flight Dynamics", Chapter 4 (trim)
    Zipfel, Chapter 8 (equilibrium)
"""

import numpy as np
from scipy.optimize import fsolve
from typing import Tuple, Dict
from dataclasses import dataclass

from config import FullAircraftConfig
from aero.coefficients import compute_all_coefficients, create_full_aero_coefficients
from forces_moments import (
    compute_dynamic_pressure,
    coefficients_to_forces_moments,
    compute_total_forces
)


@dataclass
class TrimResult:
    """Container for trim solution."""
    converged: bool
    alpha: float          # Trim angle of attack (rad)
    theta: float          # Trim pitch angle (rad) = alpha for level
    delta_e: float        # Trim elevator deflection (rad)
    thrust: float         # Trim thrust (N)
    V: float              # Airspeed (m/s)
    CL: float             # Lift coefficient
    CD: float             # Drag coefficient
    q_bar: float          # Dynamic pressure (Pa)
    residual: np.ndarray  # Final residuals


def compute_standard_atmosphere_density(h: float) -> float:
    """
    Compute air density from altitude using ISA model.
    
    Args:
        h: Altitude (m)
        
    Returns:
        rho: Air density (kg/m³)
    """
    # Troposphere (h < 11000 m)
    if h < 11000:
        T = 288.15 - 0.0065 * h
        p = 101325 * (T / 288.15) ** 5.2561
    else:
        # Simplified stratosphere
        T = 216.65
        p = 22632 * np.exp(-0.0001577 * (h - 11000))
    
    rho = p / (287.05 * T)
    return rho


def solve_trim_level(
    config: FullAircraftConfig,
    V: float,
    h: float = 10668.0,  # 35,000 ft default
    alpha_guess: float = 0.05,
    delta_e_guess: float = 0.0
) -> TrimResult:
    """
    Solve for trim conditions in steady level flight.
    
    Finds α, δe, T such that:
        1. Lift = Weight  (Z-force equilibrium)
        2. Cm = 0         (Pitching moment equilibrium)
        3. Drag = Thrust  (X-force equilibrium)
    
    Args:
        config: FullAircraftConfig
        V: Desired airspeed (m/s)
        h: Altitude (m)
        alpha_guess: Initial guess for α (rad)
        delta_e_guess: Initial guess for δe (rad)
        
    Returns:
        TrimResult dataclass
        
    References:
        Roskam Ch4, Section 4.4 (longitudinal trim)
    """
    params = config.params
    aero_coeffs = create_full_aero_coefficients(config)
    
    # Use ISA if density not overridden
    rho = compute_standard_atmosphere_density(h)
    q_bar = compute_dynamic_pressure(rho, V)
    
    g = 9.81
    W = params.mass * g
    
    def trim_residuals(x):
        """
        Residual function for fsolve.
        
        x = [alpha, delta_e]
        
        Residuals:
            r1 = CL - CL_required (lift = weight)
            r2 = Cm (pitch moment = 0)
        """
        alpha, delta_e = x
        
        # Compute coefficients at this condition
        # Level flight: β = 0, p = q = r = 0
        coeffs = compute_all_coefficients(
            alpha=alpha,
            beta=0.0,
            p=0.0, q=0.0, r=0.0,
            delta_e=delta_e,
            delta_a=0.0,
            delta_r=0.0,
            V=V,
            c_bar=params.c_bar,
            b=params.b,
            coeffs=aero_coeffs
        )
        
        # Required CL for level flight
        CL_required = W / (q_bar * params.S)
        
        # Residuals
        r1 = coeffs['CL'] - CL_required  # Lift equilibrium
        r2 = coeffs['Cm']                 # Pitch moment equilibrium
        
        return np.array([r1, r2])
    
    # Solve
    x0 = np.array([alpha_guess, delta_e_guess])
    solution, info, ier, mesg = fsolve(trim_residuals, x0, full_output=True)
    
    alpha_trim, delta_e_trim = solution
    
    # Compute final coefficients
    final_coeffs = compute_all_coefficients(
        alpha=alpha_trim,
        beta=0.0,
        p=0.0, q=0.0, r=0.0,
        delta_e=delta_e_trim,
        delta_a=0.0,
        delta_r=0.0,
        V=V,
        c_bar=params.c_bar,
        b=params.b,
        coeffs=aero_coeffs
    )
    
    CL_trim = final_coeffs['CL']
    CD_trim = final_coeffs['CD']
    
    # Compute required thrust (T = D at level flight)
    thrust_trim = q_bar * params.S * CD_trim
    
    converged = (ier == 1) and np.max(np.abs(info['fvec'])) < 1e-6
    
    return TrimResult(
        converged=converged,
        alpha=alpha_trim,
        theta=alpha_trim,  # Level flight: θ = α
        delta_e=delta_e_trim,
        thrust=thrust_trim,
        V=V,
        CL=CL_trim,
        CD=CD_trim,
        q_bar=q_bar,
        residual=info['fvec']
    )


def trim_to_state(trim: TrimResult, h: float = 10668.0) -> np.ndarray:
    """
    Convert trim result to 12-element state vector.
    
    Args:
        trim: TrimResult from solve_trim_level
        h: Altitude (m)
        
    Returns:
        state: [u, v, w, p, q, r, φ, θ, ψ, x, y, h]
    """
    V = trim.V
    alpha = trim.alpha
    
    state = np.zeros(12)
    state[0] = V * np.cos(alpha)  # u
    state[2] = V * np.sin(alpha)  # w
    state[7] = alpha              # θ = α for level flight
    state[11] = h                 # h
    
    return state


def print_trim_table(trim: TrimResult) -> str:
    """Format trim results as a table."""
    lines = [
        "=" * 50,
        "TRIM SOLUTION",
        "=" * 50,
        f"  Converged: {'Yes' if trim.converged else 'No'}",
        f"  V = {trim.V:.1f} m/s ({trim.V * 1.944:.0f} kts)",
        f"  α = {np.degrees(trim.alpha):.2f}°",
        f"  θ = {np.degrees(trim.theta):.2f}°",
        f"  δe = {np.degrees(trim.delta_e):.2f}°",
        "",
        f"  CL = {trim.CL:.4f}",
        f"  CD = {trim.CD:.5f}",
        f"  L/D = {trim.CL / trim.CD:.1f}" if trim.CD > 1e-6 else "  L/D = N/A",
        "",
        f"  Thrust = {trim.thrust:.0f} N ({trim.thrust / 4.448:.0f} lbf)",
        f"  q̄ = {trim.q_bar:.0f} Pa ({trim.q_bar / 47.88:.1f} psf)",
        "",
        f"  Residual: {np.max(np.abs(trim.residual)):.2e}",
        "=" * 50
    ]
    return "\n".join(lines)
