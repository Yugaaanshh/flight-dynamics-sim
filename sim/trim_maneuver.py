"""
General Trim Solver for Steady Maneuvers

Computes trim conditions for various steady flight maneuvers:
- Level cruise (γ=0, φ=0, nz=1)
- Coordinated banked turn (φ≠0, γ≈0, nz>1)
- Steady pull-up / push-over (γ≠0, nz>1)

Extends the basic level trim to general maneuvers following Roskam Chapter 4.

Trim conditions satisfy:
- Force equilibrium in body axes (with load factor)
- Moment equilibrium (L=M=N=0)
- Kinematics consistent with flight path γ and bank angle φ

References:
    Roskam, "Airplane Flight Dynamics", Chapter 4
    Section 4.3: Steady rectilinear flight
    Section 4.4: Steady maneuvering flight
"""

import numpy as np
from scipy.optimize import fsolve, root
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FullAircraftConfig
from eom.six_dof import SixDoFModel, Controls
from eom.six_dof import U as IDX_U, V as IDX_V, W as IDX_W
from eom.six_dof import P as IDX_P, Q as IDX_Q, R as IDX_R
from eom.six_dof import PHI as IDX_PHI, THETA as IDX_THETA, PSI as IDX_PSI, H as IDX_H
from aero.coefficients import create_full_aero_coefficients, compute_all_coefficients
from forces_moments import (
    compute_dynamic_pressure,
    coefficients_to_forces_moments,
    compute_total_forces,
    compute_alpha_beta,
    compute_airspeed_from_body_velocities
)


def compute_standard_atmosphere_density(h: float) -> float:
    """
    Compute air density from altitude using ISA model.
    
    Args:
        h: Altitude (m)
        
    Returns:
        rho: Air density (kg/m³)
    """
    if h < 11000:
        T = 288.15 - 0.0065 * h
        p = 101325 * (T / 288.15) ** 5.2561
    else:
        T = 216.65
        p = 22632 * np.exp(-0.0001577 * (h - 11000))
    
    rho = p / (287.05 * T)
    return rho


@dataclass
class TrimResult:
    """Container for general trim solution."""
    converged: bool
    x_trim: np.ndarray        # 12-element trim state
    u_trim: np.ndarray        # 4-element control [δe, δa, δr, T]
    alpha: float              # Trim angle of attack (rad)
    beta: float               # Sideslip angle (rad)
    theta: float              # Pitch angle (rad)
    phi: float                # Bank angle (rad)
    nz: float                 # Load factor achieved
    gamma: float              # Flight path angle (rad)
    V: float                  # Airspeed (m/s)
    CL: float                 # Lift coefficient
    residual: float           # Norm of trim equations
    maneuver: str             # Maneuver type


def trim_steady_state(
    model_6dof: SixDoFModel,
    target_V: float,
    target_altitude: float,
    nz_target: float = 1.0,
    bank_angle_phi: float = 0.0,
    flight_path_gamma: float = 0.0,
    initial_guess: Optional[Dict] = None,
    verbose: bool = True
) -> TrimResult:
    """
    Compute steady-state trim for a general maneuver.
    
    Maneuver types:
        - Level flight:      nz=1.0, φ=0, γ=0
        - Coordinated turn:  nz>1.0, φ≠0, γ≈0
        - Pull-up/push-over: γ≠0, nz>1.0
    
    Trim variables solved for:
        - α (angle of attack)
        - δe (elevator)
        - δa (aileron) - nonzero for turns
        - δr (rudder) - nonzero for turns
        - Thrust
    
    Args:
        model_6dof: SixDoFModel instance
        target_V: Desired airspeed (m/s)
        target_altitude: Altitude (m)
        nz_target: Target load factor (g's)
        bank_angle_phi: Bank angle (rad) for turns
        flight_path_gamma: Flight path angle (rad) for climb/dive
        initial_guess: Dict with 'alpha', 'delta_e', etc.
        verbose: Print progress
        
    Returns:
        TrimResult dataclass
        
    References:
        Roskam Ch4, Sections 4.3-4.4
    """
    params = model_6dof.params
    config = model_6dof.config
    aero_coeffs = model_6dof.aero_coeffs
    g = 9.81
    
    rho = compute_standard_atmosphere_density(target_altitude)
    q_bar = compute_dynamic_pressure(rho, target_V)
    
    # For coordinated turn: turn rate and load factor
    cos_phi = np.cos(bank_angle_phi)
    sin_phi = np.sin(bank_angle_phi)
    cos_gamma = np.cos(flight_path_gamma)
    sin_gamma = np.sin(flight_path_gamma)
    
    # Required lift for load factor
    W = params.mass * g
    L_required = W * nz_target
    CL_required = L_required / (q_bar * params.S)
    
    # Initial guesses
    if initial_guess is None:
        alpha_0 = CL_required / 5.0  # Approx CLα ≈ 5
        delta_e_0 = -0.01
        delta_a_0 = 0.0
        delta_r_0 = 0.0
    else:
        alpha_0 = initial_guess.get('alpha', 0.05)
        delta_e_0 = initial_guess.get('delta_e', -0.01)
        delta_a_0 = initial_guess.get('delta_a', 0.0)
        delta_r_0 = initial_guess.get('delta_r', 0.0)
    
    def trim_residuals(x):
        """
        Residuals for general trim.
        
        x = [alpha, delta_e, delta_a, delta_r]
        
        Residuals:
            r1: CL = CL_required (lift equilibrium)
            r2: Cm = 0 (pitch moment)
            r3: Cl = 0 (roll moment)
            r4: Cn = 0 (yaw moment)
        """
        alpha, delta_e, delta_a, delta_r = x
        
        # For steady maneuver, body rates relate to turn rate
        # In coordinated turn: p, r related to bank, q related to pull-up
        # Simplified: assume small rates for computation
        p_trim = 0.0
        q_trim = 0.0
        r_trim = 0.0
        
        # For turn, need small coordinated rates
        if abs(bank_angle_phi) > 0.01:
            # Turn rate: ω = g tan(φ) / V for coordinated turn
            omega_turn = g * np.tan(bank_angle_phi) / target_V
            # Body rates from Euler kinematics (simplified)
            p_trim = -omega_turn * sin_gamma
            r_trim = omega_turn * cos_gamma * cos_phi
        
        beta = 0.0  # Coordinated flight assumption
        
        # Compute coefficients
        coeffs = compute_all_coefficients(
            alpha=alpha,
            beta=beta,
            p=p_trim, q=q_trim, r=r_trim,
            delta_e=delta_e,
            delta_a=delta_a,
            delta_r=delta_r,
            V=target_V,
            c_bar=params.c_bar,
            b=params.b,
            coeffs=aero_coeffs
        )
        
        # Residuals
        r1 = coeffs['CL'] - CL_required
        r2 = coeffs['Cm']  # Pitch moment = 0
        r3 = coeffs['Cl']  # Roll moment = 0
        r4 = coeffs['Cn']  # Yaw moment = 0
        
        return np.array([r1, r2, r3, r4])
    
    # Solve
    x0 = np.array([alpha_0, delta_e_0, delta_a_0, delta_r_0])
    
    try:
        solution, info, ier, mesg = fsolve(trim_residuals, x0, full_output=True)
        converged = (ier == 1) and np.linalg.norm(info['fvec']) < 1e-4
    except Exception as e:
        if verbose:
            print(f"Trim solver warning: {e}")
        solution = x0
        converged = False
        info = {'fvec': np.ones(4)}
    
    alpha_trim, delta_e_trim, delta_a_trim, delta_r_trim = solution
    
    # Compute final coefficients and thrust
    coeffs_final = compute_all_coefficients(
        alpha=alpha_trim,
        beta=0.0,
        p=0.0, q=0.0, r=0.0,
        delta_e=delta_e_trim,
        delta_a=delta_a_trim,
        delta_r=delta_r_trim,
        V=target_V,
        c_bar=params.c_bar,
        b=params.b,
        coeffs=aero_coeffs
    )
    
    CL_trim = coeffs_final['CL']
    CD_trim = coeffs_final['CD']
    
    # Thrust = Drag (for level or steady flight path)
    thrust_trim = q_bar * params.S * CD_trim
    
    # Correct thrust for flight path angle
    if abs(flight_path_gamma) > 0.001:
        thrust_trim += W * sin_gamma
    
    # Build trim state
    theta_trim = alpha_trim + flight_path_gamma  # For coordinated/level
    
    x_trim = np.zeros(12)
    x_trim[IDX_U] = target_V * np.cos(alpha_trim)
    x_trim[IDX_W] = target_V * np.sin(alpha_trim)
    x_trim[IDX_PHI] = bank_angle_phi
    x_trim[IDX_THETA] = theta_trim
    x_trim[IDX_H] = target_altitude
    
    # Control vector
    u_trim = np.array([delta_e_trim, delta_a_trim, delta_r_trim, thrust_trim])
    
    # Determine maneuver type
    if abs(bank_angle_phi) < 0.01 and abs(flight_path_gamma) < 0.01:
        maneuver = "Level Cruise"
    elif abs(bank_angle_phi) > 0.01:
        maneuver = f"Coordinated Turn (phi={np.degrees(bank_angle_phi):.0f}deg)"
    else:
        maneuver = f"Climb/Dive (gamma={np.degrees(flight_path_gamma):.1f}deg)"
    
    if verbose and converged:
        print(f"  Trim converged: {maneuver}")
        print(f"    alpha = {np.degrees(alpha_trim):.2f} deg, de = {np.degrees(delta_e_trim):.2f} deg")
        print(f"    CL = {CL_trim:.4f}, nz = {nz_target:.2f}")
    
    return TrimResult(
        converged=converged,
        x_trim=x_trim,
        u_trim=u_trim,
        alpha=alpha_trim,
        beta=0.0,
        theta=theta_trim,
        phi=bank_angle_phi,
        nz=nz_target,
        gamma=flight_path_gamma,
        V=target_V,
        CL=CL_trim,
        residual=np.linalg.norm(info['fvec']),
        maneuver=maneuver
    )


def trim_level_flight(
    model_6dof: SixDoFModel,
    V: float,
    h: float,
    verbose: bool = True
) -> TrimResult:
    """
    Thin wrapper for level flight trim.
    
    Args:
        model_6dof: SixDoFModel instance
        V: Airspeed (m/s)
        h: Altitude (m)
        
    Returns:
        TrimResult
        
    References:
        Roskam Ch4, Section 4.3.1
    """
    return trim_steady_state(
        model_6dof=model_6dof,
        target_V=V,
        target_altitude=h,
        nz_target=1.0,
        bank_angle_phi=0.0,
        flight_path_gamma=0.0,
        verbose=verbose
    )


def trim_coordinated_turn(
    model_6dof: SixDoFModel,
    V: float,
    h: float,
    bank_angle_deg: float,
    verbose: bool = True
) -> TrimResult:
    """
    Trim for coordinated level turn.
    
    Load factor nz = 1/cos(φ) for level turn (γ=0).
    
    Args:
        model_6dof: SixDoFModel instance
        V: Airspeed (m/s)
        h: Altitude (m)
        bank_angle_deg: Bank angle (degrees)
        
    Returns:
        TrimResult
        
    References:
        Roskam Ch4, Section 4.4.1
    """
    phi_rad = np.radians(bank_angle_deg)
    nz = 1.0 / np.cos(phi_rad)  # Level turn load factor
    
    return trim_steady_state(
        model_6dof=model_6dof,
        target_V=V,
        target_altitude=h,
        nz_target=nz,
        bank_angle_phi=phi_rad,
        flight_path_gamma=0.0,
        verbose=verbose
    )


def trim_pullup(
    model_6dof: SixDoFModel,
    V: float,
    h: float,
    nz_target: float,
    verbose: bool = True
) -> TrimResult:
    """
    Trim for steady pull-up maneuver.
    
    For steady pull-up: γ = constant, q = g(nz-1)/V
    
    Args:
        model_6dof: SixDoFModel instance
        V: Airspeed (m/s)
        h: Altitude (m)
        nz_target: Target load factor (> 1 for pull-up)
        
    Returns:
        TrimResult
        
    References:
        Roskam Ch4, Section 4.4.2
    """
    # For pull-up, estimate flight path rate
    g = 9.81
    gamma_dot = g * (nz_target - 1) / V  # rad/s
    
    # Assume a small positive gamma for pull-up initiation
    gamma_est = 0.05 if nz_target > 1 else -0.05
    
    return trim_steady_state(
        model_6dof=model_6dof,
        target_V=V,
        target_altitude=h,
        nz_target=nz_target,
        bank_angle_phi=0.0,
        flight_path_gamma=gamma_est,
        verbose=verbose
    )


def print_trim_summary(trim: TrimResult) -> str:
    """Format trim result as summary string."""
    lines = [
        f"  {trim.maneuver}:",
        f"    Converged: {'Yes' if trim.converged else 'No'}",
        f"    V = {trim.V:.1f} m/s, h = {trim.x_trim[11]:.0f} m",
        f"    alpha = {np.degrees(trim.alpha):.2f} deg, theta = {np.degrees(trim.theta):.2f} deg, phi = {np.degrees(trim.phi):.1f} deg",
        f"    de = {np.degrees(trim.u_trim[0]):.2f} deg, da = {np.degrees(trim.u_trim[1]):.2f} deg, dr = {np.degrees(trim.u_trim[2]):.2f} deg",
        f"    nz = {trim.nz:.2f}, CL = {trim.CL:.4f}",
        f"    Thrust = {trim.u_trim[3]:.0f} N"
    ]
    return "\n".join(lines)
