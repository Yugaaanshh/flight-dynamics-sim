"""
Control Surface Influence Experiments

Parametric sweeps to study how control surface effectiveness affects
aircraft dynamics, modes, and time responses.

Experiments:
1. Elevator power (M_de): Short-period/phugoid modes and pitch response
2. Aileron power (Cl_da): Roll mode time constant and Dutch roll
3. Rudder power (Cn_dr): Dutch roll damping and yaw response

Design-style exploration: de -> dC -> mode -> response chain.

References:
    Roskam, "Airplane Flight Dynamics", Chapter 5
    Section 5.4: Effects of stability and control derivatives
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import replace
from typing import Dict, List, Tuple, Optional

from config import (
    LongitudinalDerivatives, LateralDerivatives, 
    FullAircraftConfig, AircraftParameters,
    ROSKAM_BUSINESS_JET, BUSINESS_JET_LATERAL, BUSINESS_JET_PARAMS, BUSINESS_JET_6DOF
)
from eom.longitudinal import LongitudinalLinearModel
from eom.six_dof import SixDoFModel
from analysis.linearize import linearize_6dof, analyze_modes, ModeInfo


def run_elevator_power_sweep(
    base_derivs: LongitudinalDerivatives,
    scale_factors: Tuple[float, ...] = (0.6, 0.8, 1.0, 1.2, 1.4),
    de_step_deg: float = -2.0,
    t_step: float = 1.0,
    t_final: float = 15.0,
    dt: float = 0.02
) -> Dict:
    """
    Sweep elevator control power (M_de) and analyze modes + time responses.
    
    For each scale factor s:
        1. Scale M_delta_e by s (and Z_delta_e for consistency)
        2. Build longitudinal state-space model
        3. Compute short-period and phugoid modes
        4. Simulate elevator step response
    
    Args:
        base_derivs: Baseline LongitudinalDerivatives (e.g., ROSKAM_BUSINESS_JET)
        scale_factors: Multipliers for M_delta_e
        de_step_deg: Elevator step amplitude (deg)
        t_step: Time of step application (s)
        t_final: Simulation end time (s)
        dt: Time step (s)
        
    Returns:
        Dict with results for each scale factor
        
    References:
        Roskam Ch5, Section 5.4.1 (elevator effectiveness)
    """
    results = {}
    de_step_rad = np.radians(de_step_deg)
    
    for scale in scale_factors:
        # Scale elevator derivatives
        M_de_scaled = base_derivs.M_delta_e * scale
        Z_de_scaled = base_derivs.Z_delta_e * scale
        
        # Create scaled derivatives
        scaled_derivs = replace(
            base_derivs,
            M_delta_e=M_de_scaled,
            Z_delta_e=Z_de_scaled
        )
        
        # Build linear model
        model = LongitudinalLinearModel(scaled_derivs)
        
        # Get modes
        modes = model.analyze_modes()
        
        # Simulate step response
        def dynamics(t, x):
            de = de_step_rad if t >= t_step else 0.0
            u = np.array([de])
            return model.A @ x + model.B @ u
        
        t_eval = np.arange(0, t_final, dt)
        x0 = np.zeros(4)
        
        sol = solve_ivp(dynamics, (0, t_final), x0, t_eval=t_eval, method='RK45')
        
        # Compute elevator history for plotting
        de_history = np.array([de_step_rad if t >= t_step else 0.0 for t in sol.t])
        
        results[scale] = {
            'M_de': M_de_scaled,
            'A': model.A,
            'B': model.B,
            'modes': modes,
            'time': sol.t,
            'response': {
                'u': sol.y[0],       # du (m/s)
                'alpha': sol.y[1],   # dalpha (rad)
                'q': sol.y[2],       # dq (rad/s)
                'theta': sol.y[3],   # dtheta (rad)
            },
            'delta_e': de_history
        }
    
    return results


def run_aileron_power_sweep(
    base_config: FullAircraftConfig,
    x_trim: np.ndarray,
    u_trim: np.ndarray,
    scale_factors: Tuple[float, ...] = (0.6, 0.8, 1.0, 1.2, 1.4)
) -> Dict:
    """
    Sweep aileron control power (L_delta_a, N_delta_a) and analyze lateral modes.
    
    For each scale factor s:
        1. Scale aileron derivatives (L_da, N_da)
        2. Re-linearize 6-DOF model at trim
        3. Classify modes (Dutch roll, roll, spiral)
    
    Args:
        base_config: Baseline FullAircraftConfig
        x_trim: Trim state from level flight
        u_trim: Trim controls [de, da, dr, T]
        scale_factors: Multipliers for aileron derivatives
        
    Returns:
        Dict with mode analysis for each scale
        
    References:
        Roskam Ch5, Section 5.4.2 (lateral control)
    """
    results = {}
    
    for scale in scale_factors:
        # Scale aileron derivatives
        scaled_lat = replace(
            base_config.lat_derivs,
            L_delta_a=base_config.lat_derivs.L_delta_a * scale,
            N_delta_a=base_config.lat_derivs.N_delta_a * scale
        )
        
        # Create scaled config
        scaled_config = FullAircraftConfig(
            params=base_config.params,
            long_derivs=base_config.long_derivs,
            lat_derivs=scaled_lat,
            V_trim=base_config.V_trim,
            alpha_trim=base_config.alpha_trim,
            rho=base_config.rho
        )
        
        # Build and linearize model
        model = SixDoFModel(scaled_config)
        A, B = linearize_6dof(model, x_trim, u_trim)
        
        # Analyze modes
        modes = analyze_modes(A)
        
        # Extract key mode info
        dutch_roll = next((m for m in modes if m.name == 'Dutch Roll'), None)
        roll_mode = next((m for m in modes if m.name == 'Roll'), None)
        spiral = next((m for m in modes if m.name == 'Spiral'), None)
        
        results[scale] = {
            'L_da': scaled_lat.L_delta_a,
            'N_da': scaled_lat.N_delta_a,
            'modes': modes,
            'dutch_roll': {
                'zeta': dutch_roll.zeta if dutch_roll else None,
                'omega_n': dutch_roll.omega_n if dutch_roll else None,
                'period': dutch_roll.period if dutch_roll else None
            } if dutch_roll else None,
            'roll': {
                'tau': roll_mode.time_constant if roll_mode else None,
                'lambda': roll_mode.eigenvalue.real if roll_mode else None
            } if roll_mode else None,
            'spiral': {
                'tau': spiral.time_constant if spiral else None,
                'lambda': spiral.eigenvalue.real if spiral else None
            } if spiral else None
        }
    
    return results


def run_rudder_power_sweep(
    base_config: FullAircraftConfig,
    x_trim: np.ndarray,
    u_trim: np.ndarray,
    scale_factors: Tuple[float, ...] = (0.6, 0.8, 1.0, 1.2, 1.4)
) -> Dict:
    """
    Sweep rudder control power (N_delta_r, L_delta_r, Y_delta_r) and analyze modes.
    
    For each scale factor s:
        1. Scale rudder derivatives
        2. Re-linearize 6-DOF model
        3. Analyze Dutch roll and spiral modes
    
    Args:
        base_config: Baseline FullAircraftConfig
        x_trim: Trim state
        u_trim: Trim controls
        scale_factors: Multipliers for rudder derivatives
        
    Returns:
        Dict with mode analysis for each scale
        
    References:
        Roskam Ch5, Section 5.4.3 (directional control)
    """
    results = {}
    
    for scale in scale_factors:
        # Scale rudder derivatives
        scaled_lat = replace(
            base_config.lat_derivs,
            N_delta_r=base_config.lat_derivs.N_delta_r * scale,
            L_delta_r=base_config.lat_derivs.L_delta_r * scale,
            Y_delta_r=base_config.lat_derivs.Y_delta_r * scale
        )
        
        # Create scaled config
        scaled_config = FullAircraftConfig(
            params=base_config.params,
            long_derivs=base_config.long_derivs,
            lat_derivs=scaled_lat,
            V_trim=base_config.V_trim,
            alpha_trim=base_config.alpha_trim,
            rho=base_config.rho
        )
        
        # Build and linearize model
        model = SixDoFModel(scaled_config)
        A, B = linearize_6dof(model, x_trim, u_trim)
        
        # Analyze modes
        modes = analyze_modes(A)
        
        # Extract key mode info
        dutch_roll = next((m for m in modes if m.name == 'Dutch Roll'), None)
        spiral = next((m for m in modes if m.name == 'Spiral'), None)
        
        results[scale] = {
            'N_dr': scaled_lat.N_delta_r,
            'modes': modes,
            'dutch_roll': {
                'zeta': dutch_roll.zeta if dutch_roll else None,
                'omega_n': dutch_roll.omega_n if dutch_roll else None,
            } if dutch_roll else None,
            'spiral': {
                'tau': spiral.time_constant if spiral else None,
            } if spiral else None
        }
    
    return results


def compute_response_metrics(
    time: np.ndarray,
    response: np.ndarray,
    t_step: float = 1.0
) -> Dict:
    """
    Compute step response metrics (overshoot, settling time, etc).
    
    Args:
        time: Time vector
        response: Response signal
        t_step: Time of step input
        
    Returns:
        Dict with 'overshoot', 'settling_time', 'peak_value', 'peak_time'
    """
    # Find response after step
    step_idx = np.searchsorted(time, t_step)
    resp_after = response[step_idx:]
    time_after = time[step_idx:]
    
    if len(resp_after) == 0:
        return {'overshoot': 0, 'settling_time': 0, 'peak_value': 0, 'peak_time': 0}
    
    # Final value (approximate steady state)
    final_value = resp_after[-1]
    
    # Peak value and time
    peak_idx = np.argmax(np.abs(resp_after))
    peak_value = resp_after[peak_idx]
    peak_time = time_after[peak_idx]
    
    # Overshoot (relative to final)
    if abs(final_value) > 1e-10:
        overshoot = (abs(peak_value) - abs(final_value)) / abs(final_value) * 100
    else:
        overshoot = 0
    
    # Settling time (2% band)
    tolerance = 0.02 * abs(final_value) if abs(final_value) > 1e-10 else 0.02 * abs(peak_value)
    settled = np.abs(resp_after - final_value) <= tolerance
    
    if np.any(settled):
        # Find first time it's within band and stays
        for i in range(len(settled)):
            if np.all(settled[i:]):
                settling_time = time_after[i] - t_step
                break
        else:
            settling_time = time_after[-1] - t_step
    else:
        settling_time = time_after[-1] - t_step
    
    return {
        'overshoot': max(0, overshoot),
        'settling_time': settling_time,
        'peak_value': peak_value,
        'peak_time': peak_time
    }


def print_elevator_sweep_table(results: Dict) -> str:
    """Format elevator sweep results as a table."""
    lines = [
        "\nElevator Control Power Sweep (M_de scaling)",
        "-" * 70,
        f"{'Scale':<8} {'M_de':<12} {'z_sp':<8} {'wn_sp':<10} {'Overshoot':<12} {'Settle(s)':<10}",
        "-" * 70
    ]
    
    for scale in sorted(results.keys()):
        r = results[scale]
        modes = r['modes']
        
        sp = modes.get('short_period', {})
        zeta_sp = sp.get('zeta', 0)
        wn_sp = sp.get('omega_n', 0)
        
        # Compute q metrics
        metrics = compute_response_metrics(r['time'], r['response']['q'], t_step=1.0)
        
        lines.append(
            f"{scale:<8.1f} {r['M_de']:<12.4f} {zeta_sp:<8.3f} {wn_sp:<10.3f} "
            f"{metrics['overshoot']:<12.1f} {metrics['settling_time']:<10.2f}"
        )
    
    lines.append("-" * 70)
    return "\n".join(lines)


def print_aileron_sweep_table(results: Dict) -> str:
    """Format aileron sweep results as a table."""
    lines = [
        "\nAileron Control Power Sweep (L_da scaling)",
        "-" * 60,
        f"{'Scale':<8} {'L_da':<12} {'tau_roll':<12} {'z_DR':<10} {'wn_DR':<10}",
        "-" * 60
    ]
    
    for scale in sorted(results.keys()):
        r = results[scale]
        
        tau_roll = r['roll']['tau'] if r['roll'] else 0
        zeta_dr = r['dutch_roll']['zeta'] if r['dutch_roll'] else 0
        wn_dr = r['dutch_roll']['omega_n'] if r['dutch_roll'] else 0
        
        lines.append(
            f"{scale:<8.1f} {r['L_da']:<12.4f} {tau_roll:<12.2f} {zeta_dr:<10.3f} {wn_dr:<10.3f}"
        )
    
    lines.append("-" * 60)
    return "\n".join(lines)


def print_rudder_sweep_table(results: Dict) -> str:
    """Format rudder sweep results as a table."""
    lines = [
        "\nRudder Control Power Sweep (N_dr scaling)",
        "-" * 50,
        f"{'Scale':<8} {'N_dr':<12} {'z_DR':<10} {'wn_DR':<10} {'tau_spiral':<12}",
        "-" * 50
    ]
    
    for scale in sorted(results.keys()):
        r = results[scale]
        
        zeta_dr = r['dutch_roll']['zeta'] if r['dutch_roll'] else 0
        wn_dr = r['dutch_roll']['omega_n'] if r['dutch_roll'] else 0
        tau_spiral = r['spiral']['tau'] if r['spiral'] else 0
        
        lines.append(
            f"{scale:<8.1f} {r['N_dr']:<12.4f} {zeta_dr:<10.3f} {wn_dr:<10.3f} {tau_spiral:<12.2f}"
        )
    
    lines.append("-" * 50)
    return "\n".join(lines)
