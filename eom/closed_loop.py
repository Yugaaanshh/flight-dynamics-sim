"""
Closed-Loop Simulation Wrapper

Combines 6-DOF dynamics with Flight Control System for
closed-loop stability augmentation simulation.

Usage:
    from eom.closed_loop import simulate_closed_loop
    
    result = simulate_closed_loop(
        model, trim, fcs, t_span, 
        pilot_input_func
    )

References:
    Roskam Chapter 6 - SAS simulation
    Zipfel Chapter 11 - Autopilot integration
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eom.six_dof import SixDoFModel, Controls
from control.fcs import FlightControlSystem, LongitudinalSAS, LateralSAS

# State vector indices (12-state): [u,v,w, p,q,r, phi,theta,psi, x,y,z]
IDX_U, IDX_V, IDX_W = 0, 1, 2
IDX_P, IDX_Q, IDX_R = 3, 4, 5
IDX_PHI, IDX_THETA, IDX_PSI = 6, 7, 8
IDX_X, IDX_Y, IDX_Z = 9, 10, 11


@dataclass
class ClosedLoopResult:
    """Results from closed-loop simulation."""
    t: np.ndarray              # Time vector
    x: np.ndarray              # State history (12 x N)
    u: np.ndarray              # Control history (4 x N): [de, da, dr, thrust]
    q: np.ndarray              # Pitch rate history
    theta: np.ndarray          # Pitch angle history
    phi: np.ndarray            # Roll angle history
    p: np.ndarray              # Roll rate history
    alpha: np.ndarray          # Angle of attack history
    

def extract_states(x: np.ndarray) -> dict:
    """Extract named states from state vector."""
    return {
        'u': x[IDX_U],
        'v': x[IDX_V],
        'w': x[IDX_W],
        'p': x[IDX_P],
        'q': x[IDX_Q],
        'r': x[IDX_R],
        'phi': x[IDX_PHI],
        'theta': x[IDX_THETA],
        'psi': x[IDX_PSI],
    }


def compute_alpha(u: float, w: float) -> float:
    """Compute angle of attack from body velocities."""
    if abs(u) < 0.1:
        return 0.0
    return np.arctan2(w, u)


def simulate_closed_loop(
    model: SixDoFModel,
    x0: np.ndarray,
    u_trim: np.ndarray,
    fcs: FlightControlSystem,
    t_span: Tuple[float, float],
    pilot_input_func: Optional[Callable[[float], dict]] = None,
    commands_func: Optional[Callable[[float], dict]] = None,
    dt: float = 0.01,
    method: str = 'RK45'
) -> ClosedLoopResult:
    """
    Simulate closed-loop 6-DOF dynamics with FCS.
    
    Args:
        model: SixDoFModel instance
        x0: Initial state vector (12,)
        u_trim: Trim control vector [de, da, dr, thrust]
        fcs: FlightControlSystem instance
        t_span: (t_start, t_end)
        pilot_input_func: f(t) -> {'de_pilot': val, ...}
        commands_func: f(t) -> {'phi_cmd': val, ...}
        dt: Integration step
        method: ODE solver method
        
    Returns:
        ClosedLoopResult with time histories
    """
    fcs.reset()
    
    # Storage for control history (solve_ivp doesn't give us intermediate u)
    control_log = []
    time_log = []
    
    # Default functions
    if pilot_input_func is None:
        pilot_input_func = lambda t: {
            'de_pilot': 0.0,
            'da_pilot': 0.0,
            'dr_pilot': 0.0,
            'thrust': u_trim[3]
        }
    
    if commands_func is None:
        commands_func = lambda t: {
            'q_cmd': 0.0,
            'theta_cmd': x0[IDX_THETA],
            'phi_cmd': 0.0
        }
    
    prev_t = t_span[0]
    
    def dynamics(t, x):
        nonlocal prev_t
        
        # Compute dt for controller
        ctrl_dt = max(t - prev_t, dt)
        prev_t = t
        
        # Extract states for FCS
        state = extract_states(x)
        
        # Get commands and pilot inputs
        commands = commands_func(t)
        pilot_inputs = pilot_input_func(t)
        
        # Compute control through FCS
        controls = fcs.update(state, commands, pilot_inputs, ctrl_dt)
        
        # Build control vector
        u = np.array([
            controls['delta_e'],
            controls['delta_a'],
            controls['delta_r'],
            controls['thrust']
        ])
        
        # Log
        control_log.append(u.copy())
        time_log.append(t)
        
        # Compute state derivatives using Controls dataclass
        ctrl = Controls(
            delta_e=controls['delta_e'],
            delta_a=controls['delta_a'],
            delta_r=controls['delta_r'],
            thrust=controls['thrust']
        )
        x_dot = model.dynamics(t, x, ctrl)
        
        return x_dot
    
    # Solve ODE
    t_eval = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(dynamics, t_span, x0, method=method, t_eval=t_eval, max_step=dt)
    
    # Build result
    N = len(sol.t)
    
    # Interpolate control log to match solution times
    if len(control_log) > 0:
        control_log = np.array(control_log)
        time_log = np.array(time_log)
        
        # Simple nearest-neighbor interpolation
        u_interp = np.zeros((4, N))
        for i, t in enumerate(sol.t):
            idx = np.argmin(np.abs(time_log - t))
            u_interp[:, i] = control_log[idx]
    else:
        u_interp = np.tile(u_trim, (N, 1)).T
    
    # Extract time histories
    q = sol.y[IDX_Q, :]
    theta = sol.y[IDX_THETA, :]
    phi = sol.y[IDX_PHI, :]
    p = sol.y[IDX_P, :]
    alpha = np.array([compute_alpha(sol.y[IDX_U, i], sol.y[IDX_W, i]) for i in range(N)])
    
    return ClosedLoopResult(
        t=sol.t,
        x=sol.y,
        u=u_interp,
        q=q,
        theta=theta,
        phi=phi,
        p=p,
        alpha=alpha
    )


def simulate_open_loop(
    model: SixDoFModel,
    x0: np.ndarray,
    u_trim: np.ndarray,
    t_span: Tuple[float, float],
    control_func: Optional[Callable[[float], np.ndarray]] = None,
    dt: float = 0.01,
    method: str = 'RK45'
) -> ClosedLoopResult:
    """
    Simulate open-loop 6-DOF dynamics (no FCS).
    
    Args:
        model: SixDoFModel instance
        x0: Initial state
        u_trim: Trim controls
        t_span: Time span
        control_func: f(t) -> [de, da, dr, thrust]
        dt: Time step
        
    Returns:
        ClosedLoopResult (same format as closed-loop for comparison)
    """
    if control_func is None:
        control_func = lambda t: u_trim
    
    control_log = []
    
    def dynamics(t, x):
        u = control_func(t)
        control_log.append(u.copy())
        ctrl = Controls(delta_e=u[0], delta_a=u[1], delta_r=u[2], thrust=u[3])
        return model.dynamics(t, x, ctrl)
    
    t_eval = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(dynamics, t_span, x0, method=method, t_eval=t_eval, max_step=dt)
    
    N = len(sol.t)
    
    if len(control_log) > 0:
        u_interp = np.zeros((4, N))
        for i in range(min(N, len(control_log))):
            u_interp[:, i] = control_log[i]
    else:
        u_interp = np.tile(u_trim, (N, 1)).T
    
    q = sol.y[IDX_Q, :]
    theta = sol.y[IDX_THETA, :]
    phi = sol.y[IDX_PHI, :]
    p = sol.y[IDX_P, :]
    alpha = np.array([compute_alpha(sol.y[IDX_U, i], sol.y[IDX_W, i]) for i in range(N)])
    
    return ClosedLoopResult(
        t=sol.t, x=sol.y, u=u_interp,
        q=q, theta=theta, phi=phi, p=p, alpha=alpha
    )


if __name__ == "__main__":
    print("Closed-loop simulation module ready.")
    print("Use simulate_closed_loop() or simulate_open_loop()")
