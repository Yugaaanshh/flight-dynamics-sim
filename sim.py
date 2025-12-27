"""
Simulation Utilities

Control input generators and simulation runner using scipy.integrate.solve_ivp.

Provides step, doublet, and ramp input functions for control surface deflections,
plus a wrapper for time-domain simulation of linear state-space models with
coefficient logging.

References:
    Roskam, "Airplane Flight Dynamics", Chapter 5
    - Step/doublet inputs for mode identification
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable, Tuple, Dict, List
from dataclasses import dataclass, field

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eom.longitudinal import LongitudinalLinearModel


@dataclass
class SimulationResult:
    """
    Container for simulation results including states, coefficients, and controls.
    
    Provides the complete picture of control surface → coefficient → motion chain.
    """
    t: np.ndarray                           # Time vector (s)
    y: np.ndarray                           # State history [4, n_steps]
    delta_e: np.ndarray                     # Elevator history (rad)
    CL: np.ndarray = field(default=None)    # Lift coefficient history
    CD: np.ndarray = field(default=None)    # Drag coefficient history
    Cm: np.ndarray = field(default=None)    # Moment coefficient history
    
    @property
    def u(self) -> np.ndarray:
        """Forward velocity perturbation (m/s)"""
        return self.y[0]
    
    @property
    def alpha(self) -> np.ndarray:
        """Angle of attack perturbation (rad)"""
        return self.y[1]
    
    @property
    def q(self) -> np.ndarray:
        """Pitch rate perturbation (rad/s)"""
        return self.y[2]
    
    @property
    def theta(self) -> np.ndarray:
        """Pitch angle perturbation (rad)"""
        return self.y[3]


def elevator_step(t: float, t_step: float = 1.0, amplitude: float = -0.087) -> float:
    """
    Elevator step input function.
    
    Args:
        t: Current time (s)
        t_step: Time at which step occurs (s)
        amplitude: Step magnitude (rad), default -5° (nose-up, TEU)
        
    Returns:
        Elevator deflection (rad)
        
    Note:
        Negative amplitude = trailing edge up = nose-up pitch command
    """
    return amplitude if t >= t_step else 0.0


def elevator_doublet(t: float, t_start: float = 1.0, 
                     duration: float = 2.0, amplitude: float = 0.087) -> float:
    """
    Elevator doublet input (positive then negative pulse).
    
    Used for modal identification per Roskam Chapter 10.
    
    Args:
        t: Current time (s)
        t_start: Start time of doublet (s)
        duration: Total duration of doublet (s)
        amplitude: Magnitude of each pulse (rad)
        
    Returns:
        Elevator deflection (rad)
    """
    if t_start <= t < t_start + duration / 2:
        return amplitude
    elif t_start + duration / 2 <= t < t_start + duration:
        return -amplitude
    return 0.0


def elevator_ramp(t: float, t_start: float = 1.0, 
                  t_end: float = 3.0, amplitude: float = -0.087) -> float:
    """
    Elevator ramp input.
    
    Args:
        t: Current time (s)
        t_start: Ramp start time (s)
        t_end: Time when ramp reaches full amplitude (s)
        amplitude: Final deflection (rad)
        
    Returns:
        Elevator deflection (rad)
    """
    if t < t_start:
        return 0.0
    elif t >= t_end:
        return amplitude
    else:
        return amplitude * (t - t_start) / (t_end - t_start)


def simulate_step_response(
    model: LongitudinalLinearModel,
    x0: np.ndarray = None,
    t_span: Tuple[float, float] = (0.0, 20.0),
    t_step: float = 1.0,
    amplitude: float = -0.087,
    dt: float = 0.01
) -> SimulationResult:
    """
    Simulate longitudinal response to elevator step input.
    
    Uses scipy.integrate.solve_ivp with RK45 method.
    Also logs aerodynamic coefficients to show the complete chain:
    δe → ΔCm → q̇ → motion
    
    Args:
        model: LongitudinalLinearModel instance
        x0: Initial state [Δu, Δα, Δq, Δθ], defaults to zeros
        t_span: (t_start, t_end) simulation time span (s)
        t_step: Time of step input (s)
        amplitude: Elevator step magnitude (rad)
        dt: Output time step (s)
        
    Returns:
        SimulationResult with states, controls, and coefficients
        
    References:
        Roskam Chapter 5, Section 5.4 (time response analysis)
    """
    if x0 is None:
        x0 = np.zeros(4)
    
    def dynamics(t, x):
        delta_e = elevator_step(t, t_step, amplitude)
        return model.dynamics(t, x, delta_e)
    
    t_eval = np.linspace(t_span[0], t_span[1], int((t_span[1]-t_span[0])/dt) + 1)
    
    sol = solve_ivp(
        dynamics,
        t_span,
        x0,
        method='RK45',
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10
    )
    
    # Compute coefficients at each time step
    n_steps = len(sol.t)
    CL = np.zeros(n_steps)
    CD = np.zeros(n_steps)
    Cm = np.zeros(n_steps)
    delta_e_hist = np.zeros(n_steps)
    
    for i, t in enumerate(sol.t):
        delta_e_hist[i] = elevator_step(t, t_step, amplitude)
        coeffs = model.compute_coefficients(sol.y[:, i], delta_e_hist[i])
        CL[i] = coeffs['CL']
        CD[i] = coeffs['CD']
        Cm[i] = coeffs['Cm']
    
    return SimulationResult(
        t=sol.t,
        y=sol.y,
        delta_e=delta_e_hist,
        CL=CL,
        CD=CD,
        Cm=Cm
    )


def simulate_with_input(
    model: LongitudinalLinearModel,
    control_func: Callable[[float], float],
    x0: np.ndarray = None,
    t_span: Tuple[float, float] = (0.0, 20.0),
    dt: float = 0.01
) -> SimulationResult:
    """
    Simulate longitudinal response with arbitrary control input function.
    
    Args:
        model: LongitudinalLinearModel instance
        control_func: Function f(t) -> delta_e
        x0: Initial state, defaults to zeros
        t_span: Simulation time span (s)
        dt: Output time step (s)
        
    Returns:
        SimulationResult with states, controls, and coefficients
    """
    if x0 is None:
        x0 = np.zeros(4)
    
    def dynamics(t, x):
        delta_e = control_func(t)
        return model.dynamics(t, x, delta_e)
    
    t_eval = np.linspace(t_span[0], t_span[1], int((t_span[1]-t_span[0])/dt) + 1)
    
    sol = solve_ivp(
        dynamics,
        t_span,
        x0,
        method='RK45',
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10
    )
    
    # Compute coefficients
    n_steps = len(sol.t)
    CL = np.zeros(n_steps)
    CD = np.zeros(n_steps)
    Cm = np.zeros(n_steps)
    delta_e_hist = np.zeros(n_steps)
    
    for i, t in enumerate(sol.t):
        delta_e_hist[i] = control_func(t)
        coeffs = model.compute_coefficients(sol.y[:, i], delta_e_hist[i])
        CL[i] = coeffs['CL']
        CD[i] = coeffs['CD']
        Cm[i] = coeffs['Cm']
    
    return SimulationResult(
        t=sol.t,
        y=sol.y,
        delta_e=delta_e_hist,
        CL=CL,
        CD=CD,
        Cm=Cm
    )
