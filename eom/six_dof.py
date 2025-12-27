"""
Six Degrees of Freedom Equations of Motion

Implements the nonlinear rigid-body equations of motion in body axes
following Roskam Chapter 1, Equations 1.19 (translational) and 1.25 (rotational).

State vector (12 states):
    [u, v, w, p, q, r, φ, θ, ψ, x, y, h]
    
    u, v, w    - Body-axis velocities (m/s)
    p, q, r    - Body-axis angular rates (rad/s)
    φ, θ, ψ    - Euler angles: roll, pitch, yaw (rad)
    x, y, h    - Inertial position (North, East, Altitude) (m)

Control inputs:
    [δe, δa, δr, δT]  - Elevator, aileron, rudder, throttle

Assumptions:
- Rigid body
- Flat Earth (no Coriolis/transport effects)
- Symmetric aircraft (Ixz = 0)
- Constant mass
- Standard atmosphere or specified density

References:
    Roskam, "Airplane Flight Dynamics", Chapter 1
    Equations 1.19 (translational), 1.25 (rotational)
    Zipfel, "Modeling and Simulation of Aerospace Vehicle Dynamics"
"""

import numpy as np
from typing import Tuple, Dict, Callable
from dataclasses import dataclass

from config import FullAircraftConfig, AircraftParameters
from aero.coefficients import compute_all_coefficients, AeroCoefficients, create_full_aero_coefficients
from forces_moments import (
    compute_dynamic_pressure,
    coefficients_to_forces_moments,
    compute_total_forces,
    compute_airspeed_from_body_velocities,
    compute_alpha_beta
)


# State indices
U, V, W = 0, 1, 2       # Body velocities
P, Q, R = 3, 4, 5       # Angular rates
PHI, THETA, PSI = 6, 7, 8  # Euler angles
X_POS, Y_POS, H = 9, 10, 11  # Position


@dataclass
class Controls:
    """Control surface deflections."""
    delta_e: float = 0.0   # Elevator (rad), positive TED
    delta_a: float = 0.0   # Aileron (rad), positive right wing down
    delta_r: float = 0.0   # Rudder (rad), positive TEL (nose right)
    thrust: float = 0.0    # Thrust (N)


class SixDoFModel:
    """
    Nonlinear 6-DOF rigid-body model.
    
    Implements Roskam Chapter 1 equations in body axes with Euler angle kinematics.
    
    State: x = [u, v, w, p, q, r, φ, θ, ψ, x, y, h]ᵀ
    Controls: [δe, δa, δr, T]
    
    Equations of motion:
        Translational (Roskam 1.19):
            u̇ = rv - qw + X/m - g sin θ
            v̇ = pw - ru + Y/m + g cos θ sin φ
            ẇ = qu - pv + Z/m + g cos θ cos φ
        
        Rotational (Roskam 1.25, symmetric Ixz=0):
            ṗ = (L - (Iyy - Izz)qr) / Ixx
            q̇ = (M - (Izz - Ixx)pr) / Iyy
            ṙ = (N - (Ixx - Iyy)pq) / Izz
        
        Euler kinematics:
            φ̇ = p + (q sin φ + r cos φ) tan θ
            θ̇ = q cos φ - r sin φ
            ψ̇ = (q sin φ + r cos φ) / cos θ
        
        Position (flat Earth):
            ẋ = u cos θ cos ψ + v(sin φ sin θ cos ψ - cos φ sin ψ) + w(cos φ sin θ cos ψ + sin φ sin ψ)
            ẏ = u cos θ sin ψ + v(sin φ sin θ sin ψ + cos φ cos ψ) + w(cos φ sin θ sin ψ - sin φ cos ψ)
            ḣ = u sin θ - v sin φ cos θ - w cos φ cos θ
    """
    
    def __init__(self, config: FullAircraftConfig):
        """
        Initialize 6-DOF model.
        
        Args:
            config: FullAircraftConfig with aircraft parameters and derivatives
        """
        self.config = config
        self.params = config.params
        self.aero_coeffs = create_full_aero_coefficients(config)
        
        self.g = 9.81
        self.state_names = ['u', 'v', 'w', 'p', 'q', 'r', 'φ', 'θ', 'ψ', 'x', 'y', 'h']
        
    def dynamics(self, t: float, state: np.ndarray, controls: Controls) -> np.ndarray:
        """
        Compute state derivatives.
        
        Implements Roskam Ch1 Eqns 1.19 (translational) and 1.25 (rotational).
        
        Args:
            t: Time (s)
            state: 12-element state vector
            controls: Controls dataclass
            
        Returns:
            state_dot: 12-element state derivative vector
        """
        # Unpack state
        u, v, w = state[U], state[V], state[W]
        p, q, r = state[P], state[Q], state[R]
        phi, theta, psi = state[PHI], state[THETA], state[PSI]
        
        # Compute aerodynamic angles
        V_air = compute_airspeed_from_body_velocities(u, v, w)
        alpha, beta = compute_alpha_beta(u, v, w)
        
        # Compute dynamic pressure
        q_bar = compute_dynamic_pressure(self.config.rho, V_air)
        
        # Compute aerodynamic coefficients
        coeffs = compute_all_coefficients(
            alpha=alpha,
            beta=beta,
            p=p, q=q, r=r,
            delta_e=controls.delta_e,
            delta_a=controls.delta_a,
            delta_r=controls.delta_r,
            V=V_air,
            c_bar=self.params.c_bar,
            b=self.params.b,
            coeffs=self.aero_coeffs
        )
        
        # Compute aerodynamic forces and moments
        aero_fm = coefficients_to_forces_moments(
            coeffs=coeffs,
            alpha=alpha,
            q_bar=q_bar,
            S=self.params.S,
            b=self.params.b,
            c_bar=self.params.c_bar
        )
        
        # Add gravity and thrust
        total_fm = compute_total_forces(
            aero_fm=aero_fm,
            mass=self.params.mass,
            phi=phi,
            theta=theta,
            thrust=controls.thrust,
            g=self.g
        )
        
        # Shorthand
        m = self.params.mass
        Ixx = self.params.Ixx
        Iyy = self.params.Iyy
        Izz = self.params.Izz
        
        # =================================================================
        # Translational equations (Roskam Eqn 1.19)
        # =================================================================
        u_dot = r * v - q * w + total_fm.X / m
        v_dot = p * w - r * u + total_fm.Y / m
        w_dot = q * u - p * v + total_fm.Z / m
        
        # =================================================================
        # Rotational equations (Roskam Eqn 1.25, symmetric: Ixz = 0)
        # =================================================================
        p_dot = (total_fm.L - (Iyy - Izz) * q * r) / Ixx
        q_dot = (total_fm.M - (Izz - Ixx) * p * r) / Iyy
        r_dot = (total_fm.N - (Ixx - Iyy) * p * q) / Izz
        
        # =================================================================
        # Euler angle kinematics
        # =================================================================
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        cos_theta = np.cos(theta)
        tan_theta = np.tan(theta)
        
        # Protect against gimbal lock
        if abs(cos_theta) < 1e-6:
            cos_theta = 1e-6 * np.sign(cos_theta) if cos_theta != 0 else 1e-6
        
        phi_dot = p + (q * sin_phi + r * cos_phi) * tan_theta
        theta_dot = q * cos_phi - r * sin_phi
        psi_dot = (q * sin_phi + r * cos_phi) / cos_theta
        
        # =================================================================
        # Position kinematics (flat Earth, NED)
        # =================================================================
        sin_theta = np.sin(theta)
        sin_psi = np.sin(psi)
        cos_psi = np.cos(psi)
        
        # Direction cosine matrix elements
        x_dot = (u * cos_theta * cos_psi +
                 v * (sin_phi * sin_theta * cos_psi - cos_phi * sin_psi) +
                 w * (cos_phi * sin_theta * cos_psi + sin_phi * sin_psi))
        
        y_dot = (u * cos_theta * sin_psi +
                 v * (sin_phi * sin_theta * sin_psi + cos_phi * cos_psi) +
                 w * (cos_phi * sin_theta * sin_psi - sin_phi * cos_psi))
        
        # Altitude rate (positive up)
        h_dot = u * sin_theta - v * sin_phi * cos_theta - w * cos_phi * cos_theta
        
        return np.array([
            u_dot, v_dot, w_dot,
            p_dot, q_dot, r_dot,
            phi_dot, theta_dot, psi_dot,
            x_dot, y_dot, h_dot
        ])
    
    def get_trim_state(self, V: float = None, h: float = 10668.0) -> np.ndarray:
        """
        Get approximate trim state for level flight.
        
        Args:
            V: Airspeed (m/s), defaults to config trim speed
            h: Altitude (m)
            
        Returns:
            state: 12-element initial state
        """
        if V is None:
            V = self.config.V_trim
        
        alpha_trim = self.config.alpha_trim
        
        # Level flight: θ = α, φ = ψ = 0
        state = np.zeros(12)
        state[U] = V * np.cos(alpha_trim)
        state[W] = V * np.sin(alpha_trim)
        state[THETA] = alpha_trim
        state[H] = h
        
        return state
    
    def compute_trim_thrust(self, state: np.ndarray) -> float:
        """
        Estimate thrust required for trim (drag = thrust at level flight).
        
        Args:
            state: State vector
            
        Returns:
            thrust: Required thrust (N)
        """
        u, v, w = state[U], state[V], state[W]
        V_air = compute_airspeed_from_body_velocities(u, v, w)
        alpha, beta = compute_alpha_beta(u, v, w)
        
        q_bar = compute_dynamic_pressure(self.config.rho, V_air)
        
        # At trim, lift = weight
        CL_trim = self.params.mass * self.g / (q_bar * self.params.S)
        
        # Estimate drag using parabolic polar
        CD_trim = self.aero_coeffs.CD_0 + self.aero_coeffs.K * CL_trim**2
        
        # Thrust = Drag (at level flight)
        thrust = q_bar * self.params.S * CD_trim
        
        return thrust
    
    def get_state_dict(self, state: np.ndarray) -> Dict[str, float]:
        """Convert state array to labeled dictionary."""
        V = compute_airspeed_from_body_velocities(state[U], state[V], state[W])
        alpha, beta = compute_alpha_beta(state[U], state[V], state[W])
        
        return {
            'u': state[U], 'v': state[V], 'w': state[W],
            'p': state[P], 'q': state[Q], 'r': state[R],
            'phi': state[PHI], 'theta': state[THETA], 'psi': state[PSI],
            'x': state[X_POS], 'y': state[Y_POS], 'h': state[H],
            'V': V, 'alpha': alpha, 'beta': beta
        }


def create_6dof_eom_function(model: SixDoFModel, control_func: Callable[[float], Controls]):
    """
    Create dynamics function for scipy.integrate.solve_ivp.
    
    Args:
        model: SixDoFModel instance
        control_func: Function f(t) -> Controls
        
    Returns:
        Callable for solve_ivp
    """
    def eom(t, state):
        controls = control_func(t)
        return model.dynamics(t, state, controls)
    
    return eom
