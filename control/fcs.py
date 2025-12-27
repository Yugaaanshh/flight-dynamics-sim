"""
Flight Control System - PID Controllers and SAS

Implements Stability Augmentation System (SAS) with PID controllers
for improved handling qualities per Roskam Chapter 6 / Zipfel Chapter 11.

Features:
- Generic PID controller class
- Longitudinal SAS: Pitch rate (q) damper
- Lateral SAS: Roll angle hold
- Rate/position limiters

Usage:
    from control.fcs import LongitudinalSAS, LateralSAS
    
    long_sas = LongitudinalSAS(Kp_q=2.0, Kd_q=1.0)
    delta_e = long_sas.update(q_measured, dt)

References:
    Roskam, "Airplane Flight Dynamics", Chapter 6 (SAS design)
    Zipfel, "Modeling and Simulation", Chapter 11 (Autopilots)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class PIDGains:
    """PID controller gains."""
    Kp: float = 1.0    # Proportional gain
    Ki: float = 0.0    # Integral gain
    Kd: float = 0.0    # Derivative gain


class PID:
    """
    Standard PID controller with anti-windup and output limits.
    
    u(t) = Kp * e(t) + Ki * integral(e) + Kd * de/dt
    
    Features:
    - Integral anti-windup (clamping)
    - Output saturation
    - Derivative filtering (optional)
    """
    
    def __init__(
        self,
        Kp: float = 1.0,
        Ki: float = 0.0,
        Kd: float = 0.0,
        u_min: float = -np.inf,
        u_max: float = np.inf,
        i_max: float = 10.0  # Integrator limit
    ):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.u_min = u_min
        self.u_max = u_max
        self.i_max = i_max
        
        # State
        self.integrator = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0
        self.initialized = False
    
    def reset(self):
        """Reset controller state."""
        self.integrator = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0
        self.initialized = False
    
    def update(self, error: float, dt: float) -> float:
        """
        Compute control output for given error.
        
        Args:
            error: e(t) = setpoint - measurement
            dt: Time step (s)
            
        Returns:
            u: Control output (saturated)
        """
        if dt <= 0:
            return 0.0
        
        # Proportional term
        P = self.Kp * error
        
        # Integral term with anti-windup
        self.integrator += error * dt
        self.integrator = np.clip(self.integrator, -self.i_max, self.i_max)
        I = self.Ki * self.integrator
        
        # Derivative term (with simple filtering)
        if self.initialized:
            derivative = (error - self.prev_error) / dt
            # Low-pass filter: 0.2 * new + 0.8 * old
            derivative = 0.2 * derivative + 0.8 * self.prev_derivative
            self.prev_derivative = derivative
        else:
            derivative = 0.0
            self.initialized = True
        
        D = self.Kd * derivative
        
        self.prev_error = error
        
        # Total output with saturation
        u = P + I + D
        u = np.clip(u, self.u_min, self.u_max)
        
        return u


class LongitudinalSAS:
    """
    Longitudinal Stability Augmentation System.
    
    Implements pitch rate (q) damping to improve short-period damping.
    Target: Increase zeta_sp from ~0.3 (bare airframe) to ~0.7.
    
    Control Law:
        delta_e_cmd = delta_e_pilot + Kp_q * (0 - q) + Kd_q * q_dot
        
    The q-damper feeds back pitch rate to elevator to add artificial
    pitch damping (effectively increasing Mq).
    
    References:
        Roskam Ch6: "Pitch damper increases Cmq effective"
    """
    
    def __init__(
        self,
        Kp_q: float = 2.0,      # Pitch rate gain (rad/s -> rad)
        Kd_q: float = 0.5,      # Pitch accel gain (rad/s^2 -> rad)
        Kp_theta: float = 0.0,  # Optional pitch attitude hold
        de_max: float = 0.35,   # Max deflection (rad, ~20 deg)
        rate_limit: float = 1.0  # Rad/s deflection rate limit
    ):
        self.Kp_q = Kp_q
        self.Kd_q = Kd_q
        self.Kp_theta = Kp_theta
        self.de_max = de_max
        self.rate_limit = rate_limit
        
        # Create PID for q-feedback (mainly P+D, no I for rate damper)
        self.q_pid = PID(Kp=Kp_q, Ki=0.0, Kd=Kd_q, 
                         u_min=-de_max, u_max=de_max)
        
        # Optional theta-hold PID
        self.theta_pid = PID(Kp=Kp_theta, Ki=0.1, Kd=0.0,
                             u_min=-0.1, u_max=0.1)
        
        self.prev_de = 0.0
    
    def reset(self):
        """Reset controller state."""
        self.q_pid.reset()
        self.theta_pid.reset()
        self.prev_de = 0.0
    
    def update(
        self,
        q: float,
        theta: float,
        q_cmd: float = 0.0,
        theta_cmd: float = 0.0,
        de_pilot: float = 0.0,
        dt: float = 0.01
    ) -> float:
        """
        Compute elevator command.
        
        Args:
            q: Measured pitch rate (rad/s)
            theta: Measured pitch angle (rad)
            q_cmd: Commanded pitch rate (rad/s), usually 0 for damper
            theta_cmd: Commanded pitch angle (rad)
            de_pilot: Pilot elevator input (rad)
            dt: Time step (s)
            
        Returns:
            delta_e: Total elevator command (rad)
        """
        # Pitch rate damper: fight any q to reduce oscillations
        q_error = q_cmd - q
        de_q = self.q_pid.update(q_error, dt)
        
        # Optional pitch attitude hold
        theta_error = theta_cmd - theta
        de_theta = self.theta_pid.update(theta_error, dt) if self.Kp_theta > 0 else 0.0
        
        # Total command
        de_total = de_pilot + de_q + de_theta
        
        # Rate limit
        de_rate = (de_total - self.prev_de) / dt if dt > 0 else 0
        if abs(de_rate) > self.rate_limit:
            de_total = self.prev_de + np.sign(de_rate) * self.rate_limit * dt
        
        # Position limit
        de_total = np.clip(de_total, -self.de_max, self.de_max)
        
        self.prev_de = de_total
        return de_total


class LateralSAS:
    """
    Lateral Stability Augmentation System.
    
    Implements roll angle hold and yaw damper.
    Target: Fast roll response (tau < 1.5s), wings-level hold.
    
    Control Laws:
        delta_a_cmd = Kp_phi * (phi_cmd - phi) + Kp_p * (0 - p)
        delta_r_cmd = Kp_r * (0 - r)  # Yaw damper for Dutch roll
    
    References:
        Roskam Ch6: "Roll damper and yaw damper design"
    """
    
    def __init__(
        self,
        Kp_phi: float = 1.0,    # Roll angle gain
        Kp_p: float = 0.5,      # Roll rate damper gain
        Kp_r: float = 1.0,      # Yaw rate damper gain
        da_max: float = 0.35,   # Max aileron (rad, ~20 deg)
        dr_max: float = 0.35,   # Max rudder (rad)
    ):
        self.Kp_phi = Kp_phi
        self.Kp_p = Kp_p
        self.Kp_r = Kp_r
        self.da_max = da_max
        self.dr_max = dr_max
        
        # Roll angle hold + rate damper
        self.phi_pid = PID(Kp=Kp_phi, Ki=0.05, Kd=0.0,
                           u_min=-da_max, u_max=da_max)
        self.p_pid = PID(Kp=Kp_p, Ki=0.0, Kd=0.0,
                         u_min=-da_max, u_max=da_max)
        
        # Yaw damper
        self.r_pid = PID(Kp=Kp_r, Ki=0.0, Kd=0.2,
                         u_min=-dr_max, u_max=dr_max)
    
    def reset(self):
        """Reset all controllers."""
        self.phi_pid.reset()
        self.p_pid.reset()
        self.r_pid.reset()
    
    def update(
        self,
        phi: float,
        p: float,
        r: float,
        phi_cmd: float = 0.0,
        da_pilot: float = 0.0,
        dr_pilot: float = 0.0,
        dt: float = 0.01
    ) -> Tuple[float, float]:
        """
        Compute aileron and rudder commands.
        
        Args:
            phi: Roll angle (rad)
            p: Roll rate (rad/s)
            r: Yaw rate (rad/s)
            phi_cmd: Commanded roll angle (rad)
            da_pilot: Pilot aileron input (rad)
            dr_pilot: Pilot rudder input (rad)
            dt: Time step (s)
            
        Returns:
            (delta_a, delta_r): Control commands (rad)
        """
        # Roll angle hold
        phi_error = phi_cmd - phi
        da_phi = self.phi_pid.update(phi_error, dt)
        
        # Roll rate damper
        p_error = 0.0 - p  # Damp any roll rate
        da_p = self.p_pid.update(p_error, dt)
        
        # Yaw damper (for Dutch roll)
        r_error = 0.0 - r  # Damp any yaw rate
        dr_r = self.r_pid.update(r_error, dt)
        
        # Combine
        da_total = da_pilot + da_phi + da_p
        dr_total = dr_pilot + dr_r
        
        # Saturate
        da_total = np.clip(da_total, -self.da_max, self.da_max)
        dr_total = np.clip(dr_total, -self.dr_max, self.dr_max)
        
        return da_total, dr_total


class FlightControlSystem:
    """
    Complete Flight Control System combining longitudinal and lateral SAS.
    
    Provides single interface for closed-loop control.
    """
    
    def __init__(
        self,
        long_sas: Optional[LongitudinalSAS] = None,
        lat_sas: Optional[LateralSAS] = None
    ):
        self.long_sas = long_sas or LongitudinalSAS()
        self.lat_sas = lat_sas or LateralSAS()
        self.enabled = True
    
    def reset(self):
        """Reset all controllers."""
        self.long_sas.reset()
        self.lat_sas.reset()
    
    def update(
        self,
        state: dict,
        commands: dict,
        pilot_inputs: dict,
        dt: float
    ) -> dict:
        """
        Compute all control surface commands.
        
        Args:
            state: {'q', 'theta', 'phi', 'p', 'r', ...}
            commands: {'q_cmd', 'theta_cmd', 'phi_cmd', ...}
            pilot_inputs: {'de_pilot', 'da_pilot', 'dr_pilot', 'thrust'}
            dt: Time step
            
        Returns:
            {'delta_e', 'delta_a', 'delta_r', 'thrust'}
        """
        if not self.enabled:
            return {
                'delta_e': pilot_inputs.get('de_pilot', 0),
                'delta_a': pilot_inputs.get('da_pilot', 0),
                'delta_r': pilot_inputs.get('dr_pilot', 0),
                'thrust': pilot_inputs.get('thrust', 0)
            }
        
        # Longitudinal
        delta_e = self.long_sas.update(
            q=state.get('q', 0),
            theta=state.get('theta', 0),
            q_cmd=commands.get('q_cmd', 0),
            theta_cmd=commands.get('theta_cmd', 0),
            de_pilot=pilot_inputs.get('de_pilot', 0),
            dt=dt
        )
        
        # Lateral
        delta_a, delta_r = self.lat_sas.update(
            phi=state.get('phi', 0),
            p=state.get('p', 0),
            r=state.get('r', 0),
            phi_cmd=commands.get('phi_cmd', 0),
            da_pilot=pilot_inputs.get('da_pilot', 0),
            dr_pilot=pilot_inputs.get('dr_pilot', 0),
            dt=dt
        )
        
        return {
            'delta_e': delta_e,
            'delta_a': delta_a,
            'delta_r': delta_r,
            'thrust': pilot_inputs.get('thrust', 0)
        }


if __name__ == "__main__":
    # Test PID
    pid = PID(Kp=2.0, Ki=0.1, Kd=0.5, u_min=-1, u_max=1)
    
    print("PID Test:")
    error = 1.0
    for i in range(10):
        u = pid.update(error, dt=0.1)
        error *= 0.8  # Decay error
        print(f"  t={i*0.1:.1f}s: error={error:.3f}, u={u:.3f}")
    
    print("\nLongitudinal SAS Test:")
    long_sas = LongitudinalSAS(Kp_q=2.0, Kd_q=0.5)
    q = 0.1  # 0.1 rad/s pitch rate
    for i in range(5):
        de = long_sas.update(q=q, theta=0.05, dt=0.1)
        print(f"  q={np.degrees(q):.2f} deg/s -> de={np.degrees(de):.2f} deg")
        q *= 0.7
    
    print("\nFCS Ready!")
