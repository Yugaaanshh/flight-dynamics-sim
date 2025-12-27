"""
Longitudinal Equations of Motion - Linear Small-Perturbation Model

Implements the linearized longitudinal dynamics following Roskam
Chapter 5, Equations 5.30 and Table 5.2.

State vector: x = [Δu, Δα, Δq, Δθ]ᵀ
    Δu     - perturbation in forward velocity (m/s)
    Δα     - perturbation in angle of attack (rad)
    Δq     - perturbation in pitch rate (rad/s)
    Δθ     - perturbation in pitch angle (rad)

Control input: u = Δδe (elevator deflection perturbation, rad)

Assumptions:
- Small perturbations from trimmed flight
- Stability axes for aerodynamic coefficients
- Flat Earth, constant mass
- α-state formulation (w ≈ U0·α)
- Decoupled longitudinal and lateral-directional dynamics

References:
    Roskam, "Airplane Flight Dynamics", Chapter 5
    Table 5.2: State-space formulation
    Appendix B: Aircraft data
"""

import numpy as np
from typing import Tuple, Dict, Any

from config import LongitudinalDerivatives
from aero.coefficients import compute_longitudinal_coefficients, AeroCoefficients


class LongitudinalLinearModel:
    """
    Linear state-space model for longitudinal dynamics.
    
    Constructs A and B matrices from dimensional derivatives
    per Roskam Chapter 5, Table 5.2.
    
    State: x = [Δu, Δα, Δq, Δθ]ᵀ
    Input: u = Δδe
    
    ẋ = Ax + Bu
    
    Also computes aerodynamic coefficients CL, Cm for visualization
    of the control surface → coefficient → motion chain.
    """
    
    def __init__(self, derivs: LongitudinalDerivatives, 
                 aero_coeffs: AeroCoefficients = None):
        """
        Initialize model with dimensional derivatives.
        
        Args:
            derivs: LongitudinalDerivatives dataclass with stability
                    and control derivatives + trim conditions
            aero_coeffs: Optional AeroCoefficients for CL/Cm computation.
                         If None, will be estimated from dimensional derivs.
        """
        self.derivs = derivs
        self.A, self.B = self._build_state_space(derivs)
        self.state_names = ['Δu (m/s)', 'Δα (rad)', 'Δq (rad/s)', 'Δθ (rad)']
        
        # Set up coefficient computation
        if aero_coeffs is not None:
            self.aero_coeffs = aero_coeffs
        else:
            # Estimate from dimensional derivatives using Roskam relationships
            self.aero_coeffs = self._estimate_aero_coeffs(derivs)
        
    def _build_state_space(self, d: LongitudinalDerivatives) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build A, B matrices from Roskam Table 5.2.
        
        The longitudinal equations in matrix form:
        
        ┌ u̇  ┐   ┌ Xu      Xα         0        -g·cos(θ0) ┐┌ u ┐   ┌ Xδe     ┐
        │ α̇  │ = │ Zu/U0   Zα/U0    1+Zq/U0   -g·sin(θ0)/U0 ││ α │ + │ Zδe/U0  │ δe
        │ q̇  │   │ Mu      Mα        Mq         0         ││ q │   │ Mδe     │
        └ θ̇  ┘   └ 0       0         1          0         ┘└ θ ┘   └ 0       ┘
        
        Note: The (1 + Zq/U0) term in A[1,2] accounts for the kinematic
        coupling between q and α̇ in the α-state formulation.
        
        References:
            Roskam Chapter 5, Equations 5.30, Table 5.2
        """
        U0 = d.U0
        g = d.g
        theta0 = d.theta0
        
        A = np.array([
            [d.X_u,       d.X_alpha,      0.0,              -g * np.cos(theta0)],
            [d.Z_u / U0,  d.Z_alpha / U0, 1.0 + d.Z_q / U0, -g * np.sin(theta0) / U0],
            [d.M_u,       d.M_alpha,      d.M_q,             0.0],
            [0.0,         0.0,            1.0,               0.0]
        ])
        
        B = np.array([
            [d.X_delta_e],
            [d.Z_delta_e / U0],
            [d.M_delta_e],
            [0.0]
        ])
        
        return A, B
    
    def _estimate_aero_coeffs(self, d: LongitudinalDerivatives) -> AeroCoefficients:
        """
        Estimate non-dimensional coefficients from dimensional derivatives.
        
        Uses inverse of Roskam Table 5.1 relationships:
            C_Lα = -m·Z_α / (q̄·S)
            C_mα = I_yy·M_α / (q̄·S·c̄)
        
        This provides approximate coefficients for visualization purposes.
        """
        q_bar = 0.5 * d.rho * d.U0**2 if hasattr(d, 'rho') else 0.5 * 1.225 * d.U0**2
        
        # Use defaults if geometry not specified
        S = getattr(d, 'S', 20.0)
        c_bar = getattr(d, 'c_bar', 2.0)
        m = getattr(d, 'm', 1000.0)
        Iyy = getattr(d, 'Iyy', 5000.0)
        
        # Convert dimensional to non-dimensional (Roskam Table 5.1 inverse)
        CL_alpha = -m * d.Z_alpha / (q_bar * S) if q_bar * S > 1e-6 else 5.0
        CL_delta_e = -m * d.Z_delta_e / (q_bar * S) if q_bar * S > 1e-6 else 0.3
        CL_q = -m * d.Z_q / (q_bar * S * c_bar / (2 * d.U0)) if q_bar * S > 1e-6 else 4.0
        
        Cm_alpha = Iyy * d.M_alpha / (q_bar * S * c_bar) if q_bar * S * c_bar > 1e-6 else -0.5
        Cm_delta_e = Iyy * d.M_delta_e / (q_bar * S * c_bar) if q_bar * S * c_bar > 1e-6 else -1.0
        Cm_q = Iyy * d.M_q / (q_bar * S * c_bar**2 / (2 * d.U0)) if q_bar * S > 1e-6 else -10.0
        
        return AeroCoefficients(
            CL_0=0.3,           # Typical cruise CL
            CL_alpha=CL_alpha,
            CL_q=CL_q,
            CL_delta_e=CL_delta_e,
            CD_0=0.02,
            K=0.04,
            CD_delta_e=0.0,
            Cm_0=0.0,           # Trimmed
            Cm_alpha=Cm_alpha,
            Cm_q=Cm_q,
            Cm_delta_e=Cm_delta_e
        )
    
    def dynamics(self, t: float, x: np.ndarray, delta_e: float) -> np.ndarray:
        """
        Compute state derivatives for given state and control input.
        
        Args:
            t: Time (s) - unused but required for ODE solver interface
            x: State vector [Δu, Δα, Δq, Δθ]
            delta_e: Elevator deflection perturbation (rad)
            
        Returns:
            x_dot: State derivative vector
        """
        return self.A @ x + self.B.flatten() * delta_e
    
    def compute_coefficients(self, x: np.ndarray, delta_e: float) -> Dict[str, float]:
        """
        Compute aerodynamic coefficients for current state.
        
        This demonstrates the control surface → coefficient → motion chain:
        δe → ΔCm → q̇ → motion
        
        Args:
            x: State vector [Δu, Δα, Δq, Δθ]
            delta_e: Elevator deflection (rad)
            
        Returns:
            Dict with CL, CD, Cm values
            
        References:
            Roskam Chapter 3, Equations 3.21, 3.46
        """
        alpha = x[1]  # Δα
        q = x[2]      # Δq
        
        CL, CD, Cm = compute_longitudinal_coefficients(
            alpha=alpha,
            q=q,
            delta_e=delta_e,
            V=self.derivs.U0,
            c_bar=getattr(self.derivs, 'c_bar', 2.0),
            coeffs=self.aero_coeffs
        )
        
        return {'CL': CL, 'CD': CD, 'Cm': Cm}
    
    def get_eigenvalues(self) -> np.ndarray:
        """
        Compute eigenvalues of the system matrix.
        
        Returns complex eigenvalues representing:
        - Short-period mode: fast, well-damped α/q oscillation
        - Phugoid mode: slow, lightly-damped u/θ oscillation
        
        References:
            Roskam Chapter 5, Section 5.2 (modal analysis)
        """
        return np.linalg.eigvals(self.A)
    
    def analyze_modes(self) -> Dict[str, Dict[str, Any]]:
        """
        Extract modal characteristics from eigenvalues.
        
        Returns:
            dict with short_period and phugoid mode properties:
            - eigenvalue: complex eigenvalue
            - omega_n: natural frequency (rad/s)
            - zeta: damping ratio
            - period: oscillation period (s)
        
        References:
            Roskam Chapter 5, Section 5.2.2 (mode identification)
        """
        eigs = self.get_eigenvalues()
        
        # Separate into complex conjugate pairs
        # Sort by imaginary part magnitude (frequency)
        sorted_eigs = sorted(eigs, key=lambda x: abs(x.imag), reverse=True)
        
        modes = {}
        
        # Short period: higher frequency pair (indices 0, 1)
        sp_eig = sorted_eigs[0]
        sp_omega_n = abs(sp_eig)
        sp_zeta = -sp_eig.real / sp_omega_n if sp_omega_n > 1e-10 else 0.0
        sp_period = 2 * np.pi / abs(sp_eig.imag) if abs(sp_eig.imag) > 1e-10 else np.inf
        
        modes['short_period'] = {
            'eigenvalue': sp_eig,
            'omega_n': sp_omega_n,
            'zeta': sp_zeta,
            'period': sp_period
        }
        
        # Phugoid: lower frequency pair (indices 2, 3)
        ph_eig = sorted_eigs[2]
        ph_omega_n = abs(ph_eig)
        ph_zeta = -ph_eig.real / ph_omega_n if ph_omega_n > 1e-10 else 0.0
        ph_period = 2 * np.pi / abs(ph_eig.imag) if abs(ph_eig.imag) > 1e-10 else np.inf
        
        modes['phugoid'] = {
            'eigenvalue': ph_eig,
            'omega_n': ph_omega_n,
            'zeta': ph_zeta,
            'period': ph_period
        }
        
        return modes
    
    def print_modes_table(self) -> str:
        """
        Format modal analysis as a table string.
        
        Returns:
            Formatted table showing short-period and phugoid characteristics
        """
        modes = self.analyze_modes()
        eigs = self.get_eigenvalues()
        
        lines = [
            "=" * 60,
            "LONGITUDINAL MODE ANALYSIS (Roskam Chapter 5)",
            "=" * 60,
            "",
            "Eigenvalues of A matrix:",
        ]
        
        for i, eig in enumerate(sorted(eigs, key=lambda x: -abs(x.imag))):
            lines.append(f"  λ{i+1} = {eig.real:+.4f} {eig.imag:+.4f}j")
        
        lines.extend([
            "",
            "-" * 60,
            f"{'Mode':<15} {'ωn (rad/s)':<12} {'ζ':<10} {'T (s)':<10}",
            "-" * 60,
        ])
        
        sp = modes['short_period']
        ph = modes['phugoid']
        
        lines.append(f"{'Short-Period':<15} {sp['omega_n']:<12.3f} {sp['zeta']:<10.3f} {sp['period']:<10.2f}")
        lines.append(f"{'Phugoid':<15} {ph['omega_n']:<12.3f} {ph['zeta']:<10.3f} {ph['period']:<10.2f}")
        lines.append("-" * 60)
        
        return "\n".join(lines)
