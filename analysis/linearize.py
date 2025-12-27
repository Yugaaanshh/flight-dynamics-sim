"""
Numerical Linearization and Mode Analysis

Linearizes the nonlinear 6-DOF model about any trim state using
numerical Jacobians (central finite differences).

Computes:
    ẋ = f(x, u)
    A = ∂f/∂x |_(x*, u*)
    B = ∂f/∂u |_(x*, u*)

Analyzes eigenvalues to classify dynamic modes (Roskam Ch5):
    - Short-period
    - Phugoid
    - Dutch roll
    - Roll subsidence
    - Spiral

References:
    Roskam, "Airplane Flight Dynamics", Chapter 5
    Section 5.2: Modal analysis
    Section 5.4: Derivative sensitivity
"""

import numpy as np
from typing import List, Dict, Callable, Tuple, Optional
from dataclasses import dataclass

from eom.six_dof import SixDoFModel, Controls


@dataclass
class ModeInfo:
    """Information about a dynamic mode."""
    name: str
    eigenvalue: complex
    omega_n: float          # Natural frequency (rad/s)
    zeta: float             # Damping ratio
    period: float           # Period (s), inf for aperiodic
    time_constant: float    # Time constant (s) for real modes
    is_oscillatory: bool


def jacobian_fd(
    func: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    eps: float = 1e-5
) -> np.ndarray:
    """
    Compute Jacobian using central finite differences.
    
    Args:
        func: f(x) -> y, vector function
        x: Point at which to evaluate Jacobian
        eps: Perturbation size
        
    Returns:
        J: Jacobian matrix ∂f/∂x [m x n]
    """
    n = len(x)
    f0 = func(x)
    m = len(f0)
    
    J = np.zeros((m, n))
    
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        
        f_plus = func(x_plus)
        f_minus = func(x_minus)
        
        J[:, i] = (f_plus - f_minus) / (2 * eps)
    
    return J


def linearize_6dof(
    model_6dof: SixDoFModel,
    x_trim: np.ndarray,
    u_trim: np.ndarray,
    eps: float = 1e-5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Linearize the 6-DOF nonlinear model about a trim point.
    
    Computes:
        A = ∂f/∂x |_(x_trim, u_trim)
        B = ∂f/∂u |_(x_trim, u_trim)
    
    where ẋ = f(x, u)
    
    Args:
        model_6dof: SixDoFModel instance
        x_trim: 12-element trim state [u,v,w,p,q,r,φ,θ,ψ,x,y,h]
        u_trim: 4-element control [δe, δa, δr, T]
        eps: Finite difference step size
        
    Returns:
        A: 12x12 state matrix
        B: 12x4 input matrix
        
    References:
        Roskam Ch5, Section 5.1 (linearization)
    """
    # Create controls object from u_trim
    def make_controls(u):
        return Controls(
            delta_e=u[0],
            delta_a=u[1],
            delta_r=u[2],
            thrust=u[3]
        )
    
    controls_trim = make_controls(u_trim)
    
    # Function for A matrix: f(x) at fixed u
    def f_x(x):
        return model_6dof.dynamics(0.0, x, controls_trim)
    
    # Function for B matrix: f(u) at fixed x
    def f_u(u):
        return model_6dof.dynamics(0.0, x_trim, make_controls(u))
    
    # Compute Jacobians
    A = jacobian_fd(f_x, x_trim, eps)
    B = jacobian_fd(f_u, u_trim, eps)
    
    return A, B


def extract_reduced_matrices(A: np.ndarray, B: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Extract longitudinal and lateral-directional submatrices.
    
    Longitudinal states: u, w (or α), q, θ → indices 0, 2, 4, 7
    Lateral states: v (or β), p, r, φ, ψ → indices 1, 3, 5, 6, 8
    
    Args:
        A: Full 12x12 state matrix
        B: Full 12x4 input matrix
        
    Returns:
        Dict with 'A_long', 'A_lat', 'B_long', 'B_lat'
    """
    # Longitudinal indices: u(0), w(2), q(4), θ(7)
    long_idx = [0, 2, 4, 7]
    
    # Lateral indices: v(1), p(3), r(5), φ(6)
    lat_idx = [1, 3, 5, 6]
    
    A_long = A[np.ix_(long_idx, long_idx)]
    A_lat = A[np.ix_(lat_idx, lat_idx)]
    
    # Longitudinal control: δe (index 0)
    B_long = B[np.ix_(long_idx, [0])]
    
    # Lateral controls: δa, δr (indices 1, 2)
    B_lat = B[np.ix_(lat_idx, [1, 2])]
    
    return {
        'A_long': A_long,
        'A_lat': A_lat,
        'B_long': B_long,
        'B_lat': B_lat,
        'long_states': ['u', 'w', 'q', 'θ'],
        'lat_states': ['v', 'p', 'r', 'φ']
    }


def analyze_eigenvalue(eig: complex) -> Dict:
    """
    Extract modal characteristics from a single eigenvalue.
    
    For complex eigenvalue λ = σ ± jω:
        ωn = |λ|
        ζ = -σ/ωn
        T = 2π/ω (period)
        
    For real eigenvalue λ = σ:
        τ = -1/σ (time constant)
    """
    real = eig.real
    imag = abs(eig.imag)
    
    is_oscillatory = imag > 1e-6
    
    omega_n = abs(eig)
    zeta = -real / omega_n if omega_n > 1e-10 else 0.0
    period = 2 * np.pi / imag if is_oscillatory else np.inf
    time_const = -1.0 / real if abs(real) > 1e-10 else np.inf
    
    return {
        'eigenvalue': eig,
        'omega_n': omega_n,
        'zeta': zeta,
        'period': period,
        'time_constant': time_const,
        'is_oscillatory': is_oscillatory
    }


def classify_longitudinal_modes(A_long: np.ndarray) -> List[ModeInfo]:
    """
    Classify longitudinal modes from 4x4 reduced matrix.
    
    Modes:
        - Short-period: Higher frequency, well-damped (α, q)
        - Phugoid: Lower frequency, lightly-damped (u, θ)
    
    References:
        Roskam Ch5, Section 5.2.2
    """
    eigs = np.linalg.eigvals(A_long)
    
    modes = []
    
    # Sort by imaginary part magnitude (frequency)
    sorted_eigs = sorted(eigs, key=lambda x: -abs(x.imag))
    
    # Group into pairs (complex conjugates)
    processed = set()
    pairs = []
    
    for i, e in enumerate(sorted_eigs):
        if i in processed:
            continue
        # Find conjugate
        for j, f in enumerate(sorted_eigs):
            if j > i and j not in processed:
                if abs(e.real - f.real) < 1e-6 and abs(e.imag + f.imag) < 1e-6:
                    pairs.append(e)
                    processed.add(i)
                    processed.add(j)
                    break
        if i not in processed:
            pairs.append(e)  # Real eigenvalue
            processed.add(i)
    
    # First pair is short-period (higher frequency)
    if len(pairs) >= 1:
        info = analyze_eigenvalue(pairs[0])
        modes.append(ModeInfo(
            name='Short-Period',
            eigenvalue=pairs[0],
            **{k: v for k, v in info.items() if k != 'eigenvalue'}
        ))
    
    # Second pair is phugoid (lower frequency)
    if len(pairs) >= 2:
        info = analyze_eigenvalue(pairs[1])
        modes.append(ModeInfo(
            name='Phugoid',
            eigenvalue=pairs[1],
            **{k: v for k, v in info.items() if k != 'eigenvalue'}
        ))
    
    return modes


def classify_lateral_modes(A_lat: np.ndarray) -> List[ModeInfo]:
    """
    Classify lateral-directional modes from 4x4 reduced matrix.
    
    Modes:
        - Dutch Roll: Oscillatory (β, r coupled)
        - Roll Subsidence: Fast real mode (p)
        - Spiral: Slow real mode (φ, ψ)
    
    References:
        Roskam Ch5, Section 5.3.2
    """
    eigs = np.linalg.eigvals(A_lat)
    
    modes = []
    
    # Separate oscillatory and real modes
    oscillatory = [e for e in eigs if abs(e.imag) > 1e-6]
    real_modes = [e for e in eigs if abs(e.imag) <= 1e-6]
    
    # Dutch roll is the oscillatory mode
    if len(oscillatory) >= 2:
        # Take one from the pair
        dr_eig = oscillatory[0]
        info = analyze_eigenvalue(dr_eig)
        modes.append(ModeInfo(
            name='Dutch Roll',
            eigenvalue=dr_eig,
            **{k: v for k, v in info.items() if k != 'eigenvalue'}
        ))
    
    # Sort real modes by magnitude (time constant)
    real_modes_sorted = sorted(real_modes, key=lambda x: abs(x.real), reverse=True)
    
    # Roll subsidence: faster (larger magnitude real part)
    if len(real_modes_sorted) >= 1:
        roll_eig = real_modes_sorted[0]
        info = analyze_eigenvalue(roll_eig)
        modes.append(ModeInfo(
            name='Roll',
            eigenvalue=roll_eig,
            **{k: v for k, v in info.items() if k != 'eigenvalue'}
        ))
    
    # Spiral: slower (smaller magnitude real part)
    if len(real_modes_sorted) >= 2:
        spiral_eig = real_modes_sorted[1]
        info = analyze_eigenvalue(spiral_eig)
        modes.append(ModeInfo(
            name='Spiral',
            eigenvalue=spiral_eig,
            **{k: v for k, v in info.items() if k != 'eigenvalue'}
        ))
    
    return modes


def analyze_modes(A: np.ndarray) -> List[ModeInfo]:
    """
    Analyze all modes from full 12x12 state matrix.
    
    Extracts longitudinal and lateral submatrices and classifies
    all five primary modes.
    
    Args:
        A: 12x12 state matrix
        
    Returns:
        List of ModeInfo for all identified modes
        
    References:
        Roskam Ch5, Section 5.2 (longitudinal), 5.3 (lateral)
    """
    reduced = extract_reduced_matrices(A, np.zeros((12, 4)))
    
    long_modes = classify_longitudinal_modes(reduced['A_long'])
    lat_modes = classify_lateral_modes(reduced['A_lat'])
    
    return long_modes + lat_modes


def print_mode_table(modes: List[ModeInfo], title: str = "Mode Analysis") -> str:
    """Format mode analysis as a table string."""
    lines = [
        "=" * 70,
        title,
        "=" * 70,
        f"{'Mode':<15} {'Lambda':<25} {'wn (rad/s)':<12} {'zeta':<8} {'T/tau (s)':<10}",
        "-" * 70
    ]
    
    for m in modes:
        if m.is_oscillatory:
            eig_str = f"{m.eigenvalue.real:+.4f} +/- {abs(m.eigenvalue.imag):.4f}j"
            t_str = f"T={m.period:.2f}"
        else:
            eig_str = f"{m.eigenvalue.real:+.4f}"
            t_str = f"tau={m.time_constant:.2f}"
        
        lines.append(f"{m.name:<15} {eig_str:<25} {m.omega_n:<12.3f} {m.zeta:<8.3f} {t_str:<10}")
    
    lines.append("=" * 70)
    return "\n".join(lines)


def sensitivity_analysis(
    model_6dof: SixDoFModel,
    x_trim: np.ndarray,
    u_trim: np.ndarray,
    derivative_name: str,
    scale_factors: List[float],
    verbose: bool = True
) -> List[Dict]:
    """
    Analyze mode sensitivity to a stability derivative.
    
    Varies a derivative by scale factors and recomputes modes.
    
    Args:
        model_6dof: SixDoFModel instance
        x_trim: Trim state
        u_trim: Trim controls
        derivative_name: Name of derivative to vary (e.g., 'M_q', 'M_alpha')
        scale_factors: List of scale factors (e.g., [0.8, 1.0, 1.2])
        verbose: Print progress
        
    Returns:
        List of dicts with scale factor and mode info
        
    References:
        Roskam Ch5, Section 5.4
    """
    results = []
    
    # Get original derivative value
    long_derivs = model_6dof.config.long_derivs
    lat_derivs = model_6dof.config.lat_derivs
    
    # Map derivative names to attributes
    if hasattr(long_derivs, derivative_name):
        original_value = getattr(long_derivs, derivative_name)
        is_long = True
    elif hasattr(lat_derivs, derivative_name):
        original_value = getattr(lat_derivs, derivative_name)
        is_long = False
    else:
        raise ValueError(f"Unknown derivative: {derivative_name}")
    
    for scale in scale_factors:
        # Create modified config
        new_value = original_value * scale
        
        if is_long:
            # Modify longitudinal derivative
            from dataclasses import replace
            modified_long = replace(long_derivs, **{derivative_name: new_value})
            from config import FullAircraftConfig
            modified_config = FullAircraftConfig(
                params=model_6dof.config.params,
                long_derivs=modified_long,
                lat_derivs=lat_derivs,
                V_trim=model_6dof.config.V_trim,
                alpha_trim=model_6dof.config.alpha_trim,
                rho=model_6dof.config.rho
            )
        else:
            from dataclasses import replace
            modified_lat = replace(lat_derivs, **{derivative_name: new_value})
            from config import FullAircraftConfig
            modified_config = FullAircraftConfig(
                params=model_6dof.config.params,
                long_derivs=long_derivs,
                lat_derivs=modified_lat,
                V_trim=model_6dof.config.V_trim,
                alpha_trim=model_6dof.config.alpha_trim,
                rho=model_6dof.config.rho
            )
        
        # Create new model with modified config
        modified_model = SixDoFModel(modified_config)
        
        # Linearize
        A, B = linearize_6dof(modified_model, x_trim, u_trim)
        
        # Analyze modes
        modes = analyze_modes(A)
        
        result = {
            'scale': scale,
            'derivative_value': new_value,
            'modes': modes
        }
        results.append(result)
        
        if verbose:
            print(f"  {derivative_name} × {scale:.1f} = {new_value:.4f}")
    
    return results


def print_sensitivity_table(
    results: List[Dict],
    derivative_name: str,
    mode_names: List[str] = ['Short-Period', 'Phugoid']
) -> str:
    """Format sensitivity analysis as a table."""
    lines = [
        f"\nSensitivity to {derivative_name}",
        "-" * 60,
        f"{'Scale':<10}",
    ]
    
    # Build header
    header = f"{'Scale':<10}"
    for name in mode_names:
        header += f"{'z_'+name[:3]:<10} {'wn_'+name[:3]:<10}"
    lines = [
        f"\nSensitivity to {derivative_name}",
        "-" * 60,
        header,
        "-" * 60
    ]
    
    for result in results:
        line = f"{result['scale']:<10.1f}"
        for name in mode_names:
            mode = next((m for m in result['modes'] if m.name == name), None)
            if mode:
                line += f"{mode.zeta:<10.3f} {mode.omega_n:<10.3f}"
            else:
                line += f"{'N/A':<10} {'N/A':<10}"
        lines.append(line)
    
    lines.append("-" * 60)
    return "\n".join(lines)
