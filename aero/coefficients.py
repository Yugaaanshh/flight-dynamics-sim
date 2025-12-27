"""
Aerodynamic Coefficient Models

Computes non-dimensional aerodynamic coefficients from state and controls
using linear stability derivative models following Roskam Chapter 3.

Longitudinal (Roskam Ch3 Eqns 3.21, 3.46):
    C_L = C_L0 + C_Lα·α + C_Lq·(q·c̄/2V) + C_Lδe·δe
    C_m = C_m0 + C_mα·α + C_mq·(q·c̄/2V) + C_mδe·δe

Lateral-Directional (Roskam Ch4):
    C_Y = C_Yβ·β + C_Yp·(p·b/2V) + C_Yr·(r·b/2V) + C_Yδa·δa + C_Yδr·δr
    C_l = C_lβ·β + C_lp·(p·b/2V) + C_lr·(r·b/2V) + C_lδa·δa + C_lδr·δr
    C_n = C_nβ·β + C_np·(p·b/2V) + C_nr·(r·b/2V) + C_nδa·δa + C_nδr·δr

Body-axis force coefficient:
    C_A = C_D·cos(α) - C_L·sin(α)  (axial, +aft)
    C_N = C_D·sin(α) + C_L·cos(α)  (normal, +down)

Assumptions:
- Linear model valid for small perturbations
- Stability axes for CL, CD; body axes for CA, CN
- Quasi-steady aerodynamics

References:
    Roskam, "Airplane Flight Dynamics", Chapters 3-4
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict


@dataclass
class AeroCoefficients:
    """
    Non-dimensional aerodynamic coefficient derivatives.
    
    All derivatives are per-radian.
    Notation follows Roskam Chapters 3-4.
    """
    # Lift coefficient derivatives (stability axis)
    CL_0: float = 0.0        # Lift at zero α
    CL_alpha: float = 0.0    # ∂CL/∂α (1/rad)
    CL_q: float = 0.0        # ∂CL/∂q̂ where q̂ = qc̄/2V
    CL_delta_e: float = 0.0  # ∂CL/∂δe (1/rad)
    
    # Drag coefficient derivatives
    CD_0: float = 0.0        # Zero-lift drag
    K: float = 0.0           # Induced drag factor (CD = CD0 + K·CL²)
    CD_delta_e: float = 0.0  # ∂CD/∂δe
    
    # Pitching moment coefficient derivatives
    Cm_0: float = 0.0        # Moment at zero α
    Cm_alpha: float = 0.0    # ∂Cm/∂α (negative for stability)
    Cm_q: float = 0.0        # ∂Cm/∂q̂ (pitch damping)
    Cm_delta_e: float = 0.0  # ∂Cm/∂δe (elevator effectiveness)
    
    # Side force coefficient derivatives
    CY_beta: float = 0.0     # ∂CY/∂β
    CY_p: float = 0.0        # ∂CY/∂p̂ where p̂ = pb/2V
    CY_r: float = 0.0        # ∂CY/∂r̂ where r̂ = rb/2V
    CY_delta_a: float = 0.0  # ∂CY/∂δa
    CY_delta_r: float = 0.0  # ∂CY/∂δr
    
    # Rolling moment coefficient derivatives
    Cl_beta: float = 0.0     # ∂Cl/∂β (dihedral effect)
    Cl_p: float = 0.0        # ∂Cl/∂p̂ (roll damping)
    Cl_r: float = 0.0        # ∂Cl/∂r̂
    Cl_delta_a: float = 0.0  # ∂Cl/∂δa (aileron effectiveness)
    Cl_delta_r: float = 0.0  # ∂Cl/∂δr
    
    # Yawing moment coefficient derivatives
    Cn_beta: float = 0.0     # ∂Cn/∂β (weathercock stability)
    Cn_p: float = 0.0        # ∂Cn/∂p̂
    Cn_r: float = 0.0        # ∂Cn/∂r̂ (yaw damping)
    Cn_delta_a: float = 0.0  # ∂Cn/∂δa (adverse yaw)
    Cn_delta_r: float = 0.0  # ∂Cn/∂δr (rudder effectiveness)


def compute_longitudinal_coefficients(
    alpha: float,
    q: float,
    delta_e: float,
    V: float,
    c_bar: float,
    coeffs: AeroCoefficients
) -> Tuple[float, float, float]:
    """
    Compute longitudinal aerodynamic coefficients (stability axes).
    
    Implements Roskam Chapter 3, Eqns 3.21 and 3.46.
    
    Args:
        alpha: Angle of attack (rad)
        q: Pitch rate (rad/s)
        delta_e: Elevator deflection (rad), positive TED
        V: True airspeed (m/s)
        c_bar: Mean aerodynamic chord (m)
        coeffs: AeroCoefficients dataclass
        
    Returns:
        (CL, CD, Cm): Lift, drag, pitching moment coefficients
        
    References:
        Roskam Ch3, Eqn 3.21: CL buildup
        Roskam Ch3, Eqn 3.46: Cm buildup
    """
    q_hat = q * c_bar / (2.0 * V) if V > 1e-6 else 0.0
    
    CL = (coeffs.CL_0 + 
          coeffs.CL_alpha * alpha + 
          coeffs.CL_q * q_hat + 
          coeffs.CL_delta_e * delta_e)
    
    CD = coeffs.CD_0 + coeffs.K * CL**2 + coeffs.CD_delta_e * abs(delta_e)
    
    Cm = (coeffs.Cm_0 + 
          coeffs.Cm_alpha * alpha + 
          coeffs.Cm_q * q_hat + 
          coeffs.Cm_delta_e * delta_e)
    
    return CL, CD, Cm


def compute_lateral_coefficients(
    beta: float,
    p: float,
    r: float,
    delta_a: float,
    delta_r: float,
    V: float,
    b: float,
    coeffs: AeroCoefficients
) -> Tuple[float, float, float]:
    """
    Compute lateral-directional aerodynamic coefficients.
    
    Implements Roskam Chapter 4 linear buildup.
    
    Args:
        beta: Sideslip angle (rad)
        p: Roll rate (rad/s)
        r: Yaw rate (rad/s)
        delta_a: Aileron deflection (rad)
        delta_r: Rudder deflection (rad)
        V: True airspeed (m/s)
        b: Wing span (m)
        coeffs: AeroCoefficients dataclass
        
    Returns:
        (CY, Cl, Cn): Side force, rolling moment, yawing moment coefficients
        
    References:
        Roskam Ch4, Section 4.2
    """
    p_hat = p * b / (2.0 * V) if V > 1e-6 else 0.0
    r_hat = r * b / (2.0 * V) if V > 1e-6 else 0.0
    
    CY = (coeffs.CY_beta * beta +
          coeffs.CY_p * p_hat +
          coeffs.CY_r * r_hat +
          coeffs.CY_delta_a * delta_a +
          coeffs.CY_delta_r * delta_r)
    
    Cl = (coeffs.Cl_beta * beta +
          coeffs.Cl_p * p_hat +
          coeffs.Cl_r * r_hat +
          coeffs.Cl_delta_a * delta_a +
          coeffs.Cl_delta_r * delta_r)
    
    Cn = (coeffs.Cn_beta * beta +
          coeffs.Cn_p * p_hat +
          coeffs.Cn_r * r_hat +
          coeffs.Cn_delta_a * delta_a +
          coeffs.Cn_delta_r * delta_r)
    
    return CY, Cl, Cn


def compute_all_coefficients(
    alpha: float,
    beta: float,
    p: float,
    q: float,
    r: float,
    delta_e: float,
    delta_a: float,
    delta_r: float,
    V: float,
    c_bar: float,
    b: float,
    coeffs: AeroCoefficients
) -> Dict[str, float]:
    """
    Compute all six aerodynamic coefficients.
    
    Combines longitudinal and lateral-directional models.
    Also computes body-axis force coefficients CA, CN from CL, CD.
    
    Args:
        alpha, beta: Angles (rad)
        p, q, r: Angular rates (rad/s)
        delta_e, delta_a, delta_r: Control deflections (rad)
        V: Airspeed (m/s)
        c_bar: Mean chord (m)
        b: Wing span (m)
        coeffs: AeroCoefficients dataclass
        
    Returns:
        Dict with CL, CD, Cm, CY, Cl, Cn, CA, CN
        
    References:
        Roskam Ch3-4
    """
    CL, CD, Cm = compute_longitudinal_coefficients(
        alpha, q, delta_e, V, c_bar, coeffs
    )
    
    CY, Cl, Cn = compute_lateral_coefficients(
        beta, p, r, delta_a, delta_r, V, b, coeffs
    )
    
    # Convert to body axes (Roskam Ch1)
    # CA = axial force coefficient (positive aft)
    # CN = normal force coefficient (positive down)
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    
    CA = CD * cos_alpha - CL * sin_alpha
    CN = CD * sin_alpha + CL * cos_alpha
    
    return {
        'CL': CL, 'CD': CD, 'Cm': Cm,
        'CY': CY, 'Cl': Cl, 'Cn': Cn,
        'CA': CA, 'CN': CN
    }


def dimensional_to_nondimensional_longitudinal(
    Z_alpha: float,
    M_alpha: float,
    Z_delta_e: float,
    M_delta_e: float,
    Z_q: float,
    M_q: float,
    m: float,
    S: float,
    c_bar: float,
    Iyy: float,
    rho: float,
    V: float,
    CL_0: float = 0.3,
    Cm_0: float = 0.0
) -> AeroCoefficients:
    """
    Convert dimensional derivatives to non-dimensional coefficients.
    
    Uses Roskam Chapter 5, Table 5.1 relationships:
        Z_α = -q̄ S C_Lα / m  →  C_Lα = -m Z_α / (q̄ S)
        M_α = q̄ S c̄ C_mα / I_yy  →  C_mα = I_yy M_α / (q̄ S c̄)
    
    References:
        Roskam Chapter 5, Table 5.1
    """
    q_bar = 0.5 * rho * V**2
    
    CL_alpha = -m * Z_alpha / (q_bar * S) if q_bar * S > 1e-6 else 5.0
    CL_q = -m * Z_q / (q_bar * S * c_bar / (2 * V)) if q_bar * S > 1e-6 else 4.0
    CL_delta_e = -m * Z_delta_e / (q_bar * S) if q_bar * S > 1e-6 else 0.3
    
    Cm_alpha = Iyy * M_alpha / (q_bar * S * c_bar) if q_bar * S * c_bar > 1e-6 else -0.5
    Cm_q = Iyy * M_q / (q_bar * S * c_bar**2 / (2 * V)) if q_bar * S > 1e-6 else -10.0
    Cm_delta_e = Iyy * M_delta_e / (q_bar * S * c_bar) if q_bar * S * c_bar > 1e-6 else -1.0
    
    return AeroCoefficients(
        CL_0=CL_0,
        CL_alpha=CL_alpha,
        CL_q=CL_q,
        CL_delta_e=CL_delta_e,
        CD_0=0.02,
        K=0.04,
        CD_delta_e=0.0,
        Cm_0=Cm_0,
        Cm_alpha=Cm_alpha,
        Cm_q=Cm_q,
        Cm_delta_e=Cm_delta_e
    )


def dimensional_to_nondimensional_lateral(
    lat_derivs,
    m: float,
    S: float,
    b: float,
    Ixx: float,
    Izz: float,
    rho: float,
    V: float
) -> AeroCoefficients:
    """
    Convert lateral dimensional derivatives to non-dimensional coefficients.
    
    Uses Roskam Chapter 4 relationships:
        Y_β = q̄ S C_Yβ / m  →  C_Yβ = m Y_β / (q̄ S)
        L_β = q̄ S b C_lβ / I_xx  →  C_lβ = I_xx L_β / (q̄ S b)
        N_β = q̄ S b C_nβ / I_zz  →  C_nβ = I_zz N_β / (q̄ S b)
    
    Args:
        lat_derivs: LateralDerivatives dataclass
        m, S, b, Ixx, Izz, rho, V: Aircraft and flight parameters
        
    Returns:
        AeroCoefficients with lateral derivatives filled in
    """
    q_bar = 0.5 * rho * V**2
    
    coeffs = AeroCoefficients()
    
    # Side force derivatives
    coeffs.CY_beta = m * lat_derivs.Y_beta / (q_bar * S) if q_bar * S > 1e-6 else -0.5
    coeffs.CY_p = m * lat_derivs.Y_p / (q_bar * S * b / (2 * V)) if q_bar * S > 1e-6 else 0.0
    coeffs.CY_r = m * lat_derivs.Y_r / (q_bar * S * b / (2 * V)) if q_bar * S > 1e-6 else 0.3
    coeffs.CY_delta_a = m * lat_derivs.Y_delta_a / (q_bar * S) if q_bar * S > 1e-6 else 0.0
    coeffs.CY_delta_r = m * lat_derivs.Y_delta_r / (q_bar * S) if q_bar * S > 1e-6 else 0.15
    
    # Rolling moment derivatives
    coeffs.Cl_beta = Ixx * lat_derivs.L_beta / (q_bar * S * b) if q_bar * S * b > 1e-6 else -0.1
    coeffs.Cl_p = Ixx * lat_derivs.L_p / (q_bar * S * b**2 / (2 * V)) if q_bar * S > 1e-6 else -0.4
    coeffs.Cl_r = Ixx * lat_derivs.L_r / (q_bar * S * b**2 / (2 * V)) if q_bar * S > 1e-6 else 0.1
    coeffs.Cl_delta_a = Ixx * lat_derivs.L_delta_a / (q_bar * S * b) if q_bar * S * b > 1e-6 else -0.15
    coeffs.Cl_delta_r = Ixx * lat_derivs.L_delta_r / (q_bar * S * b) if q_bar * S * b > 1e-6 else 0.02
    
    # Yawing moment derivatives
    coeffs.Cn_beta = Izz * lat_derivs.N_beta / (q_bar * S * b) if q_bar * S * b > 1e-6 else 0.1
    coeffs.Cn_p = Izz * lat_derivs.N_p / (q_bar * S * b**2 / (2 * V)) if q_bar * S > 1e-6 else -0.03
    coeffs.Cn_r = Izz * lat_derivs.N_r / (q_bar * S * b**2 / (2 * V)) if q_bar * S > 1e-6 else -0.15
    coeffs.Cn_delta_a = Izz * lat_derivs.N_delta_a / (q_bar * S * b) if q_bar * S * b > 1e-6 else 0.01
    coeffs.Cn_delta_r = Izz * lat_derivs.N_delta_r / (q_bar * S * b) if q_bar * S * b > 1e-6 else -0.1
    
    return coeffs


def create_full_aero_coefficients(config) -> AeroCoefficients:
    """
    Create complete AeroCoefficients from FullAircraftConfig.
    
    Combines longitudinal and lateral derivatives.
    
    Args:
        config: FullAircraftConfig instance
        
    Returns:
        AeroCoefficients with all derivatives
    """
    p = config.params
    ld = config.long_derivs
    lat = config.lat_derivs
    q_bar = config.q_bar
    V = config.V_trim
    
    # Longitudinal
    long_coeffs = dimensional_to_nondimensional_longitudinal(
        Z_alpha=ld.Z_alpha,
        M_alpha=ld.M_alpha,
        Z_delta_e=ld.Z_delta_e,
        M_delta_e=ld.M_delta_e,
        Z_q=ld.Z_q,
        M_q=ld.M_q,
        m=p.mass,
        S=p.S,
        c_bar=p.c_bar,
        Iyy=p.Iyy,
        rho=config.rho,
        V=V
    )
    
    # Lateral
    lat_coeffs = dimensional_to_nondimensional_lateral(
        lat_derivs=lat,
        m=p.mass,
        S=p.S,
        b=p.b,
        Ixx=p.Ixx,
        Izz=p.Izz,
        rho=config.rho,
        V=V
    )
    
    # Merge into single AeroCoefficients
    return AeroCoefficients(
        CL_0=long_coeffs.CL_0,
        CL_alpha=long_coeffs.CL_alpha,
        CL_q=long_coeffs.CL_q,
        CL_delta_e=long_coeffs.CL_delta_e,
        CD_0=long_coeffs.CD_0,
        K=long_coeffs.K,
        CD_delta_e=long_coeffs.CD_delta_e,
        Cm_0=long_coeffs.Cm_0,
        Cm_alpha=long_coeffs.Cm_alpha,
        Cm_q=long_coeffs.Cm_q,
        Cm_delta_e=long_coeffs.Cm_delta_e,
        CY_beta=lat_coeffs.CY_beta,
        CY_p=lat_coeffs.CY_p,
        CY_r=lat_coeffs.CY_r,
        CY_delta_a=lat_coeffs.CY_delta_a,
        CY_delta_r=lat_coeffs.CY_delta_r,
        Cl_beta=lat_coeffs.Cl_beta,
        Cl_p=lat_coeffs.Cl_p,
        Cl_r=lat_coeffs.Cl_r,
        Cl_delta_a=lat_coeffs.Cl_delta_a,
        Cl_delta_r=lat_coeffs.Cl_delta_r,
        Cn_beta=lat_coeffs.Cn_beta,
        Cn_p=lat_coeffs.Cn_p,
        Cn_r=lat_coeffs.Cn_r,
        Cn_delta_a=lat_coeffs.Cn_delta_a,
        Cn_delta_r=lat_coeffs.Cn_delta_r
    )
