"""
Aircraft Configuration and Aerodynamic Derivatives

Dataclasses for aircraft parameters and dimensional stability derivatives
following Roskam's "Airplane Flight Dynamics and Automatic Flight Controls"
notation (Chapter 5, Table 5.2 for longitudinal, Chapter 4 for lateral).

Assumptions:
- Small perturbations about trim condition
- Stability axes for aerodynamic coefficients
- Flat Earth, constant mass
- α-state formulation for longitudinal (w ≈ U0 α)
- Body axes for 6-DOF

References:
- Roskam Chapter 1: Equations of motion (1.19, 1.25)
- Roskam Chapter 4: Lateral-directional derivatives
- Roskam Chapter 5, Table 5.2: Longitudinal state-space
- Roskam Appendix B: Aircraft data
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class AircraftParameters:
    """
    Physical aircraft parameters for 6-DOF simulation.
    
    Reference geometry and inertias needed for equations of motion.
    All units SI.
    """
    mass: float          # kg
    S: float             # Reference wing area (m²)
    b: float             # Wing span (m)
    c_bar: float         # Mean aerodynamic chord (m)
    Ixx: float           # Moment of inertia about x-axis (kg·m²)
    Iyy: float           # Moment of inertia about y-axis (kg·m²)
    Izz: float           # Moment of inertia about z-axis (kg·m²)
    Ixz: float = 0.0     # Product of inertia (kg·m²), 0 for symmetric


@dataclass
class LongitudinalDerivatives:
    """
    Dimensional longitudinal stability and control derivatives.
    
    Notation follows Roskam Chapter 5, Table 5.2.
    
    Units (SI):
        X_u, Z_u: 1/s
        X_alpha, Z_alpha, X_delta_e, Z_delta_e: m/s²
        Z_q: m/s
        M_u: 1/(m·s)
        M_alpha, M_q, M_delta_e: 1/s²
        
    References:
        Roskam Chapter 5, Table 5.1 (derivative definitions)
        Roskam Appendix B (example aircraft data)
    """
    # Stability derivatives
    X_u: float           # ∂X/∂u (1/s)
    X_alpha: float       # ∂X/∂α (m/s²)
    Z_u: float           # ∂Z/∂u (1/s)
    Z_alpha: float       # ∂Z/∂α (m/s²)
    Z_q: float           # ∂Z/∂q (m/s)
    M_u: float           # ∂M/∂u (1/(m·s))
    M_alpha: float       # ∂M/∂α (1/s²)
    M_q: float           # ∂M/∂q (1/s)
    
    # Control derivatives
    X_delta_e: float     # ∂X/∂δe (m/s²)
    Z_delta_e: float     # ∂Z/∂δe (m/s²)
    M_delta_e: float     # ∂M/∂δe (1/s²)
    
    # Reference flight condition
    U0: float            # Trim airspeed (m/s)
    theta0: float        # Trim pitch angle (rad)
    g: float = 9.81      # Gravitational acceleration (m/s²)
    
    # For coefficient computation
    rho: float = 1.225   # Air density (kg/m³)
    S: float = 1.0       # Reference area (m²)
    c_bar: float = 1.0   # Mean chord (m)
    m: float = 1.0       # Mass (kg)
    Iyy: float = 1.0     # Pitch inertia (kg·m²)


@dataclass
class LateralDerivatives:
    """
    Dimensional lateral-directional stability and control derivatives.
    
    Notation follows Roskam Chapter 4, Tables 4.1-4.2.
    
    For 6-DOF:
        Y_β, Y_p, Y_r: Side force derivatives (m/s², m/s, m/s)
        L_β, L_p, L_r: Rolling moment derivatives (1/s², 1/s, 1/s)
        N_β, N_p, N_r: Yawing moment derivatives (1/s², 1/s, 1/s)
        
    Control derivatives for aileron (δa) and rudder (δr).
    
    References:
        Roskam Chapter 4, Section 4.2
        Roskam Appendix B
    """
    # Sideslip derivatives (β)
    Y_beta: float        # ∂Y/∂β (m/s²)
    L_beta: float        # ∂L/∂β (1/s²) - dihedral effect
    N_beta: float        # ∂N/∂β (1/s²) - weathercock stability
    
    # Roll rate derivatives (p)
    Y_p: float           # ∂Y/∂p (m/s)
    L_p: float           # ∂L/∂p (1/s) - roll damping
    N_p: float           # ∂N/∂p (1/s)
    
    # Yaw rate derivatives (r)
    Y_r: float           # ∂Y/∂r (m/s)
    L_r: float           # ∂L/∂r (1/s)
    N_r: float           # ∂N/∂r (1/s) - yaw damping
    
    # Aileron control derivatives (δa)
    Y_delta_a: float     # ∂Y/∂δa (m/s²)
    L_delta_a: float     # ∂L/∂δa (1/s²) - aileron effectiveness
    N_delta_a: float     # ∂N/∂δa (1/s²) - adverse yaw
    
    # Rudder control derivatives (δr)
    Y_delta_r: float     # ∂Y/∂δr (m/s²)
    L_delta_r: float     # ∂L/∂δr (1/s²)
    N_delta_r: float     # ∂N/∂δr (1/s²) - rudder effectiveness


@dataclass
class FullAircraftConfig:
    """
    Complete aircraft configuration for 6-DOF simulation.
    
    Combines:
    - Physical parameters (mass, inertia, geometry)
    - Longitudinal derivatives
    - Lateral-directional derivatives
    - Trim condition
    
    References:
        Roskam Chapters 1, 4, 5
    """
    params: AircraftParameters
    long_derivs: LongitudinalDerivatives
    lat_derivs: LateralDerivatives
    
    # Trim flight condition
    V_trim: float        # Trim airspeed (m/s)
    alpha_trim: float    # Trim angle of attack (rad)
    rho: float           # Air density (kg/m³)
    
    @property
    def q_bar(self) -> float:
        """Dynamic pressure at trim (Pa)"""
        return 0.5 * self.rho * self.V_trim**2


# =============================================================================
# Example Aircraft: Cessna 172-like (generic light aircraft)
# =============================================================================
CESSNA172_DERIVS = LongitudinalDerivatives(
    X_u=-0.05,
    X_alpha=0.4,
    Z_u=-1.2,
    Z_alpha=-4.5,
    Z_q=-8.2,
    M_u=0.002,
    M_alpha=-1.2,
    M_q=-1.8,
    X_delta_e=-0.1,
    Z_delta_e=-3.8,
    M_delta_e=-2.5,
    U0=50.0,
    theta0=0.05
)


# =============================================================================
# Roskam Business Jet (Appendix B, Table B.2 / Chapter 5 Table 5.4)
# =============================================================================
# Original data in imperial units, converted to SI:
#   U0 = 400 kts = 205.8 m/s
#   Derivatives converted: ft/s² → m/s², ft/s → m/s
#
# From Roskam Appendix B (Business Jet, Cruise):
#   Xu = -0.0215 1/s
#   Xα = 9.45 ft/s² → 2.88 m/s²
#   Zu = -0.227 1/s
#   Zα = -445.7 ft/s² → -135.8 m/s²
#   Zq = 0 (negligible)
#   Mu = 0.0
#   Mα = -2.34 1/s²
#   Mq = -0.38 1/s
#   Zδe = -42.2 ft/s² → -12.86 m/s²
#   Mδe = -1.73 1/s²
#
# References:
#   Roskam, "Airplane Flight Dynamics", Appendix B
# =============================================================================

ROSKAM_BUSINESS_JET = LongitudinalDerivatives(
    X_u=-0.0215,
    X_alpha=2.88,
    Z_u=-0.227,
    Z_alpha=-135.8,
    Z_q=-5.0,
    M_u=0.0,
    M_alpha=-2.34,
    M_q=-0.38,
    X_delta_e=0.0,
    Z_delta_e=-12.86,
    M_delta_e=-1.73,
    U0=205.8,
    theta0=0.0,
    rho=0.3629,
    S=21.55,
    c_bar=1.98,
    m=7257.0,
    Iyy=25488.0
)


# =============================================================================
# Business Jet Lateral-Directional Derivatives (Roskam-style jet)
# Typical values for a business jet at cruise
# =============================================================================

BUSINESS_JET_LATERAL = LateralDerivatives(
    # Sideslip derivatives
    Y_beta=-30.5,         # m/s² (side force due to sideslip)
    L_beta=-4.5,          # 1/s² (dihedral effect, negative = stable)
    N_beta=2.5,           # 1/s² (weathercock stability, positive = stable)
    
    # Roll rate derivatives
    Y_p=-0.8,             # m/s
    L_p=-0.55,            # 1/s (roll damping, always negative)
    N_p=0.05,             # 1/s (yaw due to roll rate)
    
    # Yaw rate derivatives
    Y_r=2.5,              # m/s
    L_r=0.10,             # 1/s (roll due to yaw)
    N_r=-0.25,            # 1/s (yaw damping, always negative)
    
    # Aileron derivatives
    Y_delta_a=0.0,        # m/s² (negligible)
    L_delta_a=-0.8,       # 1/s² (roll effectiveness, negative = right roll)
    N_delta_a=0.10,       # 1/s² (adverse yaw)
    
    # Rudder derivatives
    Y_delta_r=-8.0,       # m/s² (side force from rudder)
    L_delta_r=0.10,       # 1/s² (roll from rudder)
    N_delta_r=-0.45       # 1/s² (yaw from rudder, negative = nose left)
)


# =============================================================================
# Business Jet Physical Parameters
# =============================================================================

BUSINESS_JET_PARAMS = AircraftParameters(
    mass=7257.0,          # kg (16,000 lb)
    S=21.55,              # m² (232 ft²)
    b=13.5,               # m (44.3 ft) wingspan
    c_bar=1.98,           # m (6.5 ft) MAC
    Ixx=12000.0,          # kg·m²
    Iyy=25488.0,          # kg·m² (18,800 slug·ft²)
    Izz=35000.0,          # kg·m²
    Ixz=0.0               # Symmetric aircraft
)


# =============================================================================
# Complete 6-DOF Configuration for Business Jet
# =============================================================================

BUSINESS_JET_6DOF = FullAircraftConfig(
    params=BUSINESS_JET_PARAMS,
    long_derivs=ROSKAM_BUSINESS_JET,
    lat_derivs=BUSINESS_JET_LATERAL,
    V_trim=205.8,         # m/s (400 kts)
    alpha_trim=0.05,      # rad (~3°)
    rho=0.3629            # kg/m³ at 35,000 ft
)


# =============================================================================
# Roskam Learjet 24 (Alternative business jet)
# =============================================================================

ROSKAM_LEARJET = LongitudinalDerivatives(
    X_u=-0.0453,
    X_alpha=5.23,
    Z_u=-0.369,
    Z_alpha=-211.0,
    Z_q=-6.48,
    M_u=0.0,
    M_alpha=-6.92,
    M_q=-1.27,
    X_delta_e=0.0,
    Z_delta_e=-18.5,
    M_delta_e=-9.54,
    U0=154.0,
    theta0=0.0,
    rho=0.4135,
    S=21.53,
    c_bar=2.13,
    m=5216.0,
    Iyy=15830.0
)
