"""
Phase 1.5 Demo: Longitudinal Step Response with Coefficients

Complete runnable example demonstrating the control surface → coefficient → motion chain:
    δe → ΔCm → q → θ

Uses Roskam Business Jet data from Appendix B, Table B.2.

Features:
1. Model instantiation with real aircraft derivatives
2. Elevator step response simulation (-5° nose-up)
3. 3x2 plots: States (Δu, Δα, Δq, Δθ) + Coefficients (CL, Cm) + Elevator input
4. Eigenvalue analysis with mode identification table

References:
    Roskam, "Airplane Flight Dynamics and Automatic Flight Controls"
    - Chapter 3: Aerodynamic coefficient buildup
    - Chapter 5, Table 5.2: State-space formulation
    - Appendix B: Business Jet aircraft data

Run: python flight_dynamics_sim/examples/phase1_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ROSKAM_BUSINESS_JET, CESSNA172_DERIVS
from eom.longitudinal import LongitudinalLinearModel
from sim import simulate_step_response


def main():
    # =========================================================================
    # 1. Create model from Roskam Business Jet data
    # =========================================================================
    print("=" * 70)
    print("Phase 1.5 Demo: Roskam Business Jet - Elevator Step Response")
    print("Reference: Roskam Appendix B (Table B.2), Chapter 5 (Table 5.2)")
    print("=" * 70)
    
    derivs = ROSKAM_BUSINESS_JET
    model = LongitudinalLinearModel(derivs)
    
    print(f"\nAircraft: Roskam Business Jet (Cruise)")
    print(f"  U0 = {derivs.U0:.1f} m/s ({derivs.U0 * 1.944:.0f} kts)")
    print(f"  Altitude: ~35,000 ft (rho = {derivs.rho:.4f} kg/m³)")
    print(f"  Mass = {derivs.m:.0f} kg")
    print(f"  S = {derivs.S:.2f} m², c̄ = {derivs.c_bar:.2f} m")
    
    print(f"\nDimensional Derivatives (SI units):")
    print(f"  X_u = {derivs.X_u:.4f} 1/s")
    print(f"  Z_α = {derivs.Z_alpha:.2f} m/s²")
    print(f"  M_α = {derivs.M_alpha:.3f} 1/s²")
    print(f"  M_q = {derivs.M_q:.3f} 1/s")
    print(f"  M_δe = {derivs.M_delta_e:.3f} 1/s²")
    
    # =========================================================================
    # 2. Print A, B matrices
    # =========================================================================
    print(f"\nState-Space Model: ẋ = Ax + Bu")
    print(f"  State: x = [Δu, Δα, Δq, Δθ]ᵀ")
    print(f"  Input: u = Δδe")
    
    print(f"\nA matrix (Roskam Table 5.2):")
    np.set_printoptions(precision=4, suppress=True, linewidth=100)
    print(model.A)
    
    print(f"\nB matrix:")
    print(model.B.flatten())
    
    # =========================================================================
    # 3. Modal analysis with table
    # =========================================================================
    print("\n" + model.print_modes_table())
    
    # =========================================================================
    # 4. Simulate elevator step response
    # =========================================================================
    print("\n" + "=" * 70)
    print("Simulation: δe = -5° step at t = 1s (trailing edge up, nose-up)")
    print("Demonstrating: δe → ΔCm → q → α/θ response chain")
    print("=" * 70)
    
    t_step = 1.0
    amplitude_rad = np.radians(-5.0)  # -5° (TEU = nose-up)
    
    result = simulate_step_response(
        model,
        x0=np.zeros(4),
        t_span=(0.0, 60.0),  # Longer to see phugoid
        t_step=t_step,
        amplitude=amplitude_rad,
        dt=0.02
    )
    
    print(f"\nSimulation completed:")
    print(f"  Time span: [{result.t[0]:.1f}, {result.t[-1]:.1f}] s")
    print(f"  Time steps: {len(result.t)}")
    print(f"  Elevator step: {np.degrees(amplitude_rad):.1f}°")
    
    # =========================================================================
    # 5. Create 3x2 plot
    # =========================================================================
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(
        "Roskam Business Jet: δe → ΔCm → q → motion\n"
        "(Roskam Appendix B, Chapter 5 Table 5.2)",
        fontsize=14, fontweight='bold'
    )
    
    # Row 1: States (Δu, Δα)
    # ---------------------------
    ax = axes[0, 0]
    ax.plot(result.t, result.u, 'b-', linewidth=1.5)
    ax.axvline(x=t_step, color='r', linestyle='--', alpha=0.5, label='δe step')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Δu (m/s)')
    ax.set_title('Forward Velocity Perturbation (Phugoid)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    ax = axes[0, 1]
    ax.plot(result.t, np.degrees(result.alpha), 'g-', linewidth=1.5)
    ax.axvline(x=t_step, color='r', linestyle='--', alpha=0.5, label='δe step')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Δα (°)')
    ax.set_title('Angle of Attack Perturbation (Short-Period)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Row 2: States (Δq, Δθ)
    # ---------------------------
    ax = axes[1, 0]
    ax.plot(result.t, np.degrees(result.q), 'm-', linewidth=1.5)
    ax.axvline(x=t_step, color='r', linestyle='--', alpha=0.5, label='δe step')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Δq (°/s)')
    ax.set_title('Pitch Rate Perturbation')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    ax = axes[1, 1]
    ax.plot(result.t, np.degrees(result.theta), 'c-', linewidth=1.5)
    ax.axvline(x=t_step, color='r', linestyle='--', alpha=0.5, label='δe step')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Δθ (°)')
    ax.set_title('Pitch Angle Perturbation')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Row 3: Coefficients and Elevator
    # ---------------------------
    ax = axes[2, 0]
    ax.plot(result.t, result.Cm, 'r-', linewidth=1.5, label='Cm')
    ax.plot(result.t, result.CL, 'b--', linewidth=1.5, label='CL')
    ax.axvline(x=t_step, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Coefficient')
    ax.set_title('Aerodynamic Coefficients (Cm, CL)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    ax = axes[2, 1]
    ax.plot(result.t, np.degrees(result.delta_e), 'k-', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('δe (°)')
    ax.set_title('Elevator Input (TEU negative)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([np.degrees(amplitude_rad) - 1, 1])
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(os.path.dirname(__file__), 'phase1_response.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {output_path}")
    
    # plt.show()  # Uncomment for interactive display
    
    # =========================================================================
    # 6. Physical interpretation
    # =========================================================================
    modes = model.analyze_modes()
    sp = modes['short_period']
    ph = modes['phugoid']
    
    print("\n" + "=" * 70)
    print("PHYSICAL INTERPRETATION")
    print("=" * 70)
    print(f"""
Control Surface → Coefficient → Motion Chain:

1. ELEVATOR INPUT (δe = -5°, trailing edge up):
   → Increases tail download → nose-up moment

2. COEFFICIENT RESPONSE:
   → Immediate ΔCm (negative dip) from Cm_δe = {model.aero_coeffs.Cm_delta_e:.3f}
   → Delayed ΔCL changes as α responds

3. SHORT-PERIOD MODE (T = {sp['period']:.1f}s, ζ = {sp['zeta']:.2f}):
   → Fast α and q oscillation, quickly settles
   → Well-damped pitch rate response

4. PHUGOID MODE (T = {ph['period']:.1f}s, ζ = {ph['zeta']:.2f}):
   → Slow u and θ oscillation (energy exchange)
   → {"Well-damped" if ph['zeta'] > 0.1 else "Lightly damped"} - takes {"a few" if ph['zeta'] > 0.1 else "many"} cycles to settle
""")
    
    return result, model


if __name__ == "__main__":
    result, model = main()
