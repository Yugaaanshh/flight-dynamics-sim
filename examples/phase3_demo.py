"""
Phase 3 Demo: General Trim + Linearization + Sensitivity Analysis

Demonstrates:
1. Three trim conditions: Level cruise, 30° coordinated turn, 2g pull-up
2. Numerical linearization of 6-DOF model at each trim
3. Mode analysis: Short-period, Phugoid, Dutch Roll, Roll, Spiral
4. Derivative sensitivity: Effect of M_q on longitudinal modes

Uses Roskam Business Jet data.

References:
    Roskam, "Airplane Flight Dynamics", Chapters 4-5
    - Ch4: Steady flight and maneuvering trim
    - Ch5: Modal analysis and derivative sensitivity

Run: python flight_dynamics_sim/examples/phase3_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BUSINESS_JET_6DOF, ROSKAM_BUSINESS_JET
from eom.six_dof import SixDoFModel
from sim.trim_maneuver import (
    trim_level_flight,
    trim_coordinated_turn,
    trim_pullup,
    print_trim_summary
)
from analysis.linearize import (
    linearize_6dof,
    analyze_modes,
    print_mode_table,
    sensitivity_analysis,
    print_sensitivity_table
)


def main():
    print("=" * 70)
    print("Phase 3 Demo: General Trim + Linearization + Mode Sensitivity")
    print("Aircraft: Roskam Business Jet @ 400 kts / 35,000 ft")
    print("Reference: Roskam Chapters 4-5")
    print("=" * 70)
    
    # =========================================================================
    # 1. Create model
    # =========================================================================
    config = BUSINESS_JET_6DOF
    model = SixDoFModel(config)
    
    V_cruise = 205.8   # m/s (400 kts)
    h_cruise = 10668.0 # m (35,000 ft)
    
    # =========================================================================
    # 2. Compute three trim conditions
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: TRIM SOLUTIONS")
    print("=" * 70)
    
    # Level cruise
    print("\n[1] Level Cruise (nz=1, phi=0, gamma=0)")
    trim_level = trim_level_flight(model, V_cruise, h_cruise, verbose=True)
    print(print_trim_summary(trim_level))
    
    # 30° coordinated turn
    print("\n[2] Coordinated Turn (phi=30deg, nz=1.15)")
    trim_turn = trim_coordinated_turn(model, V_cruise, h_cruise, bank_angle_deg=30.0, verbose=True)
    print(print_trim_summary(trim_turn))
    
    # 2g pull-up
    print("\n[3] Pull-Up (nz=2.0)")
    trim_pullup_result = trim_pullup(model, V_cruise, h_cruise, nz_target=2.0, verbose=True)
    print(print_trim_summary(trim_pullup_result))
    
    # =========================================================================
    # 3. Linearize at each trim
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: LINEARIZATION & MODE ANALYSIS")
    print("=" * 70)
    
    trims = [
        ("Level Cruise", trim_level),
        ("30deg Turn", trim_turn),
        ("2g Pull-Up", trim_pullup_result)
    ]
    
    all_modes = {}
    
    for name, trim in trims:
        print(f"\n[{name}]")
        
        # Linearize
        A, B = linearize_6dof(model, trim.x_trim, trim.u_trim)
        
        # Analyze modes
        modes = analyze_modes(A)
        all_modes[name] = modes
        
        # Print mode table
        print(print_mode_table(modes, title=f"Modes at {name}"))
    
    # =========================================================================
    # 4. Compare modes across trim conditions
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: MODE COMPARISON ACROSS TRIMS")
    print("=" * 70)
    
    print(f"\n{'Mode':<15} {'Level zeta':<12} {'Turn zeta':<12} {'Pullup zeta':<12}")
    print("-" * 55)
    
    mode_names = ['Short-Period', 'Phugoid', 'Dutch Roll', 'Roll', 'Spiral']
    
    for mode_name in mode_names:
        line = f"{mode_name:<15}"
        for trim_name in ["Level Cruise", "30deg Turn", "2g Pull-Up"]:
            modes = all_modes[trim_name]
            mode = next((m for m in modes if m.name == mode_name), None)
            if mode:
                line += f"{mode.zeta:<12.3f}"
            else:
                line += f"{'N/A':<12}"
        print(line)
    
    print("\nObservations:")
    print("  - Short-period damping may change with load factor")
    print("  - Dutch roll and spiral modes affected by bank angle")
    
    # =========================================================================
    # 5. Derivative sensitivity analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: DERIVATIVE SENSITIVITY (Roskam Ch5.4)")
    print("=" * 70)
    
    print("\nVarying M_q (pitch damping) ±20% about level trim:")
    
    scale_factors = [0.8, 0.9, 1.0, 1.1, 1.2]
    
    sensitivity_results = sensitivity_analysis(
        model_6dof=model,
        x_trim=trim_level.x_trim,
        u_trim=trim_level.u_trim,
        derivative_name='M_q',
        scale_factors=scale_factors,
        verbose=True
    )
    
    print(print_sensitivity_table(sensitivity_results, 'M_q', ['Short-Period', 'Phugoid']))
    
    # Also test M_alpha sensitivity
    print("\nVarying M_alpha (static stability) +/-20%:")
    
    sensitivity_m_alpha = sensitivity_analysis(
        model_6dof=model,
        x_trim=trim_level.x_trim,
        u_trim=trim_level.u_trim,
        derivative_name='M_alpha',
        scale_factors=scale_factors,
        verbose=True
    )
    
    print(print_sensitivity_table(sensitivity_m_alpha, 'M_alpha', ['Short-Period', 'Phugoid']))
    
    # =========================================================================
    # 6. Create sensitivity plot
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Derivative Sensitivity Analysis (Roskam Ch5.4)", fontsize=13, fontweight='bold')
    
    # M_q sensitivity plot
    ax = axes[0]
    scales = [r['scale'] for r in sensitivity_results]
    
    sp_zeta = []
    sp_wn = []
    for r in sensitivity_results:
        sp_mode = next((m for m in r['modes'] if m.name == 'Short-Period'), None)
        sp_zeta.append(sp_mode.zeta if sp_mode else 0)
        sp_wn.append(sp_mode.omega_n if sp_mode else 0)
    
    ax.plot(scales, sp_zeta, 'b-o', linewidth=2, markersize=8, label='zeta (damping)')
    ax.set_xlabel('M_q Scale Factor')
    ax.set_ylabel('Short-Period zeta', color='b')
    ax.tick_params(axis='y', labelcolor='b')
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_title('Short-Period Damping vs M_q')
    ax.legend(loc='upper left')
    
    ax2 = ax.twinx()
    ax2.plot(scales, sp_wn, 'r--s', linewidth=2, markersize=8, label='wn (freq)')
    ax2.set_ylabel('Short-Period wn (rad/s)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.legend(loc='upper right')
    
    # M_alpha sensitivity plot
    ax = axes[1]
    
    sp_zeta_ma = []
    sp_wn_ma = []
    for r in sensitivity_m_alpha:
        sp_mode = next((m for m in r['modes'] if m.name == 'Short-Period'), None)
        sp_zeta_ma.append(sp_mode.zeta if sp_mode else 0)
        sp_wn_ma.append(sp_mode.omega_n if sp_mode else 0)
    
    ax.plot(scales, sp_zeta_ma, 'b-o', linewidth=2, markersize=8, label='zeta (damping)')
    ax.set_xlabel('M_alpha Scale Factor')
    ax.set_ylabel('Short-Period zeta', color='b')
    ax.tick_params(axis='y', labelcolor='b')
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_title('Short-Period Damping vs M_alpha')
    ax.legend(loc='upper left')
    
    ax2 = ax.twinx()
    ax2.plot(scales, sp_wn_ma, 'r--s', linewidth=2, markersize=8, label='wn (freq)')
    ax2.set_ylabel('Short-Period wn (rad/s)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    output_path = os.path.join(os.path.dirname(__file__), 'phase3_sensitivity.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSensitivity plot saved: {output_path}")
    
    # =========================================================================
    # 7. Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Phase 3 Implementation Complete:

1. GENERAL TRIM SOLVER:
   - Level cruise: alpha, de for L=W, M=0
   - Coordinated turn: accounts for nz = 1/cos(phi)
   - Pull-up: higher nz requires more alpha, de

2. NUMERICAL LINEARIZATION:
   - A, B matrices via central finite differences
   - Works for any trim condition

3. MODE ANALYSIS (Roskam Ch5):
   - Longitudinal: Short-period, Phugoid
   - Lateral: Dutch Roll, Roll, Spiral
   - Eigenvalue -> wn, zeta, T classification

4. SENSITIVITY ANALYSIS (Roskam Ch5.4):
   - M_q affects short-period damping (higher |M_q| -> more damping)
   - M_alpha affects short-period frequency (higher |M_alpha| -> higher wn)
   - Quantifies effect of stability derivatives on flying qualities
""")
    
    return all_modes, sensitivity_results


if __name__ == "__main__":
    results = main()
