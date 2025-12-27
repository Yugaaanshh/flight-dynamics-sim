"""
Phase 4 Demo: Control Surface Influence Experiments

Demonstrates how control surface effectiveness affects:
1. Elevator power (M_de): Short-period/phugoid modes and pitch response
2. Aileron power (Cl_da): Roll mode time constant and Dutch roll
3. Rudder power (Cn_dr): Dutch roll damping and yaw behavior

Design-style exploration: de -> dC -> mode -> response chain.

References:
    Roskam, "Airplane Flight Dynamics", Chapter 5
    Section 5.4: Effects of stability and control derivatives

Run: python flight_dynamics_sim/examples/phase4_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ROSKAM_BUSINESS_JET, BUSINESS_JET_6DOF
from eom.six_dof import SixDoFModel
from sim.trim_maneuver import trim_level_flight
from experiments.control_influence import (
    run_elevator_power_sweep,
    run_aileron_power_sweep,
    run_rudder_power_sweep,
    print_elevator_sweep_table,
    print_aileron_sweep_table,
    print_rudder_sweep_table,
    compute_response_metrics
)


def main():
    print("=" * 70)
    print("Phase 4 Demo: Control Surface Influence Experiments")
    print("Aircraft: Roskam Business Jet @ 400 kts / 35,000 ft")
    print("Reference: Roskam Chapter 5, Section 5.4")
    print("=" * 70)
    
    # Common scale factors
    scale_factors = (0.6, 0.8, 1.0, 1.2, 1.4)
    
    # =========================================================================
    # A. ELEVATOR POWER INFLUENCE (LONGITUDINAL)
    # =========================================================================
    print("\n" + "=" * 70)
    print("A. ELEVATOR CONTROL POWER INFLUENCE")
    print("   Varying M_de (elevator effectiveness)")
    print("=" * 70)
    
    # Run elevator sweep
    print("\nRunning elevator power sweep...")
    elev_results = run_elevator_power_sweep(
        base_derivs=ROSKAM_BUSINESS_JET,
        scale_factors=scale_factors,
        de_step_deg=-2.0,
        t_final=15.0
    )
    
    # Print table
    print(print_elevator_sweep_table(elev_results))
    
    # Create Figure 1: q(t) response family
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(scale_factors)))
    
    for i, scale in enumerate(sorted(elev_results.keys())):
        r = elev_results[scale]
        ax1.plot(r['time'], np.degrees(r['response']['q']), 
                 color=colors[i], linewidth=2, 
                 label=f"M_de x {scale:.1f}")
    
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax1.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='Step applied')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Pitch Rate q (deg/s)')
    ax1.set_title('Elevator Power Influence: q(t) Response to -2 deg Step\n(Roskam Ch5.4)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 15])
    
    fig1_path = os.path.join(os.path.dirname(__file__), 'phase4_elevator_q.png')
    fig1.savefig(fig1_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {fig1_path}")
    
    # Create Figure 2: theta(t) response family
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    for i, scale in enumerate(sorted(elev_results.keys())):
        r = elev_results[scale]
        ax2.plot(r['time'], np.degrees(r['response']['theta']), 
                 color=colors[i], linewidth=2, 
                 label=f"M_de x {scale:.1f}")
    
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Pitch Angle theta (deg)')
    ax2.set_title('Elevator Power Influence: theta(t) Response')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 15])
    
    fig2_path = os.path.join(os.path.dirname(__file__), 'phase4_elevator_theta.png')
    fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {fig2_path}")
    
    # Create Figure 3: Short-period zeta/wn vs M_de scale
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    scales_list = sorted(elev_results.keys())
    sp_zeta = [elev_results[s]['modes']['short_period']['zeta'] for s in scales_list]
    sp_wn = [elev_results[s]['modes']['short_period']['omega_n'] for s in scales_list]
    
    ax3.plot(scales_list, sp_zeta, 'b-o', linewidth=2, markersize=8, label='zeta (damping)')
    ax3.set_xlabel('M_de Scale Factor')
    ax3.set_ylabel('Short-Period zeta', color='b')
    ax3.tick_params(axis='y', labelcolor='b')
    ax3.set_ylim([0, max(sp_zeta) * 1.2])
    
    ax3b = ax3.twinx()
    ax3b.plot(scales_list, sp_wn, 'r--s', linewidth=2, markersize=8, label='wn (rad/s)')
    ax3b.set_ylabel('Short-Period wn (rad/s)', color='r')
    ax3b.tick_params(axis='y', labelcolor='r')
    
    ax3.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Short-Period Mode vs Elevator Power (M_de)\n(Roskam Ch5.4 Sensitivity)')
    ax3.legend(loc='upper left')
    ax3b.legend(loc='upper right')
    
    fig3_path = os.path.join(os.path.dirname(__file__), 'phase4_elevator_modes.png')
    fig3.savefig(fig3_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {fig3_path}")
    
    # =========================================================================
    # B. AILERON POWER INFLUENCE (LATERAL)
    # =========================================================================
    print("\n" + "=" * 70)
    print("B. AILERON CONTROL POWER INFLUENCE")
    print("   Varying L_da (roll effectiveness)")
    print("=" * 70)
    
    # Get trim for 6-DOF
    config = BUSINESS_JET_6DOF
    model_6dof = SixDoFModel(config)
    trim = trim_level_flight(model_6dof, 205.8, 10668.0, verbose=False)
    
    # Run aileron sweep
    print("\nRunning aileron power sweep...")
    ail_results = run_aileron_power_sweep(
        base_config=config,
        x_trim=trim.x_trim,
        u_trim=trim.u_trim,
        scale_factors=scale_factors
    )
    
    # Print table
    print(print_aileron_sweep_table(ail_results))
    
    # Create Figure 4: Roll mode tau and Dutch roll vs scale
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    
    scales_list = sorted(ail_results.keys())
    tau_roll = [ail_results[s]['roll']['tau'] if ail_results[s]['roll'] else 0 for s in scales_list]
    dr_zeta = [ail_results[s]['dutch_roll']['zeta'] if ail_results[s]['dutch_roll'] else 0 for s in scales_list]
    
    ax4.plot(scales_list, tau_roll, 'g-o', linewidth=2, markersize=8, label='tau_roll (s)')
    ax4.set_xlabel('L_da Scale Factor')
    ax4.set_ylabel('Roll Mode Time Constant (s)', color='g')
    ax4.tick_params(axis='y', labelcolor='g')
    
    ax4b = ax4.twinx()
    ax4b.plot(scales_list, dr_zeta, 'm--s', linewidth=2, markersize=8, label='zeta_DR')
    ax4b.set_ylabel('Dutch Roll zeta', color='m')
    ax4b.tick_params(axis='y', labelcolor='m')
    
    ax4.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Roll Mode & Dutch Roll vs Aileron Power (L_da)\n(Roskam Ch5.4)')
    ax4.legend(loc='upper left')
    ax4b.legend(loc='upper right')
    
    fig4_path = os.path.join(os.path.dirname(__file__), 'phase4_aileron_modes.png')
    fig4.savefig(fig4_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {fig4_path}")
    
    # =========================================================================
    # C. RUDDER POWER INFLUENCE
    # =========================================================================
    print("\n" + "=" * 70)
    print("C. RUDDER CONTROL POWER INFLUENCE")
    print("   Varying N_dr (yaw effectiveness)")
    print("=" * 70)
    
    # Run rudder sweep
    print("\nRunning rudder power sweep...")
    rud_results = run_rudder_power_sweep(
        base_config=config,
        x_trim=trim.x_trim,
        u_trim=trim.u_trim,
        scale_factors=scale_factors
    )
    
    # Print table
    print(print_rudder_sweep_table(rud_results))
    
    # Create Figure 5: Dutch roll vs rudder power
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    
    scales_list = sorted(rud_results.keys())
    dr_zeta_r = [rud_results[s]['dutch_roll']['zeta'] if rud_results[s]['dutch_roll'] else 0 for s in scales_list]
    dr_wn_r = [rud_results[s]['dutch_roll']['omega_n'] if rud_results[s]['dutch_roll'] else 0 for s in scales_list]
    
    ax5.plot(scales_list, dr_zeta_r, 'c-o', linewidth=2, markersize=8, label='zeta_DR')
    ax5.set_xlabel('N_dr Scale Factor')
    ax5.set_ylabel('Dutch Roll zeta', color='c')
    ax5.tick_params(axis='y', labelcolor='c')
    
    ax5b = ax5.twinx()
    ax5b.plot(scales_list, dr_wn_r, 'orange', linestyle='--', marker='s', 
              linewidth=2, markersize=8, label='wn_DR (rad/s)')
    ax5b.set_ylabel('Dutch Roll wn (rad/s)', color='orange')
    ax5b.tick_params(axis='y', labelcolor='orange')
    
    ax5.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax5.grid(True, alpha=0.3)
    ax5.set_title('Dutch Roll Mode vs Rudder Power (N_dr)\n(Roskam Ch5.4)')
    ax5.legend(loc='upper left')
    ax5b.legend(loc='upper right')
    
    fig5_path = os.path.join(os.path.dirname(__file__), 'phase4_rudder_modes.png')
    fig5.savefig(fig5_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {fig5_path}")
    
    # =========================================================================
    # D. SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: CONTROL SURFACE INFLUENCE (Roskam Ch5.4 Style)")
    print("=" * 70)
    print("""
Observations (Business Jet, Roskam-style sensitivity):

ELEVATOR CONTROL POWER (M_de):
- Higher |M_de| -> stronger pitch moment per de deflection
- Short-period: damping (zeta) increases with |M_de|
- Faster initial pitch rate response (larger q peak)
- Flight quality: more responsive but may feel "too quick" if overdone

AILERON CONTROL POWER (L_da):
- Higher |L_da| -> shorter roll-mode time constant (faster roll)
- Roll mode tau decreases roughly inversely with |L_da|
- Dutch roll damping slightly affected through roll-yaw coupling
- Flight quality: quicker roll response, important for maneuvering

RUDDER CONTROL POWER (N_dr):
- Higher |N_dr| -> stronger yaw authority
- Dutch roll damping affected (can improve or destabilize)
- Spiral mode slightly affected
- Flight quality: better sideslip control, coordinated turns

DESIGN IMPLICATIONS:
- Control power must be sized for:
  1. Adequate response (handling qualities)
  2. Controllability margins (not too sensitive)
  3. Trim authority (full flight envelope)
- Roskam Ch5.4 provides guidance on acceptable ranges
""")
    
    plt.close('all')
    
    return elev_results, ail_results, rud_results


if __name__ == "__main__":
    results = main()
