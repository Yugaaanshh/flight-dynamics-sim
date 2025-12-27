"""
Multi-Plane Comparison Demo

Loads multiple aircraft from planes.yaml and compares their
modal characteristics side-by-side.

Usage: python examples/multiplane_demo.py

References:
    Roskam, "Airplane Flight Dynamics", Chapter 5
    Zipfel, "Modeling and Simulation", Chapter 10
"""

import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_loader import get_available_aircraft, load_aircraft, print_aircraft_summary
from eom.six_dof import SixDoFModel
from sim.trim_maneuver import trim_level_flight
from analysis.linearize import linearize_6dof, analyze_modes, print_mode_table


def main():
    print("=" * 70)
    print("Multi-Plane Comparison Demo")
    print("Comparing modal characteristics across aircraft types")
    print("=" * 70)
    
    # Get available aircraft
    aircraft_names = get_available_aircraft()
    print(f"\nAvailable aircraft: {aircraft_names}")
    
    # Analyze each aircraft
    all_results = {}
    
    for name in aircraft_names:
        print(f"\n{'='*70}")
        print(f"Analyzing: {name.upper()}")
        print("=" * 70)
        
        try:
            # Load configuration
            config = load_aircraft(name)
            print(print_aircraft_summary(name, config))
            
            # Create model
            model = SixDoFModel(config)
            
            # Compute trim
            trim = trim_level_flight(model, config.V_trim, 10000.0, verbose=False)
            
            if trim.converged:
                print(f"Trim converged: alpha={np.degrees(trim.alpha):.2f} deg, de={np.degrees(trim.u_trim[0]):.2f} deg")
                
                # Linearize
                A, B = linearize_6dof(model, trim.x_trim, trim.u_trim)
                
                # Analyze modes
                modes = analyze_modes(A)
                
                print(print_mode_table(modes, title=f"{name} Modes"))
                
                all_results[name] = {
                    'config': config,
                    'trim': trim,
                    'modes': modes
                }
            else:
                print(f"  Warning: Trim did not converge for {name}")
                
        except Exception as e:
            print(f"  Error analyzing {name}: {e}")
    
    # Create comparison table
    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("MODE COMPARISON ACROSS AIRCRAFT")
        print("=" * 70)
        
        mode_types = ['Short-Period', 'Phugoid', 'Dutch Roll', 'Roll', 'Spiral']
        
        # Header
        header = f"{'Mode':<15}"
        for name in all_results.keys():
            header += f" {name[:12]:<14}"
        print(f"\n{header}")
        print("-" * (15 + 14 * len(all_results)))
        
        # Damping comparison
        print("\nDamping Ratio (zeta):")
        for mode_type in mode_types:
            line = f"  {mode_type:<13}"
            for name, data in all_results.items():
                mode = next((m for m in data['modes'] if m.name == mode_type), None)
                if mode:
                    line += f" {mode.zeta:<14.3f}"
                else:
                    line += f" {'N/A':<14}"
            print(line)
        
        # Natural frequency comparison
        print("\nNatural Frequency wn (rad/s):")
        for mode_type in mode_types:
            line = f"  {mode_type:<13}"
            for name, data in all_results.items():
                mode = next((m for m in data['modes'] if m.name == mode_type), None)
                if mode:
                    line += f" {mode.omega_n:<14.3f}"
                else:
                    line += f" {'N/A':<14}"
            print(line)
        
        # Create bar chart comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Multi-Aircraft Mode Comparison", fontsize=14, fontweight='bold')
        
        aircraft_labels = list(all_results.keys())
        x = np.arange(len(aircraft_labels))
        width = 0.15
        
        # Short-period and Dutch roll damping
        ax = axes[0]
        sp_zeta = [next((m.zeta for m in all_results[n]['modes'] if m.name == 'Short-Period'), 0) for n in aircraft_labels]
        dr_zeta = [next((m.zeta for m in all_results[n]['modes'] if m.name == 'Dutch Roll'), 0) for n in aircraft_labels]
        
        ax.bar(x - width/2, sp_zeta, width, label='Short-Period', color='steelblue')
        ax.bar(x + width/2, dr_zeta, width, label='Dutch Roll', color='coral')
        ax.set_ylabel('Damping Ratio (zeta)')
        ax.set_xticks(x)
        ax.set_xticklabels(aircraft_labels)
        ax.legend()
        ax.set_title('Oscillatory Mode Damping')
        ax.grid(True, alpha=0.3)
        
        # Roll and spiral time constants
        ax = axes[1]
        roll_tau = [next((m.time_constant for m in all_results[n]['modes'] if m.name == 'Roll'), 0) for n in aircraft_labels]
        spiral_tau = [next((m.time_constant for m in all_results[n]['modes'] if m.name == 'Spiral'), 0) for n in aircraft_labels]
        
        ax.bar(x - width/2, roll_tau, width, label='Roll Mode', color='green')
        ax.bar(x + width/2, [min(t, 50) for t in spiral_tau], width, label='Spiral (capped)', color='purple')
        ax.set_ylabel('Time Constant tau (s)')
        ax.set_xticks(x)
        ax.set_xticklabels(aircraft_labels)
        ax.legend()
        ax.set_title('Aperiodic Mode Time Constants')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = os.path.join(os.path.dirname(__file__), 'multiplane_comparison.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {output_path}")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Multi-plane comparison complete!

Key observations:
- Larger aircraft (747) tend to have slower modes
- Fighter/small jets have faster short-period response
- GA aircraft (Cessna) often have higher damping

To add your own aircraft:
1. Edit planes.yaml
2. Add your derivatives (Roskam notation)
3. Re-run this demo
""")
    
    return all_results


if __name__ == "__main__":
    results = main()
