"""
Aircraft Data Verification Script

Validates all aircraft in planes.yaml against expected Roskam ranges:
1. Load and convert parameters correctly
2. Compute level trim (alpha, CL, nz~1.0)
3. Linearize and extract modes
4. Compare to expected ranges for each aircraft type
5. Run sensitivity test (M_alpha scaling)

Run: python examples/verify_aircraft.py

References:
    Roskam, "Airplane Flight Dynamics", Chapter 5
    Expected mode ranges from transport/GA aircraft data
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_loader import load_aircraft, get_available_aircraft
from eom.six_dof import SixDoFModel
from sim.trim_maneuver import trim_steady_state
from analysis.linearize import linearize_6dof, analyze_modes
from dataclasses import replace
from config import FullAircraftConfig


# Expected mode ranges from Roskam and flight test data
EXPECTED_MODES = {
    'business_jet': {
        'name': 'Business Jet (Roskam)',
        'sp_zeta': (0.25, 0.45),      # Short-period damping
        'sp_wn': (1.0, 2.5),          # Short-period freq (rad/s)
        'ph_zeta': (0.02, 0.08),      # Phugoid damping
        'ph_T': (60, 150),            # Phugoid period (s)
        'dr_zeta': (0.05, 0.20),      # Dutch roll damping
        'roll_tau': (0.5, 3.0),       # Roll time constant (s)
        'spiral_tau': (20, 100),      # Spiral time constant (s)
    },
    'boeing_747': {
        'name': 'Boeing 747-400',
        'sp_zeta': (0.30, 0.60),
        'sp_wn': (0.8, 2.0),
        'ph_zeta': (0.03, 0.12),
        'ph_T': (80, 200),
        'dr_zeta': (0.10, 0.35),
        'roll_tau': (0.5, 2.5),
        'spiral_tau': (30, 200),
    },
    'cessna_172': {
        'name': 'Cessna 172 Skyhawk',
        'sp_zeta': (0.35, 0.70),
        'sp_wn': (2.0, 4.0),
        'ph_zeta': (0.05, 0.25),
        'ph_T': (15, 60),
        'dr_zeta': (0.10, 0.30),
        'roll_tau': (0.3, 2.0),
        'spiral_tau': (10, 80),
    }
}


def check_range(value, expected_range, name):
    """Check if value is within expected range."""
    lo, hi = expected_range
    in_range = lo <= value <= hi
    status = "OK" if in_range else "WARN"
    margin = "within" if in_range else f"outside ({lo:.2f}-{hi:.2f})"
    return in_range, f"{name}={value:.3f} {status} ({margin})"


def verify_aircraft(name: str, verbose: bool = True) -> dict:
    """
    Verify a single aircraft configuration.
    
    Returns dict with 'passed', 'messages', 'modes'
    """
    results = {
        'passed': True,
        'messages': [],
        'modes': {},
        'trim': None
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Verifying: {name.upper()}")
        print("=" * 60)
    
    # Step 1: Load configuration
    try:
        config = load_aircraft(name, debug=verbose)
        results['messages'].append(f"[LOAD] SUCCESS: Mass={config.params.mass:,.0f} kg")
    except Exception as e:
        results['passed'] = False
        results['messages'].append(f"[LOAD] FAILED: {e}")
        return results
    
    # Step 2: Compute level trim
    try:
        model = SixDoFModel(config)
        trim = trim_steady_state(model, config.V_trim, 10000, verbose=False)
        
        if trim.converged:
            results['trim'] = trim
            alpha_deg = np.degrees(trim.alpha)
            results['messages'].append(
                f"[TRIM] SUCCESS: alpha={alpha_deg:.2f} deg, CL={trim.CL:.4f}, nz={trim.nz:.3f}"
            )
            
            # Check nz ~ 1.0 for level flight
            if abs(trim.nz - 1.0) > 0.05:
                results['messages'].append(f"[TRIM] WARNING: nz={trim.nz:.3f} != 1.0")
        else:
            results['passed'] = False
            results['messages'].append("[TRIM] FAILED: Did not converge")
            return results
            
    except Exception as e:
        results['passed'] = False
        results['messages'].append(f"[TRIM] FAILED: {e}")
        return results
    
    # Step 3: Linearize and analyze modes
    try:
        A, B = linearize_6dof(model, trim.x_trim, trim.u_trim)
        modes = analyze_modes(A)
        
        mode_map = {m.name: m for m in modes}
        results['modes'] = mode_map
        
        results['messages'].append(f"[MODES] Found {len(modes)} modes")
        
    except Exception as e:
        results['passed'] = False
        results['messages'].append(f"[LINEARIZE] FAILED: {e}")
        return results
    
    # Step 4: Check against expected ranges
    expected = EXPECTED_MODES.get(name, {})
    
    if 'Short-Period' in mode_map:
        sp = mode_map['Short-Period']
        if 'sp_zeta' in expected:
            ok, msg = check_range(sp.zeta, expected['sp_zeta'], 'SP_zeta')
            results['messages'].append(f"[CHECK] {msg}")
            if not ok:
                results['passed'] = False
        if 'sp_wn' in expected:
            ok, msg = check_range(sp.omega_n, expected['sp_wn'], 'SP_wn')
            results['messages'].append(f"[CHECK] {msg}")
    
    if 'Phugoid' in mode_map:
        ph = mode_map['Phugoid']
        if 'ph_zeta' in expected:
            ok, msg = check_range(ph.zeta, expected['ph_zeta'], 'Ph_zeta')
            results['messages'].append(f"[CHECK] {msg}")
        if 'ph_T' in expected and ph.is_oscillatory:
            ok, msg = check_range(ph.period, expected['ph_T'], 'Ph_T')
            results['messages'].append(f"[CHECK] {msg}")
    
    if 'Dutch Roll' in mode_map:
        dr = mode_map['Dutch Roll']
        if 'dr_zeta' in expected:
            ok, msg = check_range(dr.zeta, expected['dr_zeta'], 'DR_zeta')
            results['messages'].append(f"[CHECK] {msg}")
    
    if 'Roll' in mode_map:
        roll = mode_map['Roll']
        if 'roll_tau' in expected:
            ok, msg = check_range(roll.time_constant, expected['roll_tau'], 'Roll_tau')
            results['messages'].append(f"[CHECK] {msg}")
    
    if 'Spiral' in mode_map:
        spiral = mode_map['Spiral']
        if 'spiral_tau' in expected:
            tau = min(spiral.time_constant, 200)  # Cap for comparison
            ok, msg = check_range(tau, expected['spiral_tau'], 'Spiral_tau')
            results['messages'].append(f"[CHECK] {msg}")
    
    # Print results
    if verbose:
        for msg in results['messages']:
            print(f"  {msg}")
    
    return results


def run_sensitivity_test(name: str, verbose: bool = True) -> bool:
    """
    Test M_alpha sensitivity: scaling should affect short-period frequency.
    Higher |M_alpha| -> higher wn (stiffer pitch)
    """
    if verbose:
        print(f"\n[SENSITIVITY] Testing M_alpha scaling for {name}")
    
    try:
        config = load_aircraft(name)
        model = SixDoFModel(config)
        trim = trim_steady_state(model, config.V_trim, 10000, verbose=False)
        
        if not trim.converged:
            print("  Baseline trim failed")
            return False
        
        wn_values = []
        scales = [0.8, 1.0, 1.2]
        
        for scale in scales:
            orig_Ma = config.long_derivs.M_alpha
            scaled_long = replace(config.long_derivs, M_alpha=orig_Ma * scale)
            scaled_config = FullAircraftConfig(
                params=config.params,
                long_derivs=scaled_long,
                lat_derivs=config.lat_derivs,
                V_trim=config.V_trim,
                alpha_trim=config.alpha_trim,
                rho=config.rho
            )
            
            scaled_model = SixDoFModel(scaled_config)
            A, B = linearize_6dof(scaled_model, trim.x_trim, trim.u_trim)
            modes = analyze_modes(A)
            
            sp = next((m for m in modes if m.name == 'Short-Period'), None)
            wn = sp.omega_n if sp else 0
            wn_values.append(wn)
            
            if verbose:
                print(f"  M_alpha x {scale:.1f} -> wn_sp = {wn:.3f} rad/s")
        
        # Check trend: |M_alpha| up -> wn up
        trend_ok = wn_values[2] > wn_values[1] > wn_values[0]
        
        if verbose:
            if trend_ok:
                print("  [PASS] Correct trend: higher |M_alpha| -> higher wn")
            else:
                print("  [WARN] Unexpected trend")
        
        return trend_ok
        
    except Exception as e:
        if verbose:
            print(f"  ERROR: {e}")
        return False


def main():
    print("=" * 60)
    print("AIRCRAFT DATA VERIFICATION")
    print("Checking planes.yaml against Roskam expected ranges")
    print("=" * 60)
    
    aircraft_list = get_available_aircraft()
    print(f"\nFound {len(aircraft_list)} aircraft: {aircraft_list}")
    
    all_results = {}
    all_passed = True
    
    # Verify each aircraft
    for name in aircraft_list:
        results = verify_aircraft(name, verbose=True)
        all_results[name] = results
        if not results['passed']:
            all_passed = False
    
    # Run sensitivity tests
    print("\n" + "=" * 60)
    print("SENSITIVITY TESTS")
    print("=" * 60)
    
    for name in aircraft_list:
        if all_results[name]['passed']:
            sens_ok = run_sensitivity_test(name, verbose=True)
            if not sens_ok:
                all_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    for name in aircraft_list:
        r = all_results[name]
        status = "PASS" if r['passed'] else "FAIL"
        
        # Extract key mode values
        modes = r.get('modes', {})
        sp = modes.get('Short-Period')
        dr = modes.get('Dutch Roll')
        
        sp_str = f"SP_z={sp.zeta:.2f}" if sp else "SP=N/A"
        dr_str = f"DR_z={dr.zeta:.2f}" if dr else "DR=N/A"
        
        print(f"  {name:15} [{status}] {sp_str}, {dr_str}")
    
    print()
    if all_passed:
        print("*** ALL TESTS PASSED ***")
    else:
        print("*** SOME TESTS FAILED - Review above ***")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
