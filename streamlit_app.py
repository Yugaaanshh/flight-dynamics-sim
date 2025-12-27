"""
Flight Dynamics Simulator - Streamlit Dashboard

Interactive web UI for aircraft trim, modal analysis, and sensitivity studies.

Features:
- Plane selection from YAML database
- Live trim computation with parameter sliders
- Mode analysis with interactive charts
- Derivative sensitivity exploration
- Step response comparison

Run: streamlit run streamlit_app.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.integrate import solve_ivp

# Add project to path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_loader import get_available_aircraft, load_aircraft
from eom.six_dof import SixDoFModel, Controls
from eom.longitudinal import LongitudinalLinearModel
from sim.trim_maneuver import trim_steady_state, trim_level_flight
from analysis.linearize import linearize_6dof, analyze_modes


# Page config
st.set_page_config(
    page_title="Flight Dynamics Simulator",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Roskam Flight Dynamics Simulator")
st.markdown("*Interactive trim, modes, and sensitivity analysis*")

# =============================================================================
# SIDEBAR - Aircraft & Flight Condition
# =============================================================================
st.sidebar.header("Aircraft Selection")

# Get available aircraft
aircraft_list = get_available_aircraft()
if not aircraft_list:
    st.error("No aircraft found in planes.yaml!")
    st.stop()

selected_aircraft = st.sidebar.selectbox(
    "Select Aircraft",
    aircraft_list,
    format_func=lambda x: x.replace('_', ' ').title()
)

# Load configuration
@st.cache_data
def get_config(name):
    return load_aircraft(name)

try:
    config = get_config(selected_aircraft)
except Exception as e:
    st.error(f"Error loading {selected_aircraft}: {e}")
    st.stop()

# Display aircraft info
st.sidebar.markdown("---")
st.sidebar.subheader("Aircraft Parameters")
st.sidebar.metric("Mass", f"{config.params.mass:,.0f} kg")
st.sidebar.metric("Wing Area", f"{config.params.S:.1f} m2")
st.sidebar.metric("Wingspan", f"{config.params.b:.1f} m")

# Flight condition sliders
st.sidebar.markdown("---")
st.sidebar.subheader("Flight Condition")

V_cruise = st.sidebar.slider(
    "Airspeed (m/s)",
    min_value=50.0,
    max_value=350.0,
    value=float(config.V_trim),
    step=5.0
)

altitude = st.sidebar.slider(
    "Altitude (m)",
    min_value=0,
    max_value=15000,
    value=10000,
    step=500
)

# =============================================================================
# MAIN TABS
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs(["📐 Trim", "📊 Modes", "📈 Sensitivity", "📉 Responses"])

# =============================================================================
# TAB 1: TRIM ANALYSIS
# =============================================================================
with tab1:
    st.header("Trim Analysis")
    st.markdown("Compute steady-state trim for various maneuvers")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Maneuver Parameters")
        
        nz_target = st.slider(
            "Load Factor (g)",
            min_value=1.0,
            max_value=3.0,
            value=1.0,
            step=0.1
        )
        
        bank_angle = st.slider(
            "Bank Angle (deg)",
            min_value=0.0,
            max_value=60.0,
            value=0.0,
            step=5.0
        )
        
        gamma_angle = st.slider(
            "Flight Path Angle (deg)",
            min_value=-10.0,
            max_value=10.0,
            value=0.0,
            step=1.0
        )
        
        compute_trim = st.button("Compute Trim", type="primary")
    
    with col2:
        if compute_trim or 'trim_result' not in st.session_state:
            try:
                model = SixDoFModel(config)
                trim = trim_steady_state(
                    model,
                    target_V=V_cruise,
                    target_altitude=altitude,
                    nz_target=nz_target,
                    bank_angle_phi=np.radians(bank_angle),
                    flight_path_gamma=np.radians(gamma_angle),
                    verbose=False
                )
                st.session_state.trim_result = trim
                st.session_state.model = model
            except Exception as e:
                st.error(f"Trim failed: {e}")
                trim = None
        else:
            trim = st.session_state.get('trim_result')
        
        if trim and trim.converged:
            st.subheader("Trim Results")
            
            # KPI metrics
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("alpha", f"{np.degrees(trim.alpha):.2f} deg")
            mcol2.metric("delta_e", f"{np.degrees(trim.u_trim[0]):.2f} deg")
            mcol3.metric("CL", f"{trim.CL:.4f}")
            mcol4.metric("Thrust", f"{trim.u_trim[3]:,.0f} N")
            
            # Full table
            st.markdown("**Full Trim State:**")
            trim_data = {
                "Parameter": ["alpha", "theta", "phi", "delta_e", "delta_a", "delta_r", "Thrust", "CL", "nz"],
                "Value": [
                    f"{np.degrees(trim.alpha):.3f} deg",
                    f"{np.degrees(trim.theta):.3f} deg",
                    f"{np.degrees(trim.phi):.1f} deg",
                    f"{np.degrees(trim.u_trim[0]):.3f} deg",
                    f"{np.degrees(trim.u_trim[1]):.3f} deg",
                    f"{np.degrees(trim.u_trim[2]):.3f} deg",
                    f"{trim.u_trim[3]:,.1f} N",
                    f"{trim.CL:.4f}",
                    f"{trim.nz:.2f} g"
                ]
            }
            st.table(trim_data)
        elif trim:
            st.warning("Trim did not converge. Try different parameters.")

# =============================================================================
# TAB 2: MODE ANALYSIS
# =============================================================================
with tab2:
    st.header("Modal Analysis")
    st.markdown("Eigenvalue analysis of linearized dynamics")
    
    if 'trim_result' in st.session_state and st.session_state.trim_result.converged:
        trim = st.session_state.trim_result
        model = st.session_state.model
        
        # Linearize
        A, B = linearize_6dof(model, trim.x_trim, trim.u_trim)
        modes = analyze_modes(A)
        
        # KPI Row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        sp = next((m for m in modes if m.name == 'Short-Period'), None)
        ph = next((m for m in modes if m.name == 'Phugoid'), None)
        dr = next((m for m in modes if m.name == 'Dutch Roll'), None)
        roll = next((m for m in modes if m.name == 'Roll'), None)
        spiral = next((m for m in modes if m.name == 'Spiral'), None)
        
        if sp:
            col1.metric("Short-Period zeta", f"{sp.zeta:.3f}")
        if ph:
            col2.metric("Phugoid zeta", f"{ph.zeta:.3f}")
        if dr:
            col3.metric("Dutch Roll zeta", f"{dr.zeta:.3f}")
        if roll:
            col4.metric("Roll tau", f"{roll.time_constant:.2f} s")
        if spiral:
            col5.metric("Spiral tau", f"{spiral.time_constant:.1f} s")
        
        # Mode table
        st.subheader("Mode Details")
        mode_data = []
        for m in modes:
            if m.is_oscillatory:
                eig_str = f"{m.eigenvalue.real:.4f} +/- {abs(m.eigenvalue.imag):.4f}j"
                time_str = f"T = {m.period:.2f} s"
            else:
                eig_str = f"{m.eigenvalue.real:.4f}"
                time_str = f"tau = {m.time_constant:.2f} s"
            
            mode_data.append({
                "Mode": m.name,
                "Eigenvalue": eig_str,
                "wn (rad/s)": f"{m.omega_n:.3f}",
                "zeta": f"{m.zeta:.3f}",
                "Period/Time Const": time_str
            })
        
        st.table(mode_data)
        
        # Plotly bar chart
        st.subheader("Mode Comparison")
        
        mode_names = [m.name for m in modes]
        zeta_vals = [m.zeta for m in modes]
        wn_vals = [m.omega_n for m in modes]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='zeta', x=mode_names, y=zeta_vals, marker_color='steelblue'))
        fig.add_trace(go.Bar(name='wn (rad/s)', x=mode_names, y=wn_vals, marker_color='coral'))
        fig.update_layout(
            barmode='group',
            title="Damping and Natural Frequency by Mode",
            xaxis_title="Mode",
            yaxis_title="Value",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("Compute a trim solution first in the Trim tab.")

# =============================================================================
# TAB 3: SENSITIVITY ANALYSIS
# =============================================================================
with tab3:
    st.header("Derivative Sensitivity")
    st.markdown("Explore how control effectiveness affects modes")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        derivative = st.selectbox(
            "Select Derivative",
            ["M_delta_e", "L_delta_a", "N_delta_r", "M_alpha", "M_q"],
            format_func=lambda x: x.replace('_', ' ')
        )
        
        scale_min = st.slider("Min Scale", 0.5, 1.0, 0.6, 0.1)
        scale_max = st.slider("Max Scale", 1.0, 2.0, 1.4, 0.1)
        
        run_sensitivity = st.button("Run Sensitivity", type="primary")
    
    with col2:
        if run_sensitivity or 'sens_results' in st.session_state:
            if run_sensitivity:
                # Run sensitivity sweep
                scales = np.linspace(scale_min, scale_max, 5)
                sens_results = {'scales': scales, 'sp_zeta': [], 'sp_wn': [], 'dr_zeta': []}
                
                base_config = config
                model = SixDoFModel(base_config)
                trim = trim_level_flight(model, V_cruise, altitude, verbose=False)
                
                if trim.converged:
                    for scale in scales:
                        # Create scaled config
                        from dataclasses import replace
                        
                        if derivative in ['M_delta_e', 'M_alpha', 'M_q']:
                            scaled_long = replace(
                                base_config.long_derivs,
                                **{derivative: getattr(base_config.long_derivs, derivative) * scale}
                            )
                            scaled_config = load_aircraft(selected_aircraft)
                            scaled_config = replace(scaled_config, long_derivs=scaled_long)
                        else:
                            scaled_lat = replace(
                                base_config.lat_derivs,
                                **{derivative: getattr(base_config.lat_derivs, derivative) * scale}
                            )
                            scaled_config = load_aircraft(selected_aircraft)
                            scaled_config = replace(scaled_config, lat_derivs=scaled_lat)
                        
                        # Linearize and analyze
                        scaled_model = SixDoFModel(scaled_config)
                        A, B = linearize_6dof(scaled_model, trim.x_trim, trim.u_trim)
                        modes = analyze_modes(A)
                        
                        sp = next((m for m in modes if m.name == 'Short-Period'), None)
                        dr = next((m for m in modes if m.name == 'Dutch Roll'), None)
                        
                        sens_results['sp_zeta'].append(sp.zeta if sp else 0)
                        sens_results['sp_wn'].append(sp.omega_n if sp else 0)
                        sens_results['dr_zeta'].append(dr.zeta if dr else 0)
                    
                    st.session_state.sens_results = sens_results
                    st.session_state.sens_deriv = derivative
            
            if 'sens_results' in st.session_state:
                sens = st.session_state.sens_results
                deriv_name = st.session_state.get('sens_deriv', derivative)
                
                st.subheader(f"Sensitivity to {deriv_name}")
                
                # Plotly chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=sens['scales'], y=sens['sp_zeta'],
                    mode='lines+markers', name='Short-Period zeta',
                    line=dict(color='blue', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=sens['scales'], y=sens['sp_wn'],
                    mode='lines+markers', name='Short-Period wn',
                    line=dict(color='red', width=2),
                    yaxis='y2'
                ))
                
                fig.update_layout(
                    title=f"Short-Period Mode vs {deriv_name} Scale",
                    xaxis_title="Scale Factor",
                    yaxis_title="zeta",
                    yaxis2=dict(title="wn (rad/s)", overlaying='y', side='right'),
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Click 'Run Sensitivity' to analyze.")

# =============================================================================
# TAB 4: STEP RESPONSES
# =============================================================================
with tab4:
    st.header("Step Response Analysis")
    st.markdown("Compare linear model responses to control inputs")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        input_type = st.selectbox(
            "Control Input",
            ["Elevator", "Aileron", "Rudder"]
        )
        
        step_magnitude = st.slider(
            "Step Magnitude (deg)",
            min_value=-5.0,
            max_value=5.0,
            value=-2.0,
            step=0.5
        )
        
        sim_duration = st.slider(
            "Duration (s)",
            min_value=5.0,
            max_value=30.0,
            value=15.0,
            step=1.0
        )
        
        run_sim = st.button("Run Simulation", type="primary")
    
    with col2:
        if run_sim:
            # Use longitudinal linear model for elevator
            if input_type == "Elevator":
                model = LongitudinalLinearModel(config.long_derivs)
                
                # Simulate
                step_rad = np.radians(step_magnitude)
                
                def dynamics(t, x):
                    de = step_rad if t >= 1.0 else 0.0
                    return model.A @ x + model.B @ np.array([de])
                
                t_span = (0, sim_duration)
                t_eval = np.linspace(0, sim_duration, 500)
                sol = solve_ivp(dynamics, t_span, np.zeros(4), t_eval=t_eval)
                
                # Plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=sol.t, y=np.degrees(sol.y[2]),
                    mode='lines', name='q (deg/s)',
                    line=dict(color='blue', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=sol.t, y=np.degrees(sol.y[3]),
                    mode='lines', name='theta (deg)',
                    line=dict(color='green', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=sol.t, y=np.degrees(sol.y[1]),
                    mode='lines', name='alpha (deg)',
                    line=dict(color='red', width=2)
                ))
                
                fig.update_layout(
                    title=f"Response to {step_magnitude} deg Elevator Step",
                    xaxis_title="Time (s)",
                    yaxis_title="State Variables (deg or deg/s)",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.info(f"{input_type} response requires 6-DOF simulation (coming soon)")
        else:
            st.info("Set parameters and click 'Run Simulation'")

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown(
    "*Based on Roskam 'Airplane Flight Dynamics' and Zipfel 'Modeling and Simulation'*"
)
