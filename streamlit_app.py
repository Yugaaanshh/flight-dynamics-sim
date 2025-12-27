"""
Flight Dynamics Simulator - Multi-Aircraft Streamlit Dashboard

Features:
- Multi-aircraft selection (Business Jet, Boeing 747, Cessna 172)
- Dynamic sliders based on aircraft reference conditions
- Compare All mode for side-by-side analysis
- Full trim, modes, sensitivity, and response tabs

Run: python -m streamlit run streamlit_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from dataclasses import replace

# Page config - MUST be first
st.set_page_config(
    page_title="Flight Dynamics Simulator",
    page_icon="✈️",
    layout="wide"
)

# Add project to path
import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import project modules
try:
    from config_loader import load_aircraft, get_available_aircraft, get_aircraft_info
    from config import BUSINESS_JET_6DOF, ROSKAM_BUSINESS_JET
    from eom.six_dof import SixDoFModel
    from eom.longitudinal import LongitudinalLinearModel
    from sim.trim_maneuver import trim_steady_state
    from analysis.linearize import linearize_6dof, analyze_modes
    IMPORTS_OK = True
except ImportError as e:
    IMPORTS_OK = False
    IMPORT_ERROR = str(e)

# Try plotly
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    import matplotlib.pyplot as plt

# =============================================================================
# HEADER
# =============================================================================
st.title("✈️ Flight Dynamics Simulator")
st.markdown("*Multi-aircraft trim, modes, and sensitivity analysis*")

if not IMPORTS_OK:
    st.error(f"Import error: {IMPORT_ERROR}")
    st.stop()

# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.header("🛩️ Aircraft Selection")

# Get available aircraft
aircraft_list = get_available_aircraft()
if not aircraft_list:
    aircraft_list = ['business_jet']

selected_aircraft = st.sidebar.selectbox(
    "Aircraft",
    aircraft_list,
    format_func=lambda x: x.replace('_', ' ').title()
)

# Load config
try:
    config = load_aircraft(selected_aircraft)
    info = get_aircraft_info(selected_aircraft)
    LOAD_OK = True
except Exception as e:
    st.sidebar.error(f"Load error: {e}")
    # Fallback to hardcoded
    config = BUSINESS_JET_6DOF
    info = {'name': 'Business Jet', 'V_ref': 205.8, 'h_ref': 10668}
    LOAD_OK = False

# Display aircraft info
st.sidebar.markdown("---")
st.sidebar.markdown(f"**{info.get('name', selected_aircraft)}**")
st.sidebar.text(f"Mass: {config.params.mass:,.0f} kg")
st.sidebar.text(f"Wing: S={config.params.S:.1f} m2, b={config.params.b:.1f} m")

# Dynamic flight condition sliders
st.sidebar.markdown("---")
st.sidebar.markdown("**Flight Condition**")

# Set defaults based on aircraft
V_default = info.get('V_ref', config.V_trim)
h_default = info.get('h_ref', 10000)

# Scale slider ranges by aircraft type
if 'cessna' in selected_aircraft.lower():
    V_min, V_max = 30.0, 100.0
    h_min, h_max = 0, 5000
elif '747' in selected_aircraft.lower():
    V_min, V_max = 150.0, 350.0
    h_min, h_max = 0, 15000
else:
    V_min, V_max = 100.0, 300.0
    h_min, h_max = 0, 12000

V_cruise = st.sidebar.slider(
    "Airspeed V (m/s)",
    min_value=V_min, max_value=V_max,
    value=float(np.clip(V_default, V_min, V_max)),
    step=5.0
)

altitude = st.sidebar.slider(
    "Altitude (m)",
    min_value=h_min, max_value=h_max,
    value=int(np.clip(h_default, h_min, h_max)),
    step=500
)

# =============================================================================
# TABS
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs(["📐 Trim", "📊 Modes", "📈 Sensitivity", "📉 Response"])

# =============================================================================
# TAB 1: TRIM
# =============================================================================
with tab1:
    st.header("Trim Analysis")
    st.markdown(f"*Aircraft: {info.get('name', selected_aircraft)}*")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Parameters")
        
        nz_target = st.slider("Load Factor nz (g)", 1.0, 3.0, 1.0, 0.1, key="t_nz")
        bank_angle = st.slider("Bank Angle phi (deg)", 0.0, 60.0, 0.0, 5.0, key="t_phi")
        gamma_angle = st.slider("Flight Path gamma (deg)", -10.0, 10.0, 0.0, 1.0, key="t_gam")
        
        run_trim = st.button("🔄 Compute Trim", type="primary", key="run_trim")
    
    with col2:
        st.subheader("Results")
        
        if run_trim:
            with st.spinner("Computing trim..."):
                try:
                    model = SixDoFModel(config)
                    trim = trim_steady_state(
                        model, target_V=V_cruise, target_altitude=altitude,
                        nz_target=nz_target,
                        bank_angle_phi=np.radians(bank_angle),
                        flight_path_gamma=np.radians(gamma_angle),
                        verbose=False
                    )
                    
                    if trim.converged:
                        st.session_state['trim'] = trim
                        st.session_state['model'] = model
                        st.session_state['config'] = config
                        
                        k1, k2, k3, k4 = st.columns(4)
                        k1.metric("alpha", f"{np.degrees(trim.alpha):.2f} deg")
                        k2.metric("delta_e", f"{np.degrees(trim.u_trim[0]):.2f} deg")
                        k3.metric("CL", f"{trim.CL:.4f}")
                        k4.metric("Thrust", f"{trim.u_trim[3]:,.0f} N")
                        
                        df = pd.DataFrame({
                            "Parameter": ["alpha", "theta", "phi", "delta_e", "delta_a", "delta_r", "CL", "nz"],
                            "Value": [
                                f"{np.degrees(trim.alpha):.3f} deg",
                                f"{np.degrees(trim.theta):.3f} deg",
                                f"{np.degrees(trim.phi):.1f} deg",
                                f"{np.degrees(trim.u_trim[0]):.3f} deg",
                                f"{np.degrees(trim.u_trim[1]):.3f} deg",
                                f"{np.degrees(trim.u_trim[2]):.3f} deg",
                                f"{trim.CL:.4f}",
                                f"{trim.nz:.2f}"
                            ]
                        })
                        st.dataframe(df, hide_index=True, use_container_width=True)
                        st.success("Trim converged!")
                    else:
                        st.warning("Trim did not converge.")
                except Exception as e:
                    st.error(f"Trim error: {e}")

# =============================================================================
# TAB 2: MODES (with Compare All)
# =============================================================================
with tab2:
    st.header("Modal Analysis")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        run_modes = st.button("🔍 Analyze Modes", type="primary", key="run_modes")
        st.markdown("---")
        compare_all = st.button("📊 Compare All Aircraft", key="compare_all")
    
    with col2:
        # Single aircraft mode analysis
        if run_modes and 'trim' in st.session_state:
            trim = st.session_state['trim']
            model = st.session_state['model']
            
            with st.spinner("Linearizing..."):
                try:
                    A, B = linearize_6dof(model, trim.x_trim, trim.u_trim)
                    modes = analyze_modes(A)
                    st.session_state['modes'] = modes
                except Exception as e:
                    st.error(f"Error: {e}")
                    modes = None
        else:
            modes = st.session_state.get('modes')
        
        if modes and not compare_all:
            cols = st.columns(5)
            mode_map = {m.name: m for m in modes}
            
            if 'Short-Period' in mode_map:
                cols[0].metric("SP zeta", f"{mode_map['Short-Period'].zeta:.3f}")
            if 'Phugoid' in mode_map:
                cols[1].metric("Ph zeta", f"{mode_map['Phugoid'].zeta:.3f}")
            if 'Dutch Roll' in mode_map:
                cols[2].metric("DR zeta", f"{mode_map['Dutch Roll'].zeta:.3f}")
            if 'Roll' in mode_map:
                cols[3].metric("Roll tau", f"{mode_map['Roll'].time_constant:.2f}s")
            if 'Spiral' in mode_map:
                cols[4].metric("Spiral", f"{min(mode_map['Spiral'].time_constant, 99):.1f}s")
            
            # Mode table
            mode_data = []
            for m in modes:
                eig_str = f"{m.eigenvalue.real:.4f}"
                if m.is_oscillatory:
                    eig_str += f" +/- {abs(m.eigenvalue.imag):.4f}j"
                mode_data.append({
                    "Mode": m.name,
                    "Eigenvalue": eig_str,
                    "wn": round(m.omega_n, 3),
                    "zeta": round(m.zeta, 3)
                })
            st.dataframe(pd.DataFrame(mode_data), hide_index=True)
        
        # Compare All Aircraft
        if compare_all:
            st.subheader("Multi-Aircraft Comparison")
            
            comparison = []
            
            with st.spinner("Analyzing all aircraft..."):
                for ac_name in aircraft_list:
                    try:
                        ac_config = load_aircraft(ac_name)
                        ac_info = get_aircraft_info(ac_name)
                        
                        # Use ref conditions
                        ac_V = ac_info.get('V_ref', ac_config.V_trim)
                        ac_h = ac_info.get('h_ref', 10000)
                        
                        ac_model = SixDoFModel(ac_config)
                        ac_trim = trim_steady_state(ac_model, ac_V, ac_h, verbose=False)
                        
                        if ac_trim.converged:
                            A, B = linearize_6dof(ac_model, ac_trim.x_trim, ac_trim.u_trim)
                            ac_modes = analyze_modes(A)
                            
                            mode_map = {m.name: m for m in ac_modes}
                            
                            comparison.append({
                                "Aircraft": ac_info.get('name', ac_name),
                                "SP zeta": mode_map.get('Short-Period', type('', (), {'zeta': 0})()).zeta,
                                "SP wn": mode_map.get('Short-Period', type('', (), {'omega_n': 0})()).omega_n,
                                "Ph zeta": mode_map.get('Phugoid', type('', (), {'zeta': 0})()).zeta,
                                "DR zeta": mode_map.get('Dutch Roll', type('', (), {'zeta': 0})()).zeta,
                                "Roll tau": mode_map.get('Roll', type('', (), {'time_constant': 0})()).time_constant,
                            })
                    except Exception as e:
                        comparison.append({
                            "Aircraft": ac_name,
                            "SP zeta": f"Err",
                            "SP wn": "",
                            "Ph zeta": "",
                            "DR zeta": "",
                            "Roll tau": ""
                        })
            
            if comparison:
                df_comp = pd.DataFrame(comparison)
                st.dataframe(df_comp, hide_index=True, use_container_width=True)
                
                # Bar chart
                if HAS_PLOTLY:
                    fig = go.Figure()
                    names = [c['Aircraft'] for c in comparison if isinstance(c.get('SP zeta'), (int, float))]
                    sp_zeta = [c['SP zeta'] for c in comparison if isinstance(c.get('SP zeta'), (int, float))]
                    dr_zeta = [c['DR zeta'] for c in comparison if isinstance(c.get('DR zeta'), (int, float))]
                    
                    fig.add_trace(go.Bar(name='Short-Period', x=names, y=sp_zeta, marker_color='steelblue'))
                    fig.add_trace(go.Bar(name='Dutch Roll', x=names, y=dr_zeta, marker_color='coral'))
                    fig.update_layout(barmode='group', title="Damping Ratio Comparison", height=400)
                    st.plotly_chart(fig, use_container_width=True)
        
        elif not modes and not compare_all:
            st.info("Compute trim first, or click 'Compare All Aircraft'")

# =============================================================================
# TAB 3: SENSITIVITY
# =============================================================================
with tab3:
    st.header("Derivative Sensitivity")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        deriv = st.selectbox("Derivative", ["M_delta_e", "M_alpha", "M_q"], key="sens_d")
        scale_range = st.slider("Scale Range", 0.5, 1.5, (0.6, 1.4), 0.1, key="sens_r")
        run_sens = st.button("📊 Run Sensitivity", type="primary", key="run_sens")
    
    with col2:
        if run_sens:
            with st.spinner("Running sensitivity..."):
                try:
                    scales = np.linspace(scale_range[0], scale_range[1], 9)
                    sp_zeta, sp_wn = [], []
                    
                    model = SixDoFModel(config)
                    trim = trim_steady_state(model, V_cruise, altitude, verbose=False)
                    
                    if trim.converged:
                        for scale in scales:
                            orig = getattr(config.long_derivs, deriv)
                            scaled_long = replace(config.long_derivs, **{deriv: orig * scale})
                            
                            from config import FullAircraftConfig
                            scaled_cfg = FullAircraftConfig(
                                params=config.params,
                                long_derivs=scaled_long,
                                lat_derivs=config.lat_derivs,
                                V_trim=config.V_trim,
                                alpha_trim=config.alpha_trim,
                                rho=config.rho
                            )
                            
                            scaled_model = SixDoFModel(scaled_cfg)
                            A, B = linearize_6dof(scaled_model, trim.x_trim, trim.u_trim)
                            modes = analyze_modes(A)
                            
                            sp = next((m for m in modes if m.name == 'Short-Period'), None)
                            sp_zeta.append(sp.zeta if sp else 0)
                            sp_wn.append(sp.omega_n if sp else 0)
                        
                        if HAS_PLOTLY:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=scales, y=sp_zeta, mode='lines+markers', name='zeta', line=dict(color='blue')))
                            fig.add_trace(go.Scatter(x=scales, y=sp_wn, mode='lines+markers', name='wn', line=dict(color='red'), yaxis='y2'))
                            fig.update_layout(
                                title=f"Short-Period vs {deriv}",
                                xaxis_title="Scale",
                                yaxis=dict(title="zeta"),
                                yaxis2=dict(title="wn (rad/s)", overlaying='y', side='right'),
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.line_chart(pd.DataFrame({'zeta': sp_zeta, 'wn': sp_wn}, index=scales))
                except Exception as e:
                    st.error(f"Error: {e}")

# =============================================================================
# TAB 4: RESPONSE
# =============================================================================
with tab4:
    st.header("Step Response")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        de_step = st.slider("Elevator (deg)", -5.0, 5.0, -2.0, 0.5, key="resp_de")
        sim_time = st.slider("Duration (s)", 5.0, 30.0, 15.0, 1.0, key="resp_t")
        run_resp = st.button("▶️ Simulate", type="primary", key="run_resp")
    
    with col2:
        if run_resp:
            try:
                lin_model = LongitudinalLinearModel(config.long_derivs)
                de_rad = np.radians(de_step)
                
                def dynamics(t, x):
                    de = de_rad if t >= 1.0 else 0.0
                    return lin_model.A @ x + lin_model.B @ np.array([de])
                
                sol = solve_ivp(dynamics, (0, sim_time), np.zeros(4), 
                               t_eval=np.linspace(0, sim_time, 500), method='RK45')
                
                if HAS_PLOTLY:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=sol.t, y=np.degrees(sol.y[2]), name='q (deg/s)'))
                    fig.add_trace(go.Scatter(x=sol.t, y=np.degrees(sol.y[3]), name='theta (deg)'))
                    fig.add_trace(go.Scatter(x=sol.t, y=np.degrees(sol.y[1]), name='alpha (deg)'))
                    fig.update_layout(title=f"{de_step} deg Elevator Step", xaxis_title="Time (s)", height=450)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    df = pd.DataFrame({
                        'q': np.degrees(sol.y[2]),
                        'theta': np.degrees(sol.y[3]),
                        'alpha': np.degrees(sol.y[1])
                    }, index=sol.t)
                    st.line_chart(df)
            except Exception as e:
                st.error(f"Error: {e}")

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.caption("Based on Roskam & Zipfel | Multi-Aircraft Edition")
