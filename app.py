#last update: 260202

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, Tuple

 # --- 1. Physics Engine ---
class CMPPhysicsEngine:
    @staticmethod
    def solve_joint_vectorized(d_ori: np.ndarray, d_J: float, Jx: float, Jy: float) -> Tuple[np.ndarray, np.ndarray]:
        """ Restore the original linkage solving logic (mechanical inverse kinematics)"""
        d2 = Jx**2 + Jy**2
        d = np.sqrt(d2)
        K = (d_ori**2 + d2 - d_J**2) / 2.0
        # Discriminant
        disc = (d_ori**2 * d2) - K**2 	# Avoid generating complex numbers in physically unreachable zones
        disc = np.maximum(disc, 0) 
        sqrt_disc = np.sqrt(disc)
        
        # Original code logic: xp = (KJx + Jysqrt_disc)/d2; yp = (KJy - Jxsqrt_disc)/d2
        xp = (K * Jx + Jy * sqrt_disc) / d2
        yp = (K * Jy - Jx * sqrt_disc) / d2
        return xp, yp

    @staticmethod
    def get_sweep_radius(mode: str, t: np.ndarray, swps_min: float, df: pd.DataFrame) -> np.ndarray:
        """Calculate sweep radius according to table definition"""
        half_cycle = (60.0 / swps_min) / 2.0
        t_cycle = half_cycle * 2
        t_mod = t % t_cycle
        t_lookup = np.where(t_mod <= half_cycle, t_mod, t_cycle - t_mod)

        rel_times = df["Relative_Time"].values
        cum_rel = np.cumsum(rel_times)
        xp = np.concatenate(([0], (cum_rel / cum_rel[-1]) * half_cycle))
        fp = np.concatenate(([df["Zone_Start"].iloc[0] * 25.4], df["Zone_End"].values * 25.4))
        return np.interp(t_lookup, xp, fp)

# --- 2. Streamlit UI & var ---
st.set_page_config(layout="wide", page_title="CMP Head Disk Trajectory Simulator")
st.markdown("<style>.stApp { max-width: 100% !important; padding: 0 20px !important; }</style>", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Setting")
    PLATEN_RPM = st.number_input("Platen RPM", 0, 200, 86)
    HEAD_RPM = st.number_input("Head RPM", 0, 200, 92)
    #STEP_SEC = st.slider("Second per STEP", 0.05, 0.1, 0.05)
    TOTAL_TIME = st.slider("TOTAL_TIME (s)", 1, 99, 10)
    POINTA_RADIUS = st.number_input("PointA Radius (mm)", 1, 150, 100)
    st.divider()
    SHOW_BLUE = st.checkbox("Show Point 1 (Blue)", True)
    SHOW_GREEN = st.checkbox("Show Point 2 (Green)", True)
    SHOW_ORANGE = st.checkbox("Show Disk (Orange)", True)

# --- Sweep Profile Tables ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("Head Sweep (Wafer)")
    h_mode = st.radio("Head Mode", ["Sine", "Custom"], horizontal=True, key="hm")
    hc1, hc2 = st.columns(2)
    h_start = hc1.number_input("Head START (in)", 6.0, 9.0, 7.2, 0.1)
    h_end = hc2.number_input("Head END (in)", 6.0, 9.0, 8.2, 0.1)
    h_zn = st.number_input("Head Sweep Zone Num", 5, 20, 10)
    
    # Build Sine‑distributed Relative Time
    h_bounds = np.linspace(h_start, h_end, h_zn + 1)
    h_rel = [1.0] * h_zn
    if h_mode == "Sine":
        mid, amp = (h_start + h_end) / 2.0, (h_end - h_start) / 2.0
        h_theta = np.arccos(np.clip((h_bounds - mid) / amp, -1, 1))
        h_rel = np.abs(np.diff(h_theta)).round(3)

    h_df = pd.DataFrame({"Zone": [f"Z{i+1}" for i in range(h_zn)], "Zone_Start": h_bounds[:-1].round(2), "Zone_End": h_bounds[1:].round(2), "Relative_Time": h_rel})
    h_editor_df = st.data_editor(h_df, hide_index=True, disabled=["Zone", "Zone_Start", "Zone_End"] if h_mode=="Custom" else True, key="h_edit")
    h_swps = st.slider("Head Swps/min", 1, 30, 10)

with col2:
    st.subheader("Disk Sweep")
    d_mode = st.radio("Disk Mode", ["Sine", "Custom"], horizontal=True, key="dm")
    dc1, dc2 = st.columns(2)
    d_start = dc1.number_input("Disk NEAR (END) (in)", 1.0, 15.0, 2.7, 0.01)
    d_end = dc2.number_input("Disk FAR (START) (in)", 1.0, 15.0, 14.7, 0.01)
    d_zn = st.number_input("Disk Sweep Zone Num", 5, 20, 13)

    # Disk logic: from Far to Near 
    d_bounds = np.linspace(d_end, d_start, d_zn + 1) # Far to Near
    d_rel = [1.0] * d_zn
    if d_mode == "Sine":
        mid, amp = (d_start + d_end) / 2.0, (d_end - d_start) / 2.0
        d_theta = np.arccos(np.clip((d_bounds - mid) / amp, -1, 1))
        d_rel = np.abs(np.diff(d_theta)).round(3)

    d_df = pd.DataFrame({"Zone": [f"Z{i+1}" for i in range(d_zn)], "Zone_Start": d_bounds[:-1].round(2), "Zone_End": d_bounds[1:].round(2), "Relative_Time": d_rel})
    d_editor_df = st.data_editor(d_df, hide_index=True, disabled=["Zone", "Zone_Start", "Zone_End"] if d_mode=="Custom" else True, key="d_edit")
    d_swps = st.slider("Disk Swps per minute", 1, 30, 13)

# --- 3. core cal. Main loop ---
STEP_SEC = 0.5
if st.button("🚀 Update & Generate Plot", use_container_width=True):
    t = np.arange(0, TOTAL_TIME + STEP_SEC, STEP_SEC)
    
    # 1. Rotation parameters and angles (restore original physical phase)
    platen_ang_deg = (PLATEN_RPM * 6.0) * t
    platen_rad = np.radians(90 - platen_ang_deg)
    
    # 2. Radius profile
    r_h = CMPPhysicsEngine.get_sweep_radius(h_mode, t, h_swps, h_editor_df)
    r_d = CMPPhysicsEngine.get_sweep_radius(d_mode, t, d_swps, d_editor_df)
    
    # 3. Wafer / Head trajectory
    wx, wy = r_h * np.cos(platen_rad), r_h * np.sin(platen_rad)
    
    # Restore Head‑to‑Platen phase correction: step * deg_pointa + 180 + platen_rad
    head_rel_rad = np.radians((HEAD_RPM * 6.0) * t + 180) + platen_rad
    p1x = wx + POINTA_RADIUS * np.cos(head_rel_rad + np.pi)
    p1y = wy + POINTA_RADIUS * np.sin(head_rel_rad + np.pi)
    p2x = wx + POINTA_RADIUS * np.cos(head_rel_rad)
    p2y = wy + POINTA_RADIUS * np.sin(head_rad_rel := head_rel_rad) # temp reuse
    p2y = wy + POINTA_RADIUS * np.sin(head_rel_rad)
    

    # 4. Disk linkage and phase rotation (restore original J_INIT and rotation logic)
    K_INIT = np.array([450.0, -420.0])
    # Pre‑compute J_INIT after a 45° rotation (restore position before solve_joint)
    cos45, sin45 = np.cos(np.radians(45)), np.sin(np.radians(45))
    J_INIT_X = K_INIT[0] * cos45 - K_INIT[1] * sin45
    J_INIT_Y = K_INIT[0] * sin45 + K_INIT[1] * cos45
    
    # Mechanical solution (raw_dx, raw_dy are in the stationary reference frame)
    raw_dx, raw_dy = CMPPhysicsEngine.solve_joint_vectorized(r_d, 610.0, J_INIT_X, J_INIT_Y)
    
    # Transform Disk coordinates to the rotating Platen reference frame: rotate by -platen_angle
    rot_inv_rad = np.radians(-platen_ang_deg)
    dx = raw_dx * np.cos(rot_inv_rad) - raw_dy * np.sin(rot_inv_rad)
    dy = raw_dx * np.sin(rot_inv_rad) + raw_dy * np.cos(rot_inv_rad)

    # 5. Revolution data
    rev_p = (PLATEN_RPM * 6.0 * t) / 360.0
    rev_w = (HEAD_RPM * 6.0 * t) / 360.0

    # --- 4. Plotly animation configuration ---
    fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], vertical_spacing=0.1)
    
    # Base layers
    style = dict(width=1, shape='spline', smoothing=1.3)
    fig.add_trace(go.Scatter(x=p1x, y=p1y, name="P1 (Blue)", line=dict(color='blue', **style), opacity=1 if SHOW_BLUE else 0), row=1, col=1)
    fig.add_trace(go.Scatter(x=p2x, y=p2y, name="P2 (Green)", line=dict(color='green', **style), opacity=1 if SHOW_GREEN else 0), row=1, col=1)
    fig.add_trace(go.Scatter(x=wx, y=wy, name="Wafer Center", line=dict(color='black', dash='dot', **style)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dx, y=dy, name="Disk (Orange)", line=dict(color='orange', **style), opacity=1 if SHOW_ORANGE else 0), row=1, col=1)
    
    
    # Distance plot
    fig.add_trace(go.Scatter(x=t, y=np.hypot(p1x, p1y), name="Dist P1", line=dict(color='blue', **style), opacity=1 if SHOW_BLUE else 0), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=np.hypot(p2x, p2y), name="Dist P2", line=dict(color='green', **style), opacity=1 if SHOW_GREEN else 0), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=r_h, name="Dist Wafer", line=dict(color='black', dash='dot', **style)), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=r_d, name="Dist Disk", line=dict(color='orange', **style), opacity=1 if SHOW_ORANGE else 0), row=2, col=1)

    # Dynamic points
    fig.add_trace(go.Scatter(x=[p1x[0]], y=[p1y[0]], mode="markers", marker=dict(size=8, color="blue"), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=[p2x[0]], y=[p2y[0]], mode="markers", marker=dict(size=8, color="green"), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=[dx[0]], y=[dy[0]], mode="markers", marker=dict(size=8, color="orange"), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=[wx[0]], y=[wy[0]], mode="markers", marker=dict(size=8, color="black"), showlegend=False), row=1, col=1)

    # Illustrative circles
    fig.add_shape(type="circle", x0=-390, y0=-390, x1=390, y1=390, line_color="black", row=1, col=1)
    fig.add_shape(type="circle", x0=-150, y0=h_start*25.4 -150, x1= 150, y1= h_start*25.4 + 150, line_color="black", row=1, col=1)
    
    # Animation frames (Rev label updates are handled here)
    frames = []
    indices = np.linspace(0, len(t)-1, 60, dtype=int)
    for i in indices:
        current_p_rev = rev_p[i]
        current_w_rev = rev_w[i]
        current_sec   = t[i]
        
        frame_annotation = [dict(
        text=f"Platen Rev: {current_p_rev:.1f} | Wafer Rev: {current_w_rev:.1f} | Time: {current_sec:.1f}s",
        xref="paper", yref="paper", 
        x=0.02, y=0.98, # Position: top‑left corner
        showarrow=False,
        font=dict(size=14, color="black"),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1,
        align="left"
    )]
        frames.append(go.Frame(
            data=[
                go.Scatter(x=p1x[:i+1], y=p1y[:i+1]), 
                go.Scatter(x=p2x[:i+1], y=p2y[:i+1]), 
                go.Scatter(x=wx[:i+1], y=wy[:i+1]), 
                go.Scatter(x=dx[:i+1], y=dy[:i+1]),
                
                go.Scatter(x=t[:i+1], y=np.hypot(p1x[:i+1], p1y[:i+1])), 
                go.Scatter(x=t[:i+1], y=np.hypot(p2x[:i+1], p2y[:i+1])),
                go.Scatter(x=t[:i+1], y=r_h[:i+1]), go.Scatter(x=t[:i+1], y=r_d[:i+1]),
                go.Scatter(x=[p1x[i]], y=[p1y[i]]), go.Scatter(x=[p2x[i]], y=[p2y[i]]), 
                go.Scatter(x=[dx[i]], y=[dy[i]]), go.Scatter(x=[wx[i]], y=[wy[i]])
            ],
            name=f"f{i}",
            layout=go.Layout(annotations=frame_annotation)
        ))
    fig.frames = frames

    fig.update_layout(
        height=900,
        xaxis=dict(range=[-450, 450], scaleanchor="y", scaleratio=1, dtick=100, showgrid=True, title="X (mm)"),
        yaxis=dict(range=[-450, 450], showgrid=True, dtick=100, title="Y (mm)"),
        xaxis2=dict(title="Time (s)", range=[0, TOTAL_TIME]),
        yaxis2=dict(title="Distance (mm)", dtick=100, range=[0, 400]),
        sliders=[{"active": 0, "steps": [{"args": [[f.name], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                                          "label": f"{t[int(f.name[1:])]:.1f}", "method": "animate"} for f in frames]}],
        updatemenus=[{"type": "buttons", "buttons": [{"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": 40, "redraw": False}, "fromcurrent": True}]},
                                                     {"label": "Pause", "method": "animate", "args": [[None], {"mode": "immediate"}]}]}]
    )
    st.plotly_chart(fig, use_container_width=True)



    # ---------------------------------------------------------
    # Dwell‑time statistics logic  (placed inside the button‑conditional statement)
    # ---------------------------------------------------------
    # ---  New: Point 1 and Disk dwell‑time distribution calculation ---
    st.divider()
    st.subheader(" Time Distribution")

   # Define interval boundaries
    bins = [0, 50, 100, 150, 200, 250, 300, 350, 390]
    bin_labels = [f"{bins[i]}~{bins[i+1]}" for i in range(len(bins)-1)]

    # --- Calculate Point 1 (Blue) ---
    p1_radii = np.hypot(p1x, p1y)
    p1_counts, _ = np.histogram(p1_radii, bins=bins)
    p1_dwell = p1_counts * STEP_SEC
    p1_percent = (p1_dwell / p1_dwell.sum() * 100) if p1_dwell.sum() > 0 else p1_dwell * 0

    # --- Calculate Disk (Orange) ---
    disk_radii = np.hypot(dx, dy) # Use previously calculated dx, dy
    disk_counts, _ = np.histogram(disk_radii, bins=bins)
    disk_dwell = disk_counts * STEP_SEC
    disk_percent = (disk_dwell / disk_dwell.sum() * 100) if disk_dwell.sum() > 0 else disk_dwell * 0

    # Create the integrated DataFrame
    analysis_df = pd.DataFrame({  
    "Radius Range (mm)": bin_labels,  
    "P1  (s)": p1_dwell.round(3),  
    "P1  (%)": p1_percent.round(2),  
    "Disk  (s)": disk_dwell.round(3),  
    "Disk  (%)": disk_percent.round(2)  
    })


    # Plot comparison bar chart
    fig_compare = go.Figure()

    # Add Point 1 series
    fig_compare.add_trace(go.Bar(
        x=bin_labels,
        y=p1_percent,
        name="Point 1 (Wafer Blue)",
        marker_color='royalblue',
        text=p1_percent.round(1).astype(str) + '%',
        textposition='auto',
    ))

    # Add Disk series
    fig_compare.add_trace(go.Bar(
        x=bin_labels,
        y=disk_percent,
        name="Disk (Orange)",
        marker_color='darkorange',
        text=disk_percent.round(1).astype(str) + '%',
        textposition='auto',
    ))

    fig_compare.update_layout(
        
        xaxis_title="Pad radius interval (mm)",
        yaxis_title="Time proportion (%)",
        barmode='group', # side‑by‑side display
        height=500,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    col_table, col_chart = st.columns([1, 1])   
    with col_table:  
        st.dataframe(analysis_df, hide_index=True, use_container_width=True)
    with col_chart:  
        st.plotly_chart(fig_compare, use_container_width=True) 
