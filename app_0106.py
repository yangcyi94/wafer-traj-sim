#last update: 260106

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots 
import math
import pandas as pd

st.set_page_config(layout="wide", page_title="traj")

st.markdown(
    """
    <style>
    .stApp { max-width: 100% !important; padding: 0 20px !important; }
    </style>
    """,
    unsafe_allow_html=True
)
# -------------------------------------------------
# 1️⃣ Initialize Session State (Ensure all variables exist)
# -------------------------------------------------
DEFAULTS = {
    "STEP_SEC": 0.05,
    "TOTAL_TIME": 10,
    "PLATEN_RPM": 86,
    "POINTA_RPM": 92,
    "POINTA_RADIUS": 100,
    "SWEEP_START": 7.2,
    "SWEEP_END": 8.2,
    "SWPS_MIN": 10,
    "PEND_MODE": "Sine",
    "DSWEEP_START": 2.7,
    "DSWEEP_END": 14.7,
    "DSWPS_MIN": 13,
    "SHOW_BLUE": True,     
    "SHOW_GREEN": False,
    "SHOW_ORANGE": False,
    "show_plot": False
}

for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# -------------------------------------------------
# 2️⃣ Sidebar: Core Simulation Settings
# -------------------------------------------------
st.sidebar.header("⚙️ Setting")
with st.sidebar:
    st.session_state.PLATEN_RPM = st.number_input("Platen RPM", 0, 200, st.session_state.PLATEN_RPM)
    st.session_state.POINTA_RPM = st.number_input("Head RPM", 0, 200, st.session_state.POINTA_RPM)
    st.session_state.STEP_SEC = st.slider("Second per STEP", 0.01, 0.1, float(st.session_state.STEP_SEC))
    st.session_state.TOTAL_TIME = st.slider("TOTAL_TIME (s)", 1, 99, int(st.session_state.TOTAL_TIME))
    st.session_state.POINTA_RADIUS = st.number_input("PointA Radius (mm)", 1, 150, st.session_state.POINTA_RADIUS)
    
    STEP_SEC   = st.session_state.STEP_SEC
    TOTAL_TIME = st.session_state.TOTAL_TIME
    STEPS_TOTAL = int(TOTAL_TIME / STEP_SEC) + 1   
    last = STEPS_TOTAL - 1

    st.caption(f"Total Steps: **{TOTAL_TIME/STEP_SEC:.0f}** (must < 1000)")
    st.divider()
    # Visibility toggles for plot elements
    st.checkbox("Show Point 1 (Blue)", key="SHOW_BLUE")
    st.checkbox("Show Point 2 (Green)", key="SHOW_GREEN")
    st.checkbox("Show Disk (Orange)", key="SHOW_ORANGE")
# -------------------------------------------------
# 3️⃣ Main UI: Detailed Sweep Parameters
# -------------------------------------------------

with st.container(border=True):
    col1, col2 = st.columns(2)
    
with col1:
    st.subheader(" Head Sweep (Wafer)")
    head_mode = st.radio("**Head Sweep Mode**", ["Sine", "Custom"], horizontal=True, key="head_sweep_mode_radio")
    st.session_state.head_mode = head_mode
    is_head_custom = (head_mode == "Custom")
    
   
    hc1, hc2 = st.columns(2)
    with hc1:
        st.session_state.SWEEP_START = st.number_input("**Head START (in)**", 7.0, 8.0, 7.2, 0.1, key="h_start_input")
    with hc2:
        st.session_state.SWEEP_END = st.number_input("**Head END (in)**", 8.0, 9.0, 8.2, 0.1, key="h_end_input")
    
    h_zone_num = st.number_input("**Head Sweep Zone Num**", 8, 13, 10, disabled=not is_head_custom, key="h_zone_num_input")

    h_near_mm = st.session_state.SWEEP_START * 25.4
    h_far_mm  = st.session_state.SWEEP_END * 25.4
    h_boundaries_mm = [h_near_mm + (h_far_mm - h_near_mm) * i / h_zone_num for i in range(h_zone_num + 1)]

    if not is_head_custom:
        h_mid = (h_near_mm + h_far_mm) / 2.0
        h_amp = (h_far_mm - h_near_mm) / 2.0
        h_rel_times, h_last_theta = [], math.pi
        for i in range(1, h_zone_num + 1):
            target_r = h_boundaries_mm[i]
            cos_val = max(-1.0, min(1.0, (target_r - h_mid) / h_amp))
            current_theta = math.acos(cos_val)
            h_rel_times.append(round(abs(current_theta - h_last_theta), 3))
            h_last_theta = current_theta
        
        h_data = {"Zone": [f"Zone {i+1}" for i in range(h_zone_num)], "Zone_Start": [round(h_boundaries_mm[i] / 25.4, 2) for i in range(h_zone_num)], "Zone_End": [round(h_boundaries_mm[i+1] / 25.4, 2) for i in range(h_zone_num)], "Relative_Time": h_rel_times}
        st.session_state.head_zone_df = pd.DataFrame(h_data)
    else:
        if "head_zone_df" not in st.session_state or len(st.session_state.head_zone_df) != h_zone_num:
            h_data = {"Zone": [f"Zone {i+1}" for i in range(h_zone_num)], "Zone_Start": [round(h_boundaries_mm[i] / 25.4, 2) for i in range(h_zone_num)], "Zone_End": [round(h_boundaries_mm[i+1] / 25.4, 2) for i in range(h_zone_num)], "Relative_Time": [1.0] * h_zone_num}
            st.session_state.head_zone_df = pd.DataFrame(h_data)

    

    st.session_state.head_zone_df = st.data_editor(st.session_state.head_zone_df, hide_index=True, disabled=not is_head_custom, 
                column_config={ "Zone": st.column_config.Column(disabled=True), 
                                "Zone_Start": st.column_config.Column(disabled=True), 
                                "Zone_End": st.column_config.Column(disabled=True),},
                key="head_zone_editor")
    st.session_state.SWPS_MIN = st.slider("**Head Swps per minute**", 1, 25, int(st.session_state.SWPS_MIN))
    
   
    h_half_cycle_time = (60.0 / st.session_state.SWPS_MIN) / 2.0
    h_total_rel_time = st.session_state.head_zone_df["Relative_Time"].sum()
    st.caption(f"📊 Head Sum of Rel Time: **{h_total_rel_time:.3f}**")
    st.caption(f"⏱️ Head Half Cycle: **{h_half_cycle_time:.2f} s**")

with col2:
    st.subheader(" Disk Sweep")
    disk_mode = st.radio("**Disk Sweep Mode**", ["Sine", "Custom"], horizontal=True, key="disk_sweep_mode_radio")
    st.session_state.disk_mode = disk_mode
    is_custom = (disk_mode == "Custom")
    

    c1, c2 = st.columns(2)
    with c1: st.session_state.DSWEEP_START = st.number_input("**Disk NEAR (END) (in)**", 2.0, 6.0, float(st.session_state.DSWEEP_START), 0.1, key="ds_start")
    with c2: st.session_state.DSWEEP_END = st.number_input("**Disk FAR (START) (in)**", 7.0, 15.0, float(st.session_state.DSWEEP_END), 0.1, key="ds_end")
    
    zone_num = st.number_input("**Disk Sweep Zone Num**", 10, 20, 13, disabled=not is_custom, key="zone_num_input")

    d_near_mm, d_far_mm = st.session_state.DSWEEP_START * 25.4, st.session_state.DSWEEP_END * 25.4
    zone_boundaries_mm = [d_far_mm - (d_far_mm - d_near_mm) * i / zone_num for i in range(zone_num + 1)]
    
    if not is_custom:
        d_mid, d_amp = (d_near_mm + d_far_mm) / 2.0, (d_far_mm - d_near_mm) / 2.0
        rel_times, last_theta = [], 0.0
        for i in range(1, zone_num + 1):
            cos_val = max(-1.0, min(1.0, (zone_boundaries_mm[i] - d_mid) / d_amp))
            current_theta = math.acos(cos_val)
            rel_times.append(round(current_theta - last_theta, 3))
            last_theta = current_theta
        st.session_state.zone_df = pd.DataFrame({"ID": [f"Zone {i+1}" for i in range(zone_num)], "Zone_Start": [round(zone_boundaries_mm[i] / 25.4, 2) for i in range(zone_num)], "Zone_End": [round(zone_boundaries_mm[i+1] / 25.4, 2) for i in range(zone_num)], "Relative_Time": rel_times})
    else:
        if "zone_df" not in st.session_state or len(st.session_state.zone_df) != zone_num:
            st.session_state.zone_df = pd.DataFrame({"Zone": [f"Zone {i+1}" for i in range(zone_num)], "Zone_Start": [round(zone_boundaries_mm[i] / 25.4, 2) for i in range(zone_num)], "Zone_End": [round(zone_boundaries_mm[i+1] / 25.4, 2) for i in range(zone_num)], "Relative_Time": [1.0] * zone_num})

    st.session_state.zone_df = st.data_editor(st.session_state.zone_df, hide_index=True, disabled=not is_custom, 
                    column_config={ "Zone": st.column_config.Column(disabled=True), 
                                    "Zone_Start": st.column_config.Column(disabled=True), 
                                    "Zone_End": st.column_config.Column(disabled=True),},
                    key="zone_data_editor_final")
    st.session_state.DSWPS_MIN = st.slider("Disk Swps per minute", 1, 20, int(st.session_state.DSWPS_MIN))
    
    d_half_cycle_time = (60.0 / st.session_state.DSWPS_MIN) / 2.0
    d_total_rel_time = st.session_state.zone_df["Relative_Time"].sum()
    st.caption(f"📊 Disk Sum of Rel Time: **{d_total_rel_time:.3f}**")
    st.caption(f"⏱️ Disk Half Cycle: **{d_half_cycle_time:.2f} s**")

if st.button("🚀 Update & Generate Plot", use_container_width=True):
    st.session_state.show_plot = True


# -------------------------------------------------
# 4️⃣  Derived constants
# -------------------------------------------------
STEP_SEC   = st.session_state.STEP_SEC
TOTAL_TIME = st.session_state.TOTAL_TIME
STEPS_TOTAL = int(TOTAL_TIME / STEP_SEC) + 1   # +1 so the final time is included
last = STEPS_TOTAL - 1

if STEPS_TOTAL > 1000:
    st.warning(
        "⚠️over 1000 steps --- "
        "please retry with SMALLER total_time or BIGGER second_per_step.")
    st.stop() 

c_blue = "rgba(0,0,255,1)" if st.session_state.SHOW_BLUE else "rgba(0,0,0,0)"
w_blue = 2 if st.session_state.SHOW_BLUE else 0
c_green = "rgba(0,255,0,1)" if st.session_state.SHOW_GREEN else "rgba(0,0,0,0)"
w_green = 2 if st.session_state.SHOW_GREEN else 0
c_orange = "rgba(255, 140, 0, 1)" if st.session_state.SHOW_ORANGE else "rgba(0,0,0,0)"
w_orange = 1 if st.session_state.SHOW_ORANGE else 0


# degrees added each step (RPM × 6°/s × STEP_SEC)
deg_per_step_platen = st.session_state.PLATEN_RPM * 6.0 * STEP_SEC
deg_per_step_pointa = st.session_state.POINTA_RPM * 6.0 * STEP_SEC


# pendulum frequency (swings per minute → cycles per step)
WAFER_PENDULUM_FREQ = st.session_state.SWPS_MIN * STEP_SEC / 30
DISK_PENDULUM_FREQ  = st.session_state.DSWPS_MIN * STEP_SEC / 60


# wafer‑center sweep range (inch → mm)
WAFER_NEAR = st.session_state.SWEEP_START * 2.54 * 10   # mm
WAFER_FAR  = st.session_state.SWEEP_END   * 2.54 * 10   # mm
WAFER_MID  = (WAFER_NEAR + WAFER_FAR) / 2.0
WAFER_AMP  = (WAFER_FAR - WAFER_NEAR) / 2.0

PLATEN_RADIUS = 390   
DISK_NEAR = st.session_state.DSWEEP_START * 25.4
DISK_FAR  = st.session_state.DSWEEP_END * 25.4
DISK_ARM_LENGTH = 610

current_zone_df = st.session_state.zone_df.copy()
current_zone_df["Zone_Start_mm"] = current_zone_df["Zone_Start"] * 25.4
current_zone_df["Zone_End_mm"]   = current_zone_df["Zone_End"] * 25.4

half_cycle_time = (60.0 / st.session_state.DSWPS_MIN) / 2.0
sum_relative_time = current_zone_df["Relative_Time"].sum()
zone_time_boundaries = []
acc = 0.0
for rt in current_zone_df["Relative_Time"]:
    zone_duration = (rt / sum_relative_time) * half_cycle_time
    acc += zone_duration
    zone_time_boundaries.append(acc)

# -------------------------------------------------
# 5️⃣  Containers for all points
# -------------------------------------------------
pa_x, pa_y   = [], []          # blue PointA
pa2_x, pa2_y = [], []          # green PointA2 (opposite side)
pl_x, pl_y   = [], []          # black Platen
wa_x_traj, wa_y_traj = [], []  # wafer‑center trajectory (list of floats)
platen_rev = []   # Platen revolutions (float, 1‑decimal)
wafer_rev  = []   # Wafer‑center revolutions (float, 1‑decimal)

# distance‑vs‑step data (distance from origin)
dist_a1, dist_a2, dist_wa, dist_d = [], [], [], []   # each list length = STEPS_TOTAL


k_x, k_y = [], []                    # ref K point for disk axis
disk_axis_x, disk_axis_y = [], []    # disk axis    
disk_n_x, disk_n_y= [], []
disk_f_x, disk_f_y= [], []      
disk_x, disk_y= [], []   

# -------------------------------------------------
# 6️⃣  Main loop – compute every step
# -------------------------------------------------

def calc_axis_rot(x, y, deg=45):
    rad = math.radians(deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    x_p = x * cos_a - y * sin_a
    y_p = x * sin_a + y * cos_a
    return x_p, y_p

def solve_joint(d_ori, d_J, Jx, Jy):
    d2 = Jx**2 + Jy**2 
    d = math.sqrt(d2)
    if d == 0: return None
    K = (d_ori**2 + d2 - d_J**2) / 2
    discriminant = (d_ori**2 * d2) - K**2
    if discriminant < 0: return None  
    sqrt_disc = math.sqrt(discriminant)
    xp = (K * Jx + Jy * sqrt_disc) / d2
    yp = (K * Jy - Jx * sqrt_disc) / d2
    return round(xp, 4), round(yp, 4)


K_INIT_X, K_INIT_Y = 450.0, -420.0
J_INIT_X, J_INIT_Y = calc_axis_rot(K_INIT_X, K_INIT_Y, 45)


half_cycle_time = (60.0 / st.session_state.DSWPS_MIN) / 2.0 
sum_relative_time = st.session_state.zone_df["Relative_Time"].sum()
total_rel_time = st.session_state.zone_df["Relative_Time"].sum()
h_half_cycle_time = (60.0 / st.session_state.SWPS_MIN) / 2.0
h_sum_rel_time = st.session_state.head_zone_df["Relative_Time"].sum()



zone_time_boundaries = []
current_time_acc = 0.0
for rt in st.session_state.zone_df["Relative_Time"]:
    zone_duration = (rt / sum_relative_time) * half_cycle_time
    current_time_acc += zone_duration
    zone_time_boundaries.append(current_time_acc)

head_time_boundaries = []
h_acc = 0.0
for rt in st.session_state.head_zone_df["Relative_Time"]:
    h_duration = (rt / h_sum_rel_time) * h_half_cycle_time
    h_acc += h_duration
    head_time_boundaries.append(h_acc)

for step in range(STEPS_TOTAL):

    t_elapsed = step * STEP_SEC
    
    
    ang_pl = 90 + step * deg_per_step_platen
    rad_pl = math.radians(ang_pl)
    pl_x.append(PLATEN_RADIUS * math.cos(rad_pl))
    pl_y.append(PLATEN_RADIUS * math.sin(rad_pl))

   
    h_t_cycle = (60.0 / st.session_state.SWPS_MIN)
    h_half_cycle = h_t_cycle / 2.0
    h_t_in_cycle = t_elapsed % h_t_cycle

    if h_t_in_cycle <= h_half_cycle:
        h_t_lookup = h_t_in_cycle
    else:
        h_t_lookup = h_t_cycle - h_t_in_cycle

    H_NEAR_MM = st.session_state.SWEEP_START * 25.4
    H_FAR_MM  = st.session_state.SWEEP_END * 25.4
    h_mid_mm = (H_NEAR_MM + H_FAR_MM) / 2.0
    h_amp_mm = (H_FAR_MM - H_NEAR_MM) / 2.0

    if st.session_state.head_mode == "Sine":
        
        phi_w = 2.0 * math.pi * (st.session_state.SWPS_MIN / 60.0) * t_elapsed
        r_t = h_mid_mm - h_amp_mm * math.cos(phi_w)
    else:
       
        r_t = H_NEAR_MM
        for i in range(len(head_time_boundaries)):
            h_s_t = head_time_boundaries[i-1] if i > 0 else 0
            h_e_t = head_time_boundaries[i]
            if h_t_lookup <= h_e_t:
                hz_start_mm = st.session_state.head_zone_df.iloc[i]["Zone_Start"] * 25.4
                hz_end_mm   = st.session_state.head_zone_df.iloc[i]["Zone_End"] * 25.4
                h_ratio = (h_t_lookup - h_s_t) / (h_e_t - h_s_t) if (h_e_t - h_s_t) > 0 else 0
                r_t = hz_start_mm + (hz_end_mm - hz_start_mm) * h_ratio
                break
        else:
            r_t = H_FAR_MM

    
    d_t_cycle = (60.0 / st.session_state.DSWPS_MIN) 
    d_half_cycle = d_t_cycle / 2.0
    d_t_in_cycle = t_elapsed % d_t_cycle

    if d_t_in_cycle <= d_half_cycle:
        d_t_lookup = d_t_in_cycle
    else:
        d_t_lookup = d_t_cycle - d_t_in_cycle

    DISK_NEAR_MM = st.session_state.DSWEEP_START * 25.4
    DISK_FAR_MM  = st.session_state.DSWEEP_END * 25.4
    d_mid = (DISK_NEAR_MM + DISK_FAR_MM) / 2.0
    d_amp = (DISK_FAR_MM - DISK_NEAR_MM) / 2.0

    if st.session_state.disk_mode == "Sine":
        phi_d = 2.0 * math.pi * (st.session_state.DSWPS_MIN / 60.0) * t_elapsed
        current_radius_d = d_mid + d_amp * math.cos(phi_d)
    else:
        current_radius_d = DISK_FAR_MM
        for i in range(len(zone_time_boundaries)):
            start_t = zone_time_boundaries[i-1] if i > 0 else 0
            end_t = zone_time_boundaries[i]
            if d_t_lookup <= end_t:
                z_start_mm = st.session_state.zone_df.iloc[i]["Zone_Start"] * 25.4
                z_end_mm   = st.session_state.zone_df.iloc[i]["Zone_End"] * 25.4
                ratio = (d_t_lookup - start_t) / (end_t - start_t) if (end_t - start_t) > 0 else 0
                current_radius_d = z_start_mm - (z_start_mm - z_end_mm) * ratio
                break
        else:
            current_radius_d = DISK_NEAR_MM

    
    d_raw = solve_joint(current_radius_d, DISK_ARM_LENGTH, J_INIT_X, J_INIT_Y)
    dxt_raw, dyt_raw = d_raw if d_raw else (0, 0)
    
   
    rot_angle = math.radians(-step * deg_per_step_platen)
    def rotate_xy(x, y, rad):
        return x * math.cos(rad) - y * math.sin(rad), x * math.sin(rad) + y * math.cos(rad)

   
    wa_rad_pos = math.radians(90 - step * deg_per_step_platen)
    wx = r_t * math.cos(wa_rad_pos)
    wy = r_t * math.sin(wa_rad_pos)
    wa_x_traj.append(wx)
    wa_y_traj.append(wy)

   
    ang_a1_rel = math.radians(step * deg_per_step_pointa + 180) - rad_pl
    pa_x.append(wx + st.session_state.POINTA_RADIUS * math.cos(ang_a1_rel))
    pa_y.append(wy + st.session_state.POINTA_RADIUS * math.sin(ang_a1_rel))

    ang_a2_rel = ang_a1_rel + math.pi
    pa2_x.append(wx + st.session_state.POINTA_RADIUS * math.cos(ang_a2_rel))
    pa2_y.append(wy + st.session_state.POINTA_RADIUS * math.sin(ang_a2_rel))

    
    dx_curr, dy_curr = rotate_xy(dxt_raw, dyt_raw, rot_angle)
    disk_x.append(dx_curr); disk_y.append(dy_curr)
    
    k_curr_x, k_curr_y = rotate_xy(K_INIT_X, K_INIT_Y, rot_angle)
    k_x.append(k_curr_x); k_y.append(k_curr_y)
    
    j_curr_x, j_curr_y = rotate_xy(J_INIT_X, J_INIT_Y, rot_angle)
    disk_axis_x.append(j_curr_x); disk_axis_y.append(j_curr_y)

    
    platen_rev.append((step * deg_per_step_platen) / 360.0)
    wafer_rev.append((step * deg_per_step_pointa) / 360.0)
    dist_a1.append(math.hypot(pa_x[-1], pa_y[-1]))
    dist_a2.append(math.hypot(pa2_x[-1], pa2_y[-1]))
    dist_wa.append(r_t)
    dist_d.append(current_radius_d)

   
    dn_raw_x, dn_raw_y = solve_joint(DISK_NEAR_MM, DISK_ARM_LENGTH, J_INIT_X, J_INIT_Y)
    df_raw_x, df_raw_y = solve_joint(DISK_FAR_MM, DISK_ARM_LENGTH, J_INIT_X, J_INIT_Y)
    dn_curr_x, dn_curr_y = rotate_xy(dn_raw_x, dn_raw_y, rot_angle)
    df_curr_x, df_curr_y = rotate_xy(df_raw_x, df_raw_y, rot_angle)
    disk_n_x.append(dn_curr_x); disk_n_y.append(dn_curr_y)
    disk_f_x.append(df_curr_x); disk_f_y.append(df_curr_y)

st.markdown(
    """
    <style>
    .stApp { max-width: 100% !important; padding: 0 20px !important; }
    </style>
    """,
    unsafe_allow_html=True
)
# -------------------------------------------------
# 7️⃣ Plotly Figure Setup
# -------------------------------------------------
fig = make_subplots(rows=2, cols=1, vertical_spacing=0.1, row_heights=[0.9, 0.3])


fig.add_shape(
    type="circle",
    xref="x", yref="y",
    x0=-PLATEN_RADIUS, y0=-PLATEN_RADIUS,
    x1= PLATEN_RADIUS, y1= PLATEN_RADIUS,
    line_color="rgba(0,0,0,0.5)",
    fillcolor="rgba(0,0,0,0)",
)

fig.add_shape(
    type="circle",
    xref="x", yref="y",
    x0=-150, y0= WAFER_NEAR-150,
    x1= 150, y1= WAFER_NEAR+150,
    line_color="rgba(0,0,0,0.5)",
    fillcolor="rgba(0,0,0,0)",
)


fig.add_trace(go.Scatter(name="Point 1 (Blue) Traj", x=[], y=[], mode="lines", line=dict(color=c_blue, width=w_blue, shape="spline", smoothing=1.3)), row=1, col=1)
fig.add_trace(go.Scatter(name="Point 2 (Green) Traj", x=[], y=[], mode="lines", line=dict(color=c_green, width=w_green, shape="spline", smoothing=1.3)), row=1, col=1)
fig.add_trace(go.Scatter(name="Wafer Center Traj", x=[], y=[], mode="lines", line=dict(color="black", width=1, shape="spline", smoothing=1.3, dash="dot")), row=1, col=1)
fig.add_trace(go.Scatter(name="Disk Traj", x=[], y=[], mode="lines", line=dict(color=c_orange, width=w_orange, shape="spline", smoothing=1.3)), row=1, col=1)
fig.add_trace(go.Scatter(name="Platen Edge Traj", x=[], y=[], mode="lines", line=dict(color="rgba(0,0,0,0.3)", width=1, shape="spline", smoothing=1.3)), row=1, col=1)
fig.add_trace(go.Scatter(name="Head Arm", x=[], y=[], mode="lines", line=dict(color="orange", width=1, shape="spline", smoothing=1.3)), row=1, col=1)


fig.add_trace(go.Scatter(name="Current Positions", x=[], y=[], mode="markers", 
                         marker=dict(size=6, color=[c_blue, c_green, "black", c_orange])), row=1, col=1)


fig.add_trace(go.Scatter(name="Dist P1", x=[], y=[], mode="lines", line=dict(color="blue", shape="spline", smoothing=1.3)), row=2, col=1)
fig.add_trace(go.Scatter(name="Dist P2", x=[], y=[], mode="lines", line=dict(color="green", shape="spline", smoothing=1.3)), row=2, col=1)
fig.add_trace(go.Scatter(name="Dist Wafer", x=[], y=[], mode="lines", line=dict(color="black", dash="dot", shape="spline", smoothing=1.3)), row=2, col=1)
fig.add_trace(go.Scatter(name="Dist Disk", x=[], y=[], mode="lines", line=dict(color="orange", shape="spline", smoothing=1.3)), row=2, col=1)


# -------------------------------------------------
# 8️⃣  Create animation frames
# -------------------------------------------------
frames = []
for i in range(STEPS_TOTAL):
    times = [k * STEP_SEC for k in range(i + 1)]
    frames.append(go.Frame(
        name=str(i),
        data=[
            go.Scatter(x=pa_x[:i+1], y=pa_y[:i+1]), # 0: Blue
            go.Scatter(x=pa2_x[:i+1], y=pa2_y[:i+1]), # 1: Green
            go.Scatter(x=wa_x_traj[:i+1], y=wa_y_traj[:i+1]), # 2: Wafer
            go.Scatter(x=disk_x[:i+1], y=disk_y[:i+1]), # 3: Disk
            go.Scatter(x=pl_x[:i+1], y=pl_y[:i+1]), # 4: Platen
            go.Scatter(x=[pa2_x[i], pa_x[i]], y=[pa2_y[i], pa_y[i]]), # 5: Arm
            go.Scatter(x=[pa_x[i], pa2_x[i], wa_x_traj[i], disk_x[i]], 
                       y=[pa_y[i], pa2_y[i], wa_y_traj[i], disk_y[i]]), # 6: Current
            go.Scatter(x=times, y=dist_a1[:i+1]), # 7: Dist
            go.Scatter(x=times, y=dist_a2[:i+1]), # 8
            go.Scatter(x=times, y=dist_wa[:i+1]), # 9
            go.Scatter(x=times, y=dist_d[:i+1]),  # 10
        ],
        layout=go.Layout(annotations=[dict(x=0.02, y=0.98, xref="paper", yref="paper", 
            text=f"Platen Rev: {platen_rev[i]:.1f} | Wafer Rev: {wafer_rev[i]:.1f}", showarrow=False,                        font=dict(size=12, color="black"),
                        bgcolor="rgba(255,255,255,0.7)",
                        bordercolor="black", borderwidth=1, align="left")])
    ))

fig.frames = frames

# -------------------------------------------------
# 9️⃣  Slider & Play / Pause controls
# -------------------------------------------------
sliders = [{
    "active": last,
    "currentvalue": {"prefix": "sec: "},
    "pad": {"t": 50},
    "steps": [{
        "method": "animate",
        "args": [[str(k)], {"mode": "immediate",
                           "frame": {"duration": 0},
                           "transition": {"duration": 0}}],
         "label": f"{k * STEP_SEC:.2f}"
    } for k in range(STEPS_TOTAL)]
}]

updatemenus = [{
    "type": "buttons", "direction": "left",
    "buttons": [
        {"label": "▶ Play", "method": "animate",
         "args": [None,
                  {"frame": {"duration": 80, "redraw": True},
                   "fromcurrent": True,
                   "transition": {"duration": 0},
                   "mode": "immediate"}]},
        {"label": "❚❚ Pause", "method": "animate",
         "args": [[None], {"mode": "immediate"}]}
    ],
    "pad": {"r": 10, "t": 10},
    "showactive": False,
    "x": 0.1, "y": 0,
    "xanchor": "right", "yanchor": "top"
}]

# -------------------------------------------------
# 10️⃣ Layout – axes ranges for both sub‑plots
# -------------------------------------------------
max_dist = max(max(dist_a1), max(dist_a2), max(dist_wa), max(dist_d)) * 1.1   # 10% margin

fig.update_layout(
    height=800,                 
    sliders=sliders,
    updatemenus=updatemenus,   
   
   
    xaxis=dict(
        range=[-PLATEN_RADIUS - 35, PLATEN_RADIUS + 35],
        zeroline=False, showgrid=True, scaleanchor="y", title="X (mm)"
    ),
    yaxis=dict(
        range=[-PLATEN_RADIUS - 35, PLATEN_RADIUS + 35],
        showgrid=True, title="Y (mm)"
    ),
    
    xaxis2=dict(
        range=[0, (STEPS_TOTAL*STEP_SEC)],
        title="Sec"
    ),
    yaxis2=dict(
        range=[0, max_dist],
        title="Distance(mm)"
    ),
    margin=dict(l=40, r=40, t=60, b=40),
)


# -------------------------------------------------
# 🔟  Display the chart
# -------------------------------------------------

st.plotly_chart(fig, width='content')
