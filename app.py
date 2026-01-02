import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots 
import math
import pandas as pd

# -------------------------------------------------
# 1️⃣  Default session‑state values (run only once)
# ------------------------------------------------- 
if "STEP_SEC" not in st.session_state:
    st.session_state.STEP_SEC = 0.02          # seconds per step
if "TOTAL_TIME" not in st.session_state:
    st.session_state.TOTAL_TIME = 10         # total simulation time (seconds)
if "PLATEN_RPM" not in st.session_state:
    st.session_state.PLATEN_RPM = 86
if "POINTA_RPM" not in st.session_state:
    st.session_state.POINTA_RPM = 92
if "POINTA_RADIUS" not in st.session_state:
    st.session_state.POINTA_RADIUS = 100

# pendulum‑related defaults
if "SWEEP_START" not in st.session_state:
    st.session_state.SWEEP_START = 7.2      # inch
if "SWEEP_END" not in st.session_state:
    st.session_state.SWEEP_END   = 8.2      # inch
if "SWPS_MIN" not in st.session_state:
    st.session_state.SWPS_MIN    = 6
if "PEND_MODE" not in st.session_state:
    st.session_state.PEND_MODE   = "Sine"   # or "Custom"

#if "DSWPS_MIN" not in st.session_state:
#    st.session_state.DSWPS_MIN    = 10
st.session_state.DSWPS_MIN    = 5


# -------------------------------------------------
# 2️⃣  Sidebar input form
# -------------------------------------------------
st.sidebar.header("Setting")
with st.sidebar.form(key="param_form"):
    st.session_state.PLATEN_RPM = st.number_input(
        "Platen RPM", min_value=0, max_value=200, value=st.session_state.PLATEN_RPM, step=1
    )
    st.session_state.POINTA_RPM = st.number_input(
        "Head RPM", min_value=0, max_value=200, value=st.session_state.POINTA_RPM, step=1
    )
    st.session_state.POINTA_RADIUS = st.number_input(
        "PointA Radius (mm)", min_value=1, max_value=150, value=st.session_state.POINTA_RADIUS, step=1
    )
    st.session_state.STEP_SEC = st.slider(
        "second per STEP", min_value=0.01, max_value=0.1,
        value=st.session_state.STEP_SEC, step=0.01, format="%.2f"
    )
    st.session_state.TOTAL_TIME = st.slider(
        "TOTAL_TIME (s)", min_value=1, max_value=99,
        value=int(st.session_state.TOTAL_TIME), step=1
    )
    st.session_state.SWEEP_START = st.slider(
        "Sweep START (inch)", min_value=6.0, max_value=7.3,
        value=float(st.session_state.SWEEP_START), step=0.1, format="%.2f"
    )
    st.session_state.SWEEP_END = st.slider(
        "Sweep END (inch)", min_value=8.0, max_value=8.4,
        value=float(st.session_state.SWEEP_END), step=0.1, format="%.2f"
    )
    st.session_state.SWPS_MIN = st.slider(
        "Swps per minute", min_value=0, max_value=20,
        value=int(st.session_state.SWPS_MIN), step=1
    )
    st.session_state.PEND_MODE = st.selectbox(
        "Sweep mode", options=["Sine", "Custom(1)"],
        index=0 if st.session_state.PEND_MODE == "Sine" else 1
    )
    st.session_state.SHOW_GREEN = st.checkbox(
    "Show point 2",
    value=True,
    key="show_green_checkbox"
    )
    submitted = st.form_submit_button("Update")
    if submitted:
        st.session_state.show_plot = True   # flag to draw the figure

# -------------------------------------------------
# 3️⃣  Placeholder / info (before Update)
# -------------------------------------------------
if not st.session_state.get("show_plot", False):
    st.info(
        "Set the parameters on the left sidebar and click **Update** to generate the plot.\n"
        "⚠️ If second_per_STEP is very small or TOTAL_TIME is large, many points will be created "
        "and rendering may take several mins."
    )
    st.stop()   # stop execution until the user clicks Update

# -------------------------------------------------
# 4️⃣  Derived constants
# -------------------------------------------------
STEP_SEC   = st.session_state.STEP_SEC
TOTAL_TIME = st.session_state.TOTAL_TIME
STEPS_TOTAL = int(TOTAL_TIME / STEP_SEC) + 1   # +1 so the final time is included

if STEPS_TOTAL > 1000:
    st.warning(
        "⚠️over 1000 steps --- "
        "please retry with SMALLER total_time or BIGGER second_per_step.")
    st.stop() 

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
DISK_NEAR = 2.7 * 25.4   # 68.58 mm END
DISK_FAR = 14.7 * 25.4  # 373.38 mm START
DISK_ARM_LENGTH = 610

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
    d = math.sqrt(d2)   # J to (0,0) distance 
    if d == 0:
        return None
    K = (d_ori**2 + d2 - d_J**2) / 2        # constK = (r^2 + d^2 - R^2) / 2
    discriminant = (d_ori**2 * d2) - K**2   # check cross
    if discriminant < 0:                    # not cross
        return None  
    sqrt_disc = math.sqrt(discriminant)
    xp = (K * Jx + Jy * sqrt_disc) / d2
    yp = (K * Jy - Jx * sqrt_disc) / d2
    return round(xp, 4), round(yp, 4)


K_INIT_X, K_INIT_Y = 450.0, -420.0          # K, ref point
J_INIT_X, J_INIT_Y = calc_axis_rot(K_INIT_X, K_INIT_Y, 45) # J, disk arm axis 
D_ININ_X, D_ININ_Y = solve_joint(DISK_NEAR, DISK_ARM_LENGTH, J_INIT_X, J_INIT_Y)
D_INIF_X, D_INIF_Y = solve_joint(DISK_FAR, DISK_ARM_LENGTH, J_INIT_X, J_INIT_Y)



for step in range(STEPS_TOTAL):
    # ---- Platen (black) ----
    ang_pl = 90 + step * deg_per_step_platen
    rad_pl = math.radians(ang_pl)
    pl_x.append(PLATEN_RADIUS * math.cos(rad_pl))
    pl_y.append(PLATEN_RADIUS * math.sin(rad_pl))
    
    # ---- common angular part (used for both pendulum modes) ----
    phi_w = 2.0 * math.pi * WAFER_PENDULUM_FREQ * step   # phase
    # ---- radius_w according to selected mode ----
    if st.session_state.PEND_MODE == "Sine":
        # sinusoidal motion between NEAR and FAR
        r_t = WAFER_MID - WAFER_AMP * math.cos(phi_w)
    else:   # Custom – linear back‑and‑forth (triangular wave)
        # normalise step into [0, 2) where 0‑1 = forward, 1‑2 = backward
        period_norm = (step * WAFER_PENDULUM_FREQ) % 2.0
        if period_norm <= 1.0:          # forward
            r_t = WAFER_NEAR + (WAFER_FAR - WAFER_NEAR) * period_norm
        else:                           # backward
            r_t = WAFER_FAR - (WAFER_FAR - WAFER_NEAR) * (period_norm - 1.0)     
    # ---- radius_d according to selected mode ----
   

    # Disk traj. cal.
    # atan2(y, x), range: -pi ~ pi
    angle_N = math.atan2(D_ININ_Y - J_INIT_Y, D_ININ_X - J_INIT_X)
    angle_F = math.atan2(D_INIF_Y - J_INIT_Y, D_INIF_X - J_INIT_X)
    # filter, make sure short acr path
    diff = angle_F - angle_N
    if diff > math.pi:
        angle_F -= 2 * math.pi
    elif diff < -math.pi:
        angle_F += 2 * math.pi

    angle_mid = (angle_N + angle_F) / 2
    amplitude = (angle_F - angle_N) / 2

    phi_d = 2.0 * math.pi * DISK_PENDULUM_FREQ * step + math.pi
    rad_di = angle_mid - amplitude * math.cos(phi_d)
    dxt = J_INIT_X + DISK_ARM_LENGTH * math.cos(rad_di)
    dyt = J_INIT_Y + DISK_ARM_LENGTH * math.sin(rad_di)

    # ---- rotate the whole centre clockwise (same as before) ----
    rad_wa = math.radians( 90 - step * deg_per_step_platen)   # clockwise
    wa_x = r_t * math.cos(rad_wa)      # current centre x (float)
    wa_y = r_t * math.sin(rad_wa)      # current centre y (float)

    # store centre for later drawing
    wa_x_traj.append(wa_x)
    wa_y_traj.append(wa_y)

    # ---- PointA (blue) ----
    ang_a1 = step * deg_per_step_pointa + 180 
    rad_a1 = math.radians(ang_a1)
    pa_x.append(wa_x + st.session_state.POINTA_RADIUS * math.cos(rad_a1-rad_pl))
    pa_y.append(wa_y + st.session_state.POINTA_RADIUS * math.sin(rad_a1-rad_pl))

    # ---- PointA2 (green, opposite side) ----
    ang_a2 = ang_a1 + 180.0
    rad_a2 = math.radians(ang_a2)
    pa2_x.append(wa_x + st.session_state.POINTA_RADIUS * math.cos(rad_a2-rad_pl))
    pa2_y.append(wa_y + st.session_state.POINTA_RADIUS * math.sin(rad_a2-rad_pl))

    # ---- K、J 點的旋轉 (與 Platen 同 RPM、順時針) ----
    #   Platen 逆時針 +deg_per_step_platen，故順時針需取負號
    rot_angle = math.radians( -step * deg_per_step_platen)   # clockwise
    k_x.append(K_INIT_X * math.cos(rot_angle) - K_INIT_Y * math.sin(rot_angle))
    k_y.append(K_INIT_X * math.sin(rot_angle) + K_INIT_Y * math.cos(rot_angle))
    disk_axis_x.append(J_INIT_X * math.cos(rot_angle) - J_INIT_Y * math.sin(rot_angle))
    disk_axis_y.append(J_INIT_X * math.sin(rot_angle) + J_INIT_Y * math.cos(rot_angle))
    disk_n_x.append(D_ININ_X * math.cos(rot_angle) - D_ININ_Y * math.sin(rot_angle))
    disk_n_y.append(D_ININ_X * math.sin(rot_angle) + D_ININ_Y * math.cos(rot_angle))
    disk_f_x.append(D_INIF_X * math.cos(rot_angle) - D_INIF_Y * math.sin(rot_angle))
    disk_f_y.append(D_INIF_X * math.sin(rot_angle) + D_INIF_Y * math.cos(rot_angle))
    disk_x.append( dxt * math.cos(rot_angle) - dyt * math.sin(rot_angle))
    disk_y.append( dxt * math.sin(rot_angle) + dyt * math.cos(rot_angle))   


    # Platen and Wafer rotate with the same angular speed (deg_per_step_platen)
    revo_platen = (step * deg_per_step_platen) / 360.0   # float
    revo_pointa = (step * deg_per_step_pointa) / 360.0   # float
    platen_rev.append(revo_platen)          # for the Platen
    wafer_rev.append(revo_pointa)           # for the Wafer‑center
    
    # ---- distance from origin (for the XY‑distance plot) ----
    dist_a1.append(math.hypot(pa_x[-1], pa_y[-1]))
    dist_a2.append(math.hypot(pa2_x[-1], pa2_y[-1]))
    dist_wa.append(math.hypot(wa_x, wa_y))
    dist_d.append(math.hypot(disk_x[-1], disk_y[-1]))


    show_green = st.session_state.SHOW_GREEN          # read once
    green_color = "rgba(0,255,0,1)" if show_green else "rgba(0,0,0,0)"
    green_width = 1 if show_green else 0

# -------------------------------------------------
# 7️⃣  Build Plotly figure
# -------------------------------------------------
fig = make_subplots(
    rows=2,
    cols=1,
    vertical_spacing= 0.1 ,
    row_heights=[0.9, 0.3],
    specs=[
        [{'type': 'scatter'}],   
        [{'type': 'scatter'}]  
    ]
)

# background reference circle
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

# ---- initial points (show last step so slider starts at the rightmost position) ----
last = STEPS_TOTAL - 1
fig.add_trace(
    go.Scatter(
        name="Points",
        x=[pa_x[last], pl_x[last], wa_x_traj[last], pa2_x[last], disk_x[last]],
        y=[pa_y[last], pl_y[last], wa_y_traj[last], pa2_y[last], disk_y[last]],
        mode="markers",
        marker=dict(
            size=[5, 5, 5, 5, 5],
            color=["blue", "black", "black", green_color, "black"],
        ),
    ),row=1, col=1
)

# Empty traces that will be filled frame‑by‑frame (prevents sending all data at once)
fig.add_trace(go.Scatter(name="A1 traj", x=[], y=[], mode="lines",
                         line=dict(color="blue", width=1)))
fig.add_trace(go.Scatter(name="Platen traj", x=[], y=[], mode="lines",
                         line=dict(color="black", width=1)))
fig.add_trace(go.Scatter(name="Wafer traj", x=[], y=[], mode="lines",
                         line=dict(color="black", width=1, dash="dot")))
fig.add_trace(go.Scatter(name="A2 traj", x=[], y=[], mode="lines",
                         line=dict(color=green_color, width=green_width)))

# orange line that connects Green → Black → Blue (three points)
fig.add_trace(go.Scatter(name="A1‑A2", x=[], y=[], mode="lines",
                         line=dict(color="orange", width=1)))
# ---------- K、J、DN、DF, disk  trace (last) ----------
fig.add_trace(go.Scatter(name="K point", x=[k_x[last]],  y=[k_y[last]],
        mode="markers", marker=dict(size=5, color="magenta")
        ),row=1, col=1)
fig.add_trace(go.Scatter(name="J point", x=[disk_axis_x[last]], y=[disk_axis_y[last]],
        mode="markers", marker=dict(size=5, color="cyan")
        ),row=1, col=1)
fig.add_trace(go.Scatter(name="DN point", x=[disk_n_x[last]],   y=[disk_n_y[last]],
        mode="markers", marker=dict(size=5, color="orange")
        ),row=1, col=1)
fig.add_trace(go.Scatter(name="DF point",  x=[disk_f_x[last]],  y=[disk_f_y[last]],
        mode="markers", marker=dict(size=5, color="red")
        ),row=1, col=1)
fig.add_trace(go.Scatter(name="D point",   x=[disk_x[last]],    y=[disk_y[last]],
        mode="markers", marker=dict(size=5, color= "Yellow")
        ),row=1, col=1)


#--------------------below
fig.add_trace(go.Scatter(name="Dist A1", x=[], y=[], mode="lines+markers",
                         line=dict(color="blue", width=2), marker=dict(size=4)),
               row=2, col=1)
fig.add_trace(go.Scatter(name="Dist A2", x=[], y=[], mode="lines+markers",
                         line=dict(color="green", width=2), marker=dict(size=4)),
               row=2, col=1)
fig.add_trace(go.Scatter(name="Dist Wafer", x=[], y=[], mode="lines+markers",
                         line=dict(color="black", width=2, dash="dot"),
                         marker=dict(size=4)),
               row=2, col=1)
fig.add_trace(go.Scatter(name="Dist Disk", x=[], y=[], mode="lines+markers",
                         line=dict(color="Orange", width=2, dash="dot"),
                         marker=dict(size=4)),
               row=2, col=1)


# -------------------------------------------------
# 8️⃣  Create animation frames
# -------------------------------------------------
frames = []
for i in range(STEPS_TOTAL):
    times = [k * STEP_SEC for k in range(i + 1)]
    frames.append(
        go.Frame(
            name=str(i),
            data=[
                # 0 points (blue, black, black‑wafer, green)
                go.Scatter(
                    x=[pa_x[i], pl_x[i], wa_x_traj[i], pa2_x[i],  disk_x[i]],
                    y=[pa_y[i], pl_y[i], wa_y_traj[i], pa2_y[i],  disk_y[i]],
                    mode="markers",
                    marker=dict(
                        size=[5, 5, 5, 5, 5],
                        color=["blue", "black", "black", green_color, "black"],
                    ),
                ),
                # 1~4 trajectories up to current step
                go.Scatter(x=pa_x[: i + 1], y=pa_y[: i + 1], # A1
                           mode="lines", line=dict(color="rgba(0,0,255,1)", width=2, shape="spline", smoothing=1.3)),
                go.Scatter(x=pl_x[: i + 1], y=pl_y[: i + 1], # Platen
                           mode="lines", line=dict(color="rgba(0,0,0,1)", width=1, shape="spline", smoothing=1.3)),
                go.Scatter(x=wa_x_traj[: i + 1], y=wa_y_traj[: i + 1], # wafer center
                           mode="lines", line=dict(color="rgba(0,0,0,1)", width=1, shape="spline", smoothing=1.3, dash="dot")),
                go.Scatter(x=pa2_x[: i + 1], y=pa2_y[: i + 1], #A2
                           mode="lines", line=dict(color=green_color, width=green_width, shape="spline", smoothing=1.3)),

                # 5 orange line: Green  → Blue (three points)
                go.Scatter(x=[pa2_x[i], pa_x[i]], y=[pa2_y[i], pa_y[i]], 
                           mode="lines", line=dict(color="rgba(255, 140, 0, 1)", width=1, shape="spline", smoothing=1.3),),
                
                # 6. K 
                go.Scatter(
                    x=[k_x[i]],
                    y=[k_y[i]],
                    mode="markers",
                    marker=dict(size=5, color="rgba(255, 0, 255, 0)"), #"magenta"
                ),
                # 7. J 
                go.Scatter(
                    x=[disk_axis_x[i]],
                    y=[disk_axis_y[i]],
                    mode="markers",
                    marker=dict(size=5, color="rgba(0, 230, 230, 0)"),  #"cyan"
                ),
                               
                # 8. DN 點
                go.Scatter(
                    x=[disk_n_x[i]],
                    y=[disk_n_y[i]],
                    mode="markers",
                    marker=dict(size=5, color="rgba(255, 140, 0, 0)"), #"orange"
                ),
                # 9. DF 點
                go.Scatter(
                    x=[disk_f_x[i]],
                    y=[disk_f_y[i]],
                    mode="markers",
                    marker=dict(size=5, color="rgba(255, 0, 0, 0)"), #"red"
                ),
                # 10 disk
                go.Scatter(x=disk_x[: i + 1], y=disk_y[: i + 1],
                           mode="lines", line=dict(color="orange", width=1, shape="spline", smoothing=1.3, dash="dot")
                ),
                
                
                # 11 Dist A1
                go.Scatter(
                    x=times,
                    y=dist_a1[: i + 1],
                    mode="lines+markers",
                    line=dict(color="blue", width=1, shape="spline", smoothing=1.3),
                    marker=dict(size=4),
                ),

                #---------below
                # 12 Dist A2
                go.Scatter(
                    x=times,
                    y=dist_a2[: i + 1],
                    mode="lines+markers",
                    line=dict(color="green", width=1, shape="spline", smoothing=1.3),
                    marker=dict(size=4),
                ),
                # 13 Dist Wafer
                go.Scatter(
                    x=times,
                    y=dist_wa[: i + 1],
                    mode="lines+markers",
                    line=dict(color="black", width=1, shape="spline", smoothing=1.3, dash="dot"),
                    marker=dict(size=4),
                ),
                # 14 Dist Wafer
                go.Scatter(
                    x=times,
                    y=dist_d[: i + 1],
                    mode="lines+markers",
                    line=dict(color="Orange", width=1, shape="spline", smoothing=1.3, dash="dot"),
                    marker=dict(size=4),
                ),
            ],
          
            layout=go.Layout(
                annotations=[
                    dict(
                        x=0.02, y=0.95,
                        xref="paper", yref="paper",
                        text=(
                            f"Platen ⟳: {platen_rev[i]:.1f}  |  "
                            f"Wafer ⟳: {wafer_rev[i]:.1f}"
                        ),
                        showarrow=False,
                        font=dict(size=12, color="black"),
                        bgcolor="rgba(255,255,255,0.7)",
                        bordercolor="black",
                        borderwidth=1,
                        align="left",
                    )
                ]
            ),
            traces=list(range(15)),  
        )
    )
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

st.plotly_chart(fig, use_container_width=True)





