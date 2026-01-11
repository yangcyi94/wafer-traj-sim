#last update: 260111

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, Tuple

# --- 1. 高性能物理運算引擎 ---
class CMPPhysicsEngine:
    @staticmethod
    def solve_joint_vectorized(d_ori: np.ndarray, d_J: float, Jx: float, Jy: float) -> Tuple[np.ndarray, np.ndarray]:
        """精確還原原本的連桿求解邏輯 (機械逆解)"""
        d2 = Jx**2 + Jy**2
        d = np.sqrt(d2)
        K = (d_ori**2 + d2 - d_J**2) / 2.0
        # 判別式
        disc = (d_ori**2 * d2) - K**2
        disc = np.maximum(disc, 0) # 避免物理不可達區域產生虛數
        sqrt_disc = np.sqrt(disc)
        
        # 原本代碼邏輯: xp = (K*Jx + Jy*sqrt_disc)/d2; yp = (K*Jy - Jx*sqrt_disc)/d2
        xp = (K * Jx + Jy * sqrt_disc) / d2
        yp = (K * Jy - Jx * sqrt_disc) / d2
        return xp, yp

    @staticmethod
    def get_sweep_radius(mode: str, t: np.ndarray, swps_min: float, df: pd.DataFrame) -> np.ndarray:
        """根據表格定義計算擺動半徑"""
        half_cycle = (60.0 / swps_min) / 2.0
        t_cycle = half_cycle * 2
        t_mod = t % t_cycle
        t_lookup = np.where(t_mod <= half_cycle, t_mod, t_cycle - t_mod)

        rel_times = df["Relative_Time"].values
        cum_rel = np.cumsum(rel_times)
        xp = np.concatenate(([0], (cum_rel / cum_rel[-1]) * half_cycle))
        fp = np.concatenate(([df["Zone_Start"].iloc[0] * 25.4], df["Zone_End"].values * 25.4))
        return np.interp(t_lookup, xp, fp)

# --- 2. Streamlit UI 與 參數管理 ---
st.set_page_config(layout="wide", page_title="CMP Trajectory Simulator Pro")
st.markdown("<style>.stApp { max-width: 100% !important; padding: 0 20px !important; }</style>", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Setting")
    PLATEN_RPM = st.number_input("Platen RPM", 0, 200, 86)
    HEAD_RPM = st.number_input("Head RPM", 0, 200, 92)
    STEP_SEC = st.slider("Second per STEP", 0.01, 0.1, 0.05)
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
    h_start = hc1.number_input("Head START (in)", 7.0, 9.0, 7.2, 0.1)
    h_end = hc2.number_input("Head END (in)", 7.0, 9.0, 8.2, 0.1)
    h_zn = st.number_input("Head Sweep Zone Num", 5, 20, 10)
    
    # 建立 Sine 分佈 Relative Time
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
    d_start = dc1.number_input("Disk NEAR (END) (in)", 2.0, 15.0, 2.7, 0.1)
    d_end = dc2.number_input("Disk FAR (START) (in)", 2.0, 15.0, 14.7, 0.1)
    d_zn = st.number_input("Disk Sweep Zone Num", 5, 20, 13)

    # Disk 邏輯恢復: 從 Far 到 Near
    d_bounds = np.linspace(d_end, d_start, d_zn + 1) # Far to Near
    d_rel = [1.0] * d_zn
    if d_mode == "Sine":
        mid, amp = (d_start + d_end) / 2.0, (d_end - d_start) / 2.0
        d_theta = np.arccos(np.clip((d_bounds - mid) / amp, -1, 1))
        d_rel = np.abs(np.diff(d_theta)).round(3)

    d_df = pd.DataFrame({"Zone": [f"Z{i+1}" for i in range(d_zn)], "Zone_Start": d_bounds[:-1].round(2), "Zone_End": d_bounds[1:].round(2), "Relative_Time": d_rel})
    d_editor_df = st.data_editor(d_df, hide_index=True, disabled=["Zone", "Zone_Start", "Zone_End"] if d_mode=="Custom" else True, key="d_edit")
    d_swps = st.slider("Disk Swps per minute", 1, 30, 13)

# --- 3. 核心運算主迴圈 ---
if st.button("🚀 Update & Generate Plot", use_container_width=True):
    t = np.arange(0, TOTAL_TIME + STEP_SEC, STEP_SEC)
    
    # 1. 旋轉參數與角度 (還原原本物理相位)
    platen_ang_deg = (PLATEN_RPM * 6.0) * t
    platen_rad = np.radians(90 - platen_ang_deg)
    
    # 2. 半徑 Profile
    r_h = CMPPhysicsEngine.get_sweep_radius(h_mode, t, h_swps, h_editor_df)
    r_d = CMPPhysicsEngine.get_sweep_radius(d_mode, t, d_swps, d_editor_df)
    
    # 3. Wafer / Head 軌跡
    wx, wy = r_h * np.cos(platen_rad), r_h * np.sin(platen_rad)
    
    # 還原 Head 相對 Platen 的相位修正: step * deg_pointa + 180 - platen_rad
    head_rel_rad = np.radians((HEAD_RPM * 6.0) * t + 180) + platen_rad
    p1x = wx + POINTA_RADIUS * np.cos(head_rel_rad + np.pi)
    p1y = wy + POINTA_RADIUS * np.sin(head_rel_rad + np.pi)
    p2x = wx + POINTA_RADIUS * np.cos(head_rel_rad)
    p2y = wy + POINTA_RADIUS * np.sin(head_rad_rel := head_rel_rad) # temp reuse
    p2y = wy + POINTA_RADIUS * np.sin(head_rel_rad)
    

    # 4. Disk 連桿與相位旋轉 (還原原本 J_INIT 與 旋轉邏輯)
    K_INIT = np.array([450.0, -420.0])
    # 預先計算 45 度旋轉後的 J_INIT (還原 solve_joint 之前的位置)
    cos45, sin45 = np.cos(np.radians(45)), np.sin(np.radians(45))
    J_INIT_X = K_INIT[0] * cos45 - K_INIT[1] * sin45
    J_INIT_Y = K_INIT[0] * sin45 + K_INIT[1] * cos45
    
    # 機械解 (raw_dx, raw_dy 是在靜止參考系)
    raw_dx, raw_dy = CMPPhysicsEngine.solve_joint_vectorized(r_d, 610.0, J_INIT_X, J_INIT_Y)
    
    # 將 Disk 座標轉到旋轉的 Platen 參考系: rotate by -platen_angle
    rot_inv_rad = np.radians(-platen_ang_deg)
    dx = raw_dx * np.cos(rot_inv_rad) - raw_dy * np.sin(rot_inv_rad)
    dy = raw_dx * np.sin(rot_inv_rad) + raw_dy * np.cos(rot_inv_rad)

    # 5. Rev 數據
    rev_p = (PLATEN_RPM * 6.0 * t) / 360.0
    rev_w = (HEAD_RPM * 6.0 * t) / 360.0

    # --- 4. Plotly 動畫配置 ---
    fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], vertical_spacing=0.1)
    
    # 基礎圖層
    style = dict(width=1.5, shape='spline', smoothing=1.3)
    fig.add_trace(go.Scatter(x=p1x, y=p1y, name="P1 (Blue)", line=dict(color='blue', **style), opacity=1 if SHOW_BLUE else 0), row=1, col=1)
    fig.add_trace(go.Scatter(x=p2x, y=p2y, name="P2 (Green)", line=dict(color='green', **style), opacity=1 if SHOW_GREEN else 0), row=1, col=1)
    fig.add_trace(go.Scatter(x=wx, y=wy, name="Wafer Center", line=dict(color='black', dash='dot', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dx, y=dy, name="Disk (Orange)", line=dict(color='orange', **style), opacity=1 if SHOW_ORANGE else 0), row=1, col=1)
    
    
    # 距離圖
    fig.add_trace(go.Scatter(x=t, y=np.hypot(p1x, p1y), name="Dist P1", line=dict(color='blue'), opacity=1 if SHOW_BLUE else 0), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=np.hypot(p2x, p2y), name="Dist P2", line=dict(color='green'), opacity=1 if SHOW_GREEN else 0), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=r_h, name="Dist Wafer", line=dict(color='black', dash='dot')), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=r_d, name="Dist Disk", line=dict(color='orange'), opacity=1 if SHOW_ORANGE else 0), row=2, col=1)

    # 動態點
    fig.add_trace(go.Scatter(x=[p1x[0]], y=[p1y[0]], mode="markers", marker=dict(size=8, color="blue"), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=[p2x[0]], y=[p2y[0]], mode="markers", marker=dict(size=8, color="green"), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=[dx[0]], y=[dy[0]], mode="markers", marker=dict(size=8, color="orange"), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=[wx[0]], y=[wy[0]], mode="markers", marker=dict(size=8, color="black"), showlegend=False), row=1, col=1)

    # 示意圓圈
    fig.add_shape(type="circle", x0=-390, y0=-390, x1=390, y1=390, line_color="black", row=1, col=1)
    fig.add_shape(type="circle", x0=-150, y0=h_start*25.4 -150, x1= 150, y1= h_start*25.4 + 150, line_color="black", row=1, col=1)
    
    # 動畫幀 (更新 Rev 標籤就在這裡)
    frames = []
    indices = np.linspace(0, len(t)-1, 60, dtype=int)
    for i in indices:
        current_p_rev = rev_p[i]
        current_w_rev = rev_w[i]
        current_sec   = t[i]
        
        frame_annotation = [dict(
        text=f"Platen Rev: {current_p_rev:.1f} | Wafer Rev: {current_w_rev:.1f} | Time: {current_sec:.1f}s",
        xref="paper", yref="paper", 
        x=0.02, y=0.98, # 位置：左上角
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
        xaxis=dict(range=[-450, 450], scaleanchor="y", scaleratio=1, showgrid=True, title="X (mm)"),
        yaxis=dict(range=[-450, 450], showgrid=True, title="Y (mm)"),
        xaxis2=dict(title="Time (s)", range=[0, TOTAL_TIME]),
        yaxis2=dict(title="Distance (mm)", range=[0, 400]),
        sliders=[{"active": 0, "steps": [{"args": [[f.name], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                                          "label": f"{t[int(f.name[1:])]:.1f}", "method": "animate"} for f in frames]}],
        updatemenus=[{"type": "buttons", "buttons": [{"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": 40, "redraw": False}, "fromcurrent": True}]},
                                                     {"label": "Pause", "method": "animate", "args": [[None], {"mode": "immediate"}]}]}]
    )
    st.plotly_chart(fig, use_container_width=True)
