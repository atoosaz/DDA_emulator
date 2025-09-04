import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Balloon DDA – Simple Controller-Only Difficulty", layout="wide")
st.title("Balloon Game – Controller-Only Difficulty ")

sb = st.sidebar

# =============================
# Player / model parameters (minimal)
# =============================
sb.header("Player / Model")
alpha = sb.slider("Learning Rate α", 0.0, 1.0, 0.11, 0.01)
gamma = sb.slider("Fatigue γ", 0.0, 1.0, 0.00, 0.01)
beta  = sb.slider("Perf Sensitivity β", 0.0, 1.5, 0.3, 0.1)
theta = sb.slider("Bias θ", -4.0, 4.0, 0.0, 0.1)
s_max = sb.slider("Skill Ceiling s_max", 0.2, 3.0, 2.0, 0.05)
s0    = sb.slider("Initial Skill s₀", 0.0, 1.0, 0.06, 0.01)

# Initial performance sliders (used at t=0)
sb.subheader("Initial Observed Performance at t=0")
p0_points = sb.slider("Initial Performance", 0.0, 1.0, 0.67, 0.01)
#p0_lives  = sb.slider("Initial Lives ℓ₀",  0.0, 1.0, 0.41, 0.01)
p0_lives = 0.40
# =============================
# Controller (adjusts difficulty directly)
# =============================
sb.header("Controller")
controller = sb.selectbox("Type", ["P", "D", "PID"])
target_p = sb.slider("Target p*", 0.0, 1.0, 0.70, 0.01)

if controller == "P":
    Kp = sb.slider("Kp", 0.0, 5.0, 0.70, 0.05)
    Ki = 0.0
    Kd = 0.0
elif controller == "D":
    Kd = sb.slider("Kd", 0.0, 10.0, 2.0, 0.05)
    Kp = 0.0
    Ki = 0.0
else:  # PID
    Kp = sb.slider("Kp", 0.0, 5.0, 1.0, 0.05)
    Ki = sb.slider("Ki", 0.0, 1.0, 0.05, 0.01)
    Kd = sb.slider("Kd", 0.0, 10.0, 2.0, 0.05)

# =============================
# Fixed design constants (kept simple)
# =============================
T = 300
rng = np.random.default_rng(7)
W = 15                   # window for rolling means
d_lo, d_hi = -3.0, 3.0   # clamp difficulty to a wide range to avoid runaway

# Noise toggle
sb.header("Observation Noise")
use_noise = sb.checkbox("Add noise to points & lives", value=True)

# =============================
# State arrays
# =============================
skill = np.zeros(T); skill[0] = s0
d     = np.zeros(T); d[0] = -3.0   # start at the lower bound

points = np.zeros(T)
lives  = np.zeros(T)
p_raw  = np.zeros(T)   # instant combined performance
p_avg  = np.zeros(T)   # rolling average (used by controller)

# Derived “physical” difficulties (visualization only)
d_dropr  = np.zeros(T)   # e.g., pixels/sec
d_spawnr = np.zeros(T)   # e.g., spawns/interval

# Display ranges for derived difficulties
min_drop,  max_drop  = 35.0, 400.0
min_spawn, max_spawn = 1.0, 5.0

# initialize derived channels at t=0 to mins
d_dropr[0]  = min_drop
d_spawnr[0] = min_spawn

integral = 0.0
prev_pavg = target_p

def rolling_mean(arr, t, W):
    start = max(0, t - W + 1)
    return float(np.mean(arr[start:t+1]))

# =============================
# Simulation
# =============================
for t in range(T-1):
    # --- expected performance from logistic(skill - difficulty)
    logit = float(np.clip(beta*(skill[t] - d[t]) - theta, -20.0, 20.0))
    mu = 1.0 / (1.0 + np.exp(-logit))

    # --- observations (points & lives in [0,1])
    # if t == 0:
    #     # respect user-chosen initial observations
    #     points[t] = np.clip(p0_points, 0.0, 1.0)
    #     lives[t]  = np.clip(p0_lives,  0.0, 1.0)
    # else:
    #     if use_noise:
    #         adv = d[t] - skill[t]  # positive if game is harder than player
    #         noise_scale_points = 0.03 + 0.12*(1.0 / (1.0 + np.exp(-adv)))   # in [0.03, 0.15]
    #         noise_scale_lives  = 0.02 + 0.08*(1.0 / (1.0 + np.exp(-adv)))   # in [0.02, 0.10]
    #         points[t] = np.clip(mu + rng.normal(0.0, noise_scale_points), 0.0, 1.0)
    #         lives[t]  = np.clip(0.5*mu + rng.normal(0.0, noise_scale_lives), 0.0, 1.0)
    #     else:
    #         points[t] = np.clip(mu,     0.0, 1.0)
    #         lives[t]  = np.clip(0.5*mu, 0.0, 1.0)
    if use_noise:
        adv = d[t] - skill[t]  # positive if game is harder than player
        noise_scale_points = 0.03 + 0.12*(1.0 / (1.0 + np.exp(-adv)))   # in [0.03, 0.15]
        noise_scale_lives  = 0.02 + 0.08*(1.0 / (1.0 + np.exp(-adv)))   # in [0.02, 0.10]
        points[t] = np.clip(mu + rng.normal(0.0, noise_scale_points), 0.0, 1.0)
        lives[t]  = np.clip(0.5*mu + rng.normal(0.0, noise_scale_lives), 0.0, 1.0)
    else:
        points[t] = np.clip(mu,     0.0, 1.0)
        lives[t]  = np.clip(0.5*mu, 0.0, 1.0)

    # --- combine to a single performance p (simple fixed mix)
    p_raw[t] = (0.6*points[t] + 0.4*lives[t]) + (0.3*p0_points + 0.2*p0_points)
    p_avg[t] = rolling_mean(p_raw, t, W)

    # --- controller adjusts difficulty directly
    if controller == "P":
        e = target_p - p_avg[t]
        u = Kp * e
    elif controller == "D":
        dp = p_avg[t] - prev_pavg if t > 0 else 0.0
        u  = Kd * dp
    else:  # PID
        e  = target_p - p_avg[t]
        dp = p_avg[t] - prev_pavg if t > 0 else 0.0
        # simple anti-windup: only integrate when not saturating further
        tentative = Kp*e + Ki*(integral + e) + Kd*dp
        at_upper = d[t] >= d_hi - 1e-6
        at_lower = d[t] <= d_lo + 1e-6
        if not ((at_upper and tentative < 0) or (at_lower and tentative > 0)):
            integral += e
        u = Kp*e + Ki*integral + Kd*dp

    prev_pavg = p_avg[t]

    # sign convention: if p is low (too hard), e>0 => u>0 => we REDUCE difficulty
    d_next = d[t] - u
    d[t+1] = float(np.clip(d_next, d_lo, d_hi))

    # --- derived "physical" difficulties (for visualization only)
    # map d ∈ [d_lo, d_hi] to display ranges
    d_dropr[t+1]  = (d[t+1] - d_lo) * (max_drop - min_drop) / (2.0*d_hi)  + min_drop
    d_spawnr[t+1] = (d[t+1] - d_lo) * (max_spawn - min_spawn) / (2.0*d_hi) + min_spawn

    # --- skill update: learn in mid p, fatigue when difficulty > skill
    workload = max(0.0, d[t] - skill[t])
    skill[t+1] = skill[t] + alpha*p_avg[t]*(1.0 - p_avg[t])*(1.0 - skill[t]/s_max) - gamma*workload
    skill[t+1] = float(np.clip(skill[t+1], 0.0, s_max))

# finalize last sample for plotting
logit_last = float(np.clip(beta*(skill[-1] - d[-1]) - theta, -20.0, 20.0))
mu_last = 1.0 / (1.0 + np.exp(-logit_last))
if use_noise:
    points[-1] = np.clip(mu_last + rng.normal(0.0, 0.03), 0.0, 1.0)
    lives[-1]  = np.clip(0.5*mu_last + rng.normal(0.0, 0.02), 0.0, 1.0)
else:
    points[-1] = np.clip(mu_last,     0.0, 1.0)
    lives[-1]  = np.clip(0.5*mu_last, 0.0, 1.0)
p_raw[-1]  = 0.6*points[-1] + 0.4*lives[-1]
p_avg[-1]  = rolling_mean(p_raw, T-1, W)

# =============================
# Plot 1: Skill + Performance (colored right axis)
# =============================
fig1, ax1 = plt.subplots(figsize=(20, 6))
ax1.plot(skill, label="Skill", linewidth=2)
ax1.set_title("Skill and Performance vs Time")
ax1.set_xlabel("Time")
ax1.set_ylabel("Skill")
ax1.grid(alpha=0.25)

ax1b = ax1.twinx()
perf_color = "tab:orange"
ax1b.plot(p_avg, label="Performance p(t) (avg)", linewidth=2, color=perf_color)
ax1b.set_ylabel("Performance", color=perf_color)
ax1b.tick_params(axis='y', colors=perf_color)
ax1b.spines['right'].set_color(perf_color)

# build joint legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1b.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
plt.tight_layout()
st.pyplot(fig1)

# =============================
# Plot 2: Performance (raw & avg) + Difficulty
# =============================
fig2, ax2 = plt.subplots(figsize=(20, 10))
ax2.plot(p_avg, label="Performance p(t) (avg)", linewidth=2)
ax2.plot(p_raw, label="Performance p(t) (raw)", linewidth=1, alpha=0.7)
ax2.plot(points, linestyle="--", linewidth=1, label="Points (raw)")
ax2.plot(lives, linestyle="--", linewidth=1, label="Lives (raw)")
ax2.plot((d - d_lo)/(2 * d_hi), label="Difficulty d(t) (normalized)", linewidth=2)

ax2.axhline(target_p, linestyle="--", linewidth=1.2, label="Target p*")
ax2.set_title("Performance and Difficulty Over Time")
ax2.set_xlabel("Time")
ax2.set_ylabel("Normalized value")
ax2.grid(alpha=0.25)
ax2.legend(loc="lower right")
plt.tight_layout()
st.pyplot(fig2)

# =============================
# Plot 3: Derived Difficulties (continuous + integerized)
# =============================
drop_int  = np.round(d_dropr).astype(int)
spawn_int = np.round(d_spawnr).astype(int)

fig3, ax3 = plt.subplots(figsize=(20,6))
ax3.plot(d_dropr,  label="Drop difficulty (pixels/sec)", linewidth=2)
ax3.plot(d_spawnr*100, label="Spawn difficulty (spawns/interval ×100)", linewidth=2)

# integerized overlays
ax3.plot(drop_int,  linestyle=":", linewidth=1, label="Drop difficulty (int)")
ax3.plot(spawn_int*100, linestyle=":", linewidth=1, label="Spawn difficulty (int ×100)")

ax3.set_title("Derived Difficulties")
ax3.set_xlabel("Time")
ax3.set_ylabel("Levels")
ax3.grid(alpha=0.25)
ax3.legend(loc="upper left")
plt.tight_layout()
st.pyplot(fig3)

st.caption(
    "Plot 1 shows Skill and averaged Performance (right axis colored). "
    "Plot 2 shows avg and raw performance plus raw Points/Lives as dashed lines, and normalized Difficulty. "
    "Plot 3 displays derived difficulties (continuous) and their integerized versions."
)
