# DDA_emulator
This project is a lightweight, web-based emulator of a rehabilitation “Balloon” game that demonstrates Dynamic Difficulty Adjustment (DDA) driven by a feedback controller.
# Balloon Game – Controller-Only DDA Emulator (Streamlit)

A lightweight, web-ready emulator of a rehabilitation **Balloon** game that demonstrates **Dynamic Difficulty Adjustment (DDA)** using only classic feedback control (P / D / PID). The app treats **difficulty as a single scalar** tuned by a controller to hold a **target performance** (“flow”) level while simulating a player’s **skill** and **observed performance** (points, lives).


##  Overview

* **Goal:** keep performance near a target $p^\*$ by adjusting a single difficulty variable $d_t$.
* **Player model:** latent **skill** learns (mid-zone) and fatigues when the game is harder than the player.
* **Observations:** **points** and **lives** $\in[0,1]$ with optional noise; combined into $p_{\text{raw}}$ and a rolling average $p_{\text{avg}}$.
* **Controller:** P / D / PID acts on error $e=p^\*-p_{\text{avg}}$; PID includes a simple anti-windup.


## Core Idea (Math)

Expected performance:

$$
\mu_t=\sigma\!\big(\beta(\text{skill}_t - d_t)-\theta\big),\quad
\sigma(z)=\tfrac{1}{1+e^{-z}}
$$

Observed metrics (clipped to $[0,1]$):

$$
\text{points}_t \approx \mu_t,\quad
\text{lives}_t \approx 0.5\,\mu_t
$$

Combined performance and controller input:

$$ p_{\text{raw},t} = 0.6\,\text{points}_t + 0.4\,\text{lives}_t, $$

$$ p_{\text{avg},t} = \operatorname{rollmean}_W\!\big(p_{\text{raw}}\big), e_t = p^{*} - p_{\text{avg},t}. $$


Difficulty update (sign chosen so **low $p$** → **reduce difficulty**):

$$
d_{t+1}=\text{clip}\big(d_t - (K_p e_t + K_i \sum e + K_d\,\Delta p),\ [d_{\text{lo}},d_{\text{hi}}]\big)
$$

Skill update (learn in mid-zone, fatigue when $d_t>\text{skill}_t$):

$$
s_{t+1}=\mathrm{clip}\!\left(
s_t+\alpha\,p_{\mathrm{avg},t}\!\left(1-p_{\mathrm{avg},t}\right)
\left(1-\frac{s_t}{s_{\max}}\right)
-\gamma\,\max\!\left(0,\,d_t-s_t\right),\;
[\,0,\,s_{\max}\,]
\right).
$$


## What You See

1. **Skill + Performance** (twin axes; performance axis colorized)
2. **Performance & Difficulty:** $p_{\text{avg}}$, $p_{\text{raw}}$, dashed **points**/**lives**, and **normalized difficulty**
3. **Derived difficulties:** continuous **drop speed** (px/s) & **spawn rate** (spawns/interval ×100) **plus integerized overlays** (engine-friendly)


## Key Features

* **Controller-only difficulty** (no multi-knob coupling)
* **Humanized observations:** optional heteroskedastic noise ↑ when $d>\text{skill}$
* **Initial conditions:** sliders for **initial skill** and **initial observed** points/lives
* **Safety/stability:** difficulty clamps; PID anti-windup
* **Clean, minimal UI** in Streamlit


## Controls & Parameters (Sidebar)

* **Player / Model:**
  `α` (learning), `γ` (fatigue), `β` (sensitivity), `θ` (bias), `s_max` (ceiling), `s0` (initial skill)
  Initial observations: `p₀_points`, `p₀_lives`
  Noise: **Add noise** (checkbox)

* **Controller:**
  Type = **P / D / PID**, target `p*`, gains `Kp`, `Ki`, `Kd`

* **Internals (fixed in code):**
  Horizon `T`, rolling window `W`, difficulty bounds `[d_lo, d_hi]`


##  Run Locally

```bash
# 1) Create env (optional)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install
pip install streamlit==1.38.0 numpy matplotlib

# 3) Run
streamlit run app.py    # or your filename
```


## Deploy on the Web

**Streamlit Community Cloud (easiest):**

1. Push `app.py` and `requirements.txt` to GitHub:

   ```
   streamlit==1.38.0
   numpy
   matplotlib
   ```
2. Go to share.streamlit.io → **New app** → select repo/branch/file → **Deploy**.
3. Share the public URL; manage access (public or by email).

**Hugging Face Spaces:**

* Create a Space (SDK: Streamlit), upload `app.py` + `requirements.txt`, deploy automatically.

**Render/Railway:**

* Web Service → start command:

  ```
  streamlit run app.py --server.port $PORT --server.address 0.0.0.0
  ```

**Docker (advanced):**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit","run","app.py","--server.port=8501","--server.address=0.0.0.0"]
```



## 🛠Planned Extensions (Optional)

* **Player profiles** (Noob/Casual/Skilled/Pro): sets `s0, s_max, α, γ, β`, noise, and perceptual lag
* **Population mode**: run many sampled players → mean ± band plots for controller robustness
* **Auto-tuning**: grid/Bayesian search on gains using metrics (time-in-flow, overshoot, settling time)

