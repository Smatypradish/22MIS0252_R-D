# Parametric Curve Parameter Estimation

## Problem Overview
Given 1500 2D points $(x_i, y_i)$ sampled from the parametric curve for $6 < t < 60$:

$$
x(t) = t \cos(\theta) - e^{M|t|} \sin(0.3t) \sin(\theta) + X
$$
$$
y(t) = 42 + t \sin(\theta) + e^{M|t|} \sin(0.3t) \cos(\theta)
$$

Our goal is to identify the unknown parameters:
- $\theta \in (0^\circ, 50^\circ)$
- $M \in (-0.05, 0.05)$
- $X \in (0, 100)$

---

## Approach & Thought Process

### 1. Geometric Intuition (Coordinate Frame Rotation)
Looking at the equations, we notice standard 2D rotation and translation terms. 
- Origin shift: translated by $(X, 42)$
- Rotation angle: rotated by $\theta$

If we write this out in matrix form:

$$
\begin{pmatrix} x(t) - X \\ y(t) - 42 \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} t \\ e^{M|t|} \sin(0.3t) \end{pmatrix}
$$

By applying the inverse rotation matrix $R(-\theta)$, we can transform the points back into a clean, unrotated frame $(u, v)$:

$$
\begin{pmatrix} u \\ v \end{pmatrix} = \begin{pmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} x - X \\ y - 42 \end{pmatrix}
$$

This trick is super helpful because it separates the coordinates cleanly:

1. **Longitudinal coordinate ($u$):**

$$
t = (x - X)\cos\theta + (y - 42)\sin\theta
$$

2. **Transverse coordinate ($v$):**

$$
v_{\text{observed}} = -(x - X)\sin\theta + (y - 42)\cos\theta
$$
$$
v_{\text{expected}} = e^{M|t|} \sin(0.3t)
$$

### 2. Loss Formulation & Optimization
For each data point $(x_i, y_i)$ in the dataset, the residual (error) is simply:

$$
r_i(\theta, M, X) = v_{i, \text{observed}} - e^{M|t_i|}\sin(0.3 t_i)
$$

Since we have a closed-form expression for $t_i$ given $\theta$ and $X$, we don't need an expensive nearest-point projection algorithm. We just plug the points in and measure how far off they are from the expected wave.

I set up a two-step optimization pipeline using Python's `scipy.optimize`:
1. **Global Search (Differential Evolution):** The sine term causes a lot of ripples (local minima) in the loss landscape. Differential evolution explores the bounded parameter space robustly to find the global basin.
2. **Local Refinement (Least Squares):** Once we are in the right neighborhood, Levenberg-Marquardt refines the parameters with high tolerance to get the exact values.

---

## Results

After optimization, the parameters converged exactly to:
- **$\theta$:** $0.523599\text{ rad}$ which is exactly $\mathbf{30^\circ}$ ($\pi / 6$)
- **$M$:** $\mathbf{0.03}$
- **$X$:** $\mathbf{55.0}$

### Error Metrics:
- **Mean L1 Reconstruction Error on CSV points:** $\approx 2.05 \times 10^{-5}$ *(consistent with standard 6-decimal CSV float precision)*
- **L1 Distance on Uniformly Sampled $t$:** $\mathbf{0.0000}$ (Exact Match between predicted and expected curves)

---

## Desmos Submission Equation
The LaTeX formatted parametric equation (to paste directly into Desmos for domain $6 \le t \le 60$):

```latex
\left(t*\cos(0.523599)-e^{0.03\left|t\right|}\cdot\sin(0.3t)\sin(0.523599)+55,42+t*\sin(0.523599)+e^{0.03\left|t\right|}\cdot\sin(0.3t)\cos(0.523599)\right)
```

## Running the Code
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the script:
   ```bash
   python solve.py
   ```
This will output the metrics and generate `curve_fit.png` showing the expected curve overlaying the dataset perfectly.
