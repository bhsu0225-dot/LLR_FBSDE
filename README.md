# High-dimensional semilinear parabolic PDEs: a localized stochastic particle solver

This repository contains MATLAB implementations accompanying the paper *A Mesh-free, derivative-free, matrix-free, and highly parallel localized stochastic method for high-dimensional semilinear parabolic PDEs*. Each program solves a semilinear parabolic PDE by combining forward particle simulation with a backward theta-scheme recursion. Conditional expectations are approximated by local linear regression on a nearest-neighbour particle subset, and the resulting scalar nonlinear equation at each particle is solved by Newton iteration.

The examples include a high-dimensional Allen--Cahn-type equation, a viscous Burgers-type equation, and a Hamilton--Jacobi equation. All computation is carried out directly on the particles.

## Requirements

- MATLAB with support for local functions in scripts and implicit array expansion.
- Parallel Computing Toolbox when parallel execution is enabled.

Each file contains its own numerical helper functions. No external datasets or third-party numerical packages are required.

## Method

All three programs follow the same structure:

1. **Forward simulation.** Particles are started at the common initial point $x_0$ and evolved under pure Brownian diffusion with generator $\nu \Delta$ (or $D\Delta$ in the Hamilton--Jacobi example).
2. **Terminal condition.** The prescribed exact solution is evaluated at the terminal time $T$ on every particle.
3. **Backward recursion.** For each time level $k=N,\dots,1$ and each particle $m$, a neighbourhood is selected around the anchor $X_k^m$, a weighted local linear fit provides the value and gradient estimates, and a theta/Newton update produces $Y_k^m$.

Three features are common to the programs.

- **Neighbour selection.** Particles inside a ball of radius $\varepsilon_k$ around the anchor are used. If the ball contains too few particles, the nearest neighbours are taken instead. The anchor itself is always excluded.
- **Kernel weighting.** Gaussian, Epanechnikov, and triweight kernels are available. The Gaussian kernel exp(-r^2) is the default.
- **Local linear regression.** Weighted least squares with a ridge penalty on the gradient coefficients gives a local intercept and gradient. In the Burgers program the same fit also returns the normalized kernel weights used for the conditional-expectation estimates.

## Files and numerical examples

Each program can be run independently. Helper functions with the same names are local to their respective files.

### Example 1: high-dimensional Allen--Cahn-type equation

**`high_dim_allen_cahn_solver.m`** solves

$$u_t - \nu \Delta u + g(u) = 0, \qquad g(u) = \frac{a}{2}\big[(1+u)\log(1+u) + (1-u)\log(1-u)\big] - \frac{b}{2}u,$$

with $a,b>0$. The driver supplies the matching source term for the manufactured solution

$$u(t,x) = e^{\cos t -\lVert x\rVert^2}\cos\Big(\prod_{j=1}^{d} x_j\Big).$$

The program reports the estimate of $u(0,x_0)$, the exact value, the absolute and relative errors, and the elapsed time.

- The gradient computed by the local regression is not used in the Allen--Cahn update; the nonlinearity depends on $u$ only.
- At level $k+1$ the nonlinearity is averaged over the neighbours with equal weights.
- The Newton iteration uses at most 20 steps, stops when the increment falls below 1e-6, and clamps the iterate into $(-1,1)$ if it becomes non-finite. The default settings are $d=100$, $N=400$, and $M=100$, so the example runs on a laptop.

### Example 2: high-dimensional Burgers-type equation

**`high_dim_burgers_solver.m`** solves a viscous Burgers-type equation whose exact solution is the logistic travelling wave

$$u(t,x) = \frac{e^{z}}{1+e^{z}}, \qquad z = t + \frac{1}{d}\sum_{j=1}^{d} x_j,$$

with nonlinearity

$$f(u,\nabla u) = \Big(u - \tfrac{2+d}{2d}\Big)\sum_{j=1}^{d}\partial_{x_j}u.$$

The default viscosity is $\nu=d^{2}/2$, which keeps the diffusion in scale with the dimension.

- The radius is $\varepsilon_k=\kappa\sqrt{\Delta t}$ with $\kappa=0.5$.
- Conditional expectations use the normalized kernel weights returned by the local fit, both for $E_k[u_{k+1}]$ and for $E_k[f_{k+1}]$.
- The gradient at level $k+1$ enters the nonlinear term, so the scheme is derivative-free only in the sense that no derivatives of the neural-network or mesh type are formed; spatial gradients come from the local fit.
- The Newton iteration uses at most 12 steps and stops when the increment falls below 1e-8.
- The random seed is fixed (`seed = 123`). The default settings are $d=10$, $N=800$, and $M=200$.

### Example 3: high-dimensional Hamilton--Jacobi equation

**`high_dim_hamilton_jacobi_solver.m`** solves

$$u_t - D\Delta u + \kappa u \lvert\nabla u\rvert^2 = q,$$

with diffusion coefficient $D$, nonlinearity coefficient $\kappa$, and source $q$ chosen for the heat-kernel exact solution

$$u(t,x) = (1+4Dt)^{-d/2}\exp\Big(-\frac{\lVert x\rVert^2}{1+4Dt}\Big),$$

so that $q = 4Dd\,u/(1+4Dt)$.

- The Hamiltonian depends on the squared gradient norm, which is supplied by the local regression at both $t_k$ and $t_{k+1}$.
- The conditional expectation $E_k[Y_{k+1}]$ is a plain average over the selected neighbours, while the nonlinear term at level $k+1$ is also averaged with equal weights.
- The Newton iteration is deliberately short: at most 3 steps with an increment tolerance of 1e-8. This is consistent with the near-quadratic scalar equation but is the tightest of the three examples.
- The random stream is reset with `rng('default')`. The default settings are $d=10$, $N=400$, and $M=1000$.

## Running the programs

Download the repository, set the MATLAB current folder to the directory containing the three `.m` files, edit the parameter block near the beginning of the desired file, and run it by name:

```matlab
% Allen--Cahn-type equation
AllenCahn_Case2

% Burgers-type equation
Burgers_main

% Hamilton--Jacobi equation
Hamiltonjacobi_main
```

No other experiment needs to be run first. For an initial execution check, reduce the particle count `M` or the number of time steps `N`. Use the defaults when generating final results. Each run prints a progress percentage and elapsed time, followed by a result block with the estimated and exact values and the errors.

### Parallel execution

Parallelism is across particles. The `parfor m = 1:M` loop assigns each particle its own neighbourhood selection, local regression, and Newton solve.

**Allen--Cahn and Hamilton--Jacobi programs**

Both files execute the particle loop as a `parfor` and use the pool that is currently open. If no pool exists, MATLAB's automatic parallel pool starts when Parallel Computing Toolbox is installed; otherwise the loop runs serially.

**Burgers program**

`high_dim_burgers_solver.m` behaves the same way. To control the worker count explicitly, open a pool before the call:

```matlab
if isempty(gcp('nocreate'))
    parpool('local', 4);
end
high_dim_burgers_solver
```

Replace `4` with a suitable worker count. Set

```matlab
parpool('local', 0)   % or: gcp('nocreate'); delete(gcp('nocreate'))
```

before the call to force serial execution.

Increasing the worker count increases memory use, because each worker holds its own slice of the particle arrays `X` and `u` over all time levels. Adjust it to the available memory as well as the number of CPU cores.

## Parameters and implementation details

The main settings are collected near the beginning of each file:

- `d`: physical-space dimension.
- `nu` / `D`: diffusion coefficient in the Allen--Cahn / Burgers and Hamilton--Jacobi programs.
- `T`, `N`: terminal time and number of time steps. Time step is `dt = T/N`.
- `M`: number of particles launched from the initial point.
- `theta`: implicitness of the theta-scheme. `theta = 1` is fully implicit (backward Euler, the default). `theta = 0.5` gives a Crank--Nicolson-type update, which is exposed in the Burgers program only.
- `x0`: common starting point of all particles.
- `kernel_name`: weighting kernel, one of `'gaussian'`, `'epanechnikov'`, or `'triweight'`.
- `kappa`: in the Burgers program, the radius factor `eps_k = kappa*sqrt(dt)`; in the Hamilton--Jacobi program, the nonlinearity coefficient.
- `a`, `b`: positive constants in the Allen--Cahn nonlinearity `g(u)`. Making either one larger strengthens the logarithmic part of the potential and makes the scalar Newton solve less benign.
- `lambda_ridge`: ridge penalty applied to the gradient coefficients of the local fit. Set to 1e-3 in the Burgers program and 1e-6 in the other two.
- `min_nn`: minimum number of neighbours kept per anchor. The Burgers program exposes it; the others fall back to all particles except the anchor itself.
- `seed` / `rng('default')`: the Burgers program fixes `seed = 123`, the Hamilton--Jacobi program calls `rng('default')`, and the Allen--Cahn program leaves the stream untouched.

## Error measurements

Errors are computed against the prescribed exact solutions, not against a finer numerical reference solution.

- **Endpoint error.** Each program reports the absolute error `abs(u0_est - u_exact)` and a relative error at the single query point `x0`. 
- **Scope.** These are pointwise errors at the initial space-time point, not norms over the domain. 
- **Reproducibility.** The reported values depend on the random stream and on the worker scheduling of `parfor`, so results can vary slightly between runs unless the seed is fixed.

## Output

The programs write their results to the Command Window and do not create output folders or save figures.

| Program | Console output | Saved files |
| --- | --- | --- |
| `high_dim_allen_cahn_solver.m` | Result block with estimated and exact values, absolute and relative errors, elapsed time | None |
| `high_dim_burgers_solver.m` | Result block with estimated and exact values, absolute and relative errors, elapsed time | None |
| `high_dim_hamilton_jacobi_solver.m` | Estimated and exact values, absolute error, relative error, elapsed time | None |

To keep a record of a run, wrap the call in a `diary` session, or add a `save` statement after the solver returns. For example, the Allen--Cahn result can be captured with

```matlab
diary('allen_cahn_d100.log')
high_dim_allen_cahn_solver
diary off
```


