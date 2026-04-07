# 2D Shallow Water Equation – CUDA and MPI Implementation

This repository contains implementations of numerical solvers for the **2D Shallow Water Equations** using different computational paradigms:

- **Sequential (CPU)**
- **MPI (distributed memory parallelism)**
- **CUDA (GPU acceleration)**

The numerical methods implemented include:

- **Lax–Friedrichs scheme**
- **Lax–Wendroff scheme**

Both **finite difference (FD)** and **finite volume (FV)** formulations are provided.

---

# Repository Structure

```
2D-Shallow-Water-Equation-CUDA-and-MPI
│
├── CUDA
│   ├── Lax_friedrich_fd.cu
│   ├── Lax_friedrich_fv.cu
│   └── Lax_wendroff_fd.cu
│
├── MPI
│   ├── Lax_friedrich_fd.c
│   ├── Lax_friedrich_fv.c
│   └── Lax_wendroff_fd.c
│
└── Sequential
    ├── Lax_friedrich_fd.c   (Finite Difference method with Lax–Friedrichs scheme)
    ├── Lax_friedrich_fv.c   (Finite Volume method with Lax–Friedrichs scheme)
    └── Lax_wendroff_fd.c    (Finite Difference method with Lax–Wendroff scheme)
```

---

# Implemented Numerical Methods

## Lax–Friedrichs Scheme
A first‑order explicit scheme commonly used for solving hyperbolic partial differential equations.  
It introduces numerical dissipation that helps stabilize the solution.

## Lax–Wendroff Scheme
A second‑order accurate method that improves accuracy compared to Lax–Friedrichs by incorporating Taylor expansion in time.

---

# Parallel Implementations

## CUDA Version
The CUDA implementation accelerates the solver using **GPU parallelism**.

## MPI Version
The MPI implementation distributes the computational grid across multiple processes for **distributed-memory parallel computing**.

## Sequential Version
A baseline CPU implementation used for validation and performance comparison.

---

# Purpose

This project is intended for:

- studying **numerical methods for hyperbolic PDEs**
- experimenting with **GPU computing (CUDA)**
- exploring **distributed parallel computing (MPI)**
- benchmarking performance across different computing architectures

---

# Possible Extensions

- Add visualization for simulation results
- Compare performance between CPU, MPI, and CUDA implementations
- Implement higher‑order numerical schemes
- Extend to more complex boundary conditions
