<img src="KAMMUY_logo.jpg" alt="KAMMUY logo" width="300"/>

# KAMMUY - Kinetic And Magnetohydrodynamic MUlti-hierarchY simulation code

KAMMUY solves MHD & PIC simultaneously to understand multi-scale phenomena of plasmas.
PIC and MHD simulation is performed by GPUs using CUDA & C++ and Thrust library.
KAMMUY tries to adopt heterogeneous computing.
This is the modern computing style in a CPU & GPU environment.

KAMMUY(KAMUY or KAMUI, 神威) refers to gods or spiritual beings in the traditional belief system of the Ainu people in Hokkaido, Japan.

## Schemes

### MHD

- HLLD
- MUSCL (minmod)
- RK2
- Projection

### PIC

- Yee lattice
- Leapfrog
- Langdon-Marder type correction

### Interface

- interlocking method
- convolution

## Library

- Thrust
- AmgX
  - PCG_W.json is used.

## reference

- T. Sugiyama & K. Kusano, Journal of Computational Physics, 227, 1340 (2007)
- S. Usami et al., Physics of Plasmas, 20, 061208 (2013)
- Lars K.S. Daldorff et al., Journal of Computational Physics, 268, 236 (2014)
- K.D. Makwana et al., Computer Physics Communication, 221, 81 (2017)
- M. Haahr et al., Astronomy & Astrophysics, 696, A191 (2025)
