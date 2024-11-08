<img src="./figures/KAMMUY_logo.jpg" alt="KAMMUY logo" width="300"/>

# KAMMUY - Kinetic And Magnetohydrodynamic MUlti-hierarchY simulation code

KAMMUY solves MHD & PIC simultaneously to understand multi-scale phenomena of plasmas.
PIC and MHD simulation is performed by GPUs using CUDA & C++ and Thrust library.
KAMMUY tries to adopt heterogeneous computing.
This is the modern computing style in a CPU & GPU environment.

KAMMUY(KAMUY or KAMUI, 神威) refers to gods or spiritual beings in the traditional belief system of the Ainu people in Hokkaido, Japan.

# Not Completed !


## Schemes (multi GPU)

### MHD
- HLLD
- MUSCL(minmod)
- RK2
- Upwind CT

### PIC
- Yee lattice
- Leapfrog
- Langdon-Marder type correction

### Interface
- interlocking method
- convolution
  

## Schemes (single GPU)

### MHD
- HLLD
- MUSCL(minmod)
- RK2
- CT

### PIC
- Yee lattice
- Leapfrog
- Langdon-Marder type correction

### Interface
- interlocking method
- convolution

## reference

- T. Sugiyama & K. Kusano, J. Comput. Phys., 227, 1340, 2007
- S. Usami et al., Phys. Plasmas., 20, 061208, 2013
- Lars K.S. Daldorff et al., J. Comput. Phys. 268, 236, 2014
- K.D. Makwana et al., Comput. Phys. Comm. 221, 81, 2017
