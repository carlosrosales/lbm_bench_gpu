//-------------------------------------------------------------------------------
// File     : constants.h
// Revision : 1.0 (2016/02/22)
// Author   : Carlos Rosales [ carlos.rosales.fernandez (at) gmail.com ]
//-------------------------------------------------------------------------------
// Define constant values to use in the computation. This includes both equation
// coefficients and some commonly used values that we want to calculate only 
// once and reuse throughout the code. 
//-------------------------------------------------------------------------------

//-------------------------------------------------------------------------------
// Copyright 2016 Carlos Rosales Fernandez and The University of Texas at Austin.
// Copyright 2008 Carlos Rosales Fernandez, David S. Whyte and IHPC (A*STAR).
//
// This file is part of MP-LABS.
//
// MP-LABS is free software: you can redistribute it and/or modify it under the
// terms of the GNU GPL version 3 or (at your option) any later version.
//
// MP-LABS is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
// A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with
// MP-LABS, in the file COPYING.txt. If not, see <http://www.gnu.org/licenses/>.
//-------------------------------------------------------------------------------

#define MAX_BUBBLES 20
#define xmin 1
#define ymin 1
#define zmin 1

// MPI partitioning constants
#define MPI_DIM     1
#define REORDER     1
#define master      0
int PERIODIC = 1;

// Unchangeable constants
#define MB 1048576.f
#define GB 1073741820.f

#define inv6  0.16666666666666666667f
#define inv12 0.08333333333333333333f
#define invPi 0.31830988618379067154f

#define Cs       0.57735026918962576451f
#define Cs_sq    0.33333333333333333333f
#define invCs_sq 3.00000000000000000000f

// Distribution weights
#define W0  0.33333333333333333333f
#define W1  0.05555555555555555556f
#define W2  0.02777777777777777778f

// Modified distribution weights
#define WC0 1.00000000000000000000f
#define WC1 0.16666666666666666667f
#define WC2 0.08333333333333333333f

// CPU Global scalars
int xmax, ymax, zmax, NX_h, NXY_h, NY_h, NZ_h;
int zl, zlg, zu, zug;
int maxStep, relaxStep, save, stat, step;
int BLOCK_SIZE_X, BLOCK_SIZE_Y, gridSize, nBubbles, RELAX = -1;
int top, bot, nprocs, proc, vproc, bufSize;
MPI_Comm MPI_COMM_VGRID;

float Gamma, sigma, width, rhoH, rhoL, RELAX_GRAV = 0.f;
float alpha, alpha4, kappa, invEta2, eta, eta2;
float tauPhi, invTauPhi, invTauPhiOne, phiStar, phiStarSq;
float tauRho, invTauRhoOne;
float gravity, grav, Eo, invInitVol, invRelaxVol;
float Wn0, Wn1, Wn2, WnC0, WnC1, WnC2;
float setupTime = 0.f, relaxTime = 0.f, mainTime = 0.f, elapsedTime = 0.f;

// CPU Global arrays
float bubbles[MAX_BUBBLES][4];

// GPU constants
__device__   float grav_d;
__constant__ int   dcol, NX, NXY, NY, NZ, zl_d, zu_d;
__constant__ float alpha4_d, kappa_d, phiStarSq_d, Gamma_d;
__constant__ float invEta2_d, invTauPhi_d, invTauPhiOne_d, invTauRhoOne_d;
__constant__ float K0, K1, K2, KC0, KC1, KC2, eta_d, eta2_d;

