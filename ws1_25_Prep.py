#================================================================= 
#
# AE2220-II: Computational Modelling 
# Code for work session 1, preparation part 2
#
#=================================================================
# This code provides a base for computing the linearised 
# perturbation potential around a slender 2D body 
# symmetric about the x axis. Some key lines in the code include:
#
# lines 25-32: Input parameters 
# lines 70-77: Implementation of finite-difference scheme.
# 
#=================================================================

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#=========================================================
# Input parameters
#=========================================================
x1         = -0.5;               # Forward boundary position
x2         = 2.0;                # Rear boundary position
y2         = 1.0;                # Upper boundary position
nmax       = 40;                 # Number of mesh points in i
jmax       = 10;                 # Number of mesh points in j
c          = 1.0;                # Advection speed
maxl       = 50;                 # maximum grid lines on plots
stride     = 1;                  # Point skip rate for suface plot


#=========================================================
# Derived parameters
#=========================================================
dx        = (x2-x1)/(nmax-1);        # Mesh spacing in x
dy        = (y2)/(jmax-1);           # Mesh spacing in y




#=========================================================
# Load the mesh coordinates.
#=========================================================
x      = np.zeros((nmax,jmax));          # Mesh x coordinates
y      = np.zeros((nmax,jmax));          # Mesh y coordinates

# Mesh coordinates
for j in range(0, jmax):
  x[:,j] = np.linspace(x1, x2, nmax)

for n in range(0, nmax):
  y[n,:] = np.linspace(0, y2, jmax)



#=========================================================
# March explicitly in x, solving for the unknown 
# Riemann invariant, R
#=========================================================
R = np.zeros((nmax,jmax));  # Note R(0,:)=0 for n=0;


#**************************************
# Uncomment the "for" and "R" lines 
# below then add code where requested
#**************************************
for n in range(0, nmax-1):   # March from x=0 to x=2
  
   # Apply boundary condition at y=0
  R[n+1,0] = math.exp(-20 * (x[n+1, 0] - 0.2)**2)
   
   # Update interior values using a first-order accurate upwind scheme
  for j in range(1, jmax):
    R[n+1,j] = R[n,j] - c*dx/dy*(R[n,j] - R[n,j-1])



#------------------------------------------------------
#  Compute wave magnitude at upper boundary
#------------------------------------------------------
wavemag = 0.;
for n in range(0, nmax-1):   # March from x=0 to x=2
  avg     = 0.5*(R[n,jmax-1] + R[n+1,jmax-1]);
  wavemag = wavemag + avg*dx;

print ("nmax     jmax    wavemag")
print (nmax,"    ",jmax,"    ",wavemag)


#------------------------------------------------------
#  Plot results
#------------------------------------------------------
fig = plt.figure(figsize=(18,10))

ax1 = plt.subplot2grid((2,2), (0,0), colspan=2, rowspan=2, projection='3d')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('R')
ax1.plot_surface(x, y, R, shade=True, rstride=stride,cstride=stride,
cmap=plt.cm.CMRmap, linewidth=0, antialiased=True);

ax1.view_init(30, -120)
plt.show()


print ("done")
