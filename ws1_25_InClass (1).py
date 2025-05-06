#================================================================= 
#
# AE2220-II: Computational Modelling 
# Code for work session 1
#
#=================================================================
# This code provides a base for computing the linearised 
# perturbation potential around a slender 2D body 
# symmetric about the x axis. Some key lines in the code include:
#
# lines 23-35:  Input parameters 
# lines 56-60:  Definition of the body geometry
# lines 96-103: Implementation of finite-difference scheme.
# 
#=================================================================

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#=========================================================
# Input parameters
#=========================================================
Mach       = 1.8;                # Mach number
Ufs        = 1.0;                # Freestream velocity
targArea   = 0.10;               # Body target area

x1         = -0.5;               # Forward boundary position
x2         = 2.1;                # Rear boundary position
y2         = 1.0;                # Upper boundary position
nmax       = 400;                 # Number of mesh points in i
jmax       = 200;                 # Number of mesh points in j

plots      = 1;                  # Make plots if not zero
stride     = 1;                  # Point skip rate for suface plot
maxl       = 50;                 # maximum grid lines on plots


#=========================================================
# Derived parameters
#=========================================================
beta      = math.sqrt(Mach*Mach-1);  # sqrt(M^2-1)
dx        = (x2-x1)/(nmax-1);        # Mesh spacing in x
dy        = (y2)/(jmax-1);           # Mesh spacing in y


#=========================================================
# Define the lower boundary (perturbation) geometry 
# and compute its derivative theta=dy/dx
#=========================================================
xlower = np.linspace(x1, x2, nmax);
ylower = np.zeros(nmax);
theta  = np.zeros(nmax);

# Body surface definition
for n in range(0, nmax):
  if xlower[n] > 0.0 and xlower[n]<1.0:
     xl = xlower[n];
     ylower[n] = 0.3*(1-xl)*(xl); 
     
     
# Body surface Derivative 
# (dy/dx = theta assumed zero at x=x1 and x=x2)
for n in range(1, nmax-1):
 theta[n] = (ylower[n+1]-ylower[n-1])/(2*dx);



#=========================================================
# Load the mesh coordinates and
# compute theta, the dy/dx of the body geometry
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
R = np.zeros((nmax,jmax));  # Note R(0,:)=0 for i=0;

mu = (1/beta)*dx/dy;  # CFL number for upwind scheme
#**************************************
# Uncomment the "for" and "R" lines 
# below then add code where requested
#**************************************
for n in range(0, nmax-1):   # March from x=0 to x=2
  
   # Apply boundary condition at y=0
  R[n+1,0] = 2*Ufs*theta[n+1] 
   
   # Update interior values using a first-order accurate upwind scheme
  for j in range(1, jmax):
    R[n+1,j] = R[n,j] - mu*(R[n,j] - R[n,j-1])



#=========================================================
# Compute velocities, cp, area, wave drag, cp constraint
#=========================================================
u=np.zeros((nmax,jmax));
v=np.zeros((nmax,jmax));
cp=np.zeros((nmax,jmax));

for n in range(nmax):
  for j in range(jmax):
     v[n,j]  =  R[n,j]/2.
     u[n,j]  = -v[n,j]/beta;
     cp[n,j] = -2*u[n,j]/Ufs;

area    = 0.
drag    = 0.
for n in range(nmax-1):
   xmid   = (xlower[n]+xlower[n+1])/2;
   ymid   = (ylower[n]+ylower[n+1])/2;
   pmid   = (cp[n,0]+cp[n+1,0])/2;
   tmid   = (theta[n+1]+theta[n])/2;
   dpdx   = (cp[n+1,0]-cp[n,0])/(xlower[n+1]-xlower[n]);
   area   += 2.*dx*ymid;
   drag   += 2.*dx*pmid*tmid;

cpcon   = 0.
for n in range(nmax):
   if xlower[n] > 1.5 and xlower[n] <2.1:
     cpcon=max(abs(cp[n,jmax-1]),cpcon)


q=dx/(dy*beta);
print ("------------------------------------------")
print ("Summary:")
print ("------------------------------------------")
print ("Mach, Beta        = ",Mach,",",beta)
print ("nmax x jmax       = ",nmax,"x",jmax)
print ("q=dx/(dy*beta)    = ",dx/(dy*beta))
print ("area/target       = ",area/targArea)
print ("body drag/op      = ",drag/(12*targArea*targArea/beta))
print ("body drag         = ",drag)
print ("|cp|(1.5-2.1,"+str(y2)+") = ",cpcon)
print ("------------------------------------------")

#------------------------------------------------------
#  Plot results
#------------------------------------------------------
if plots != 0:

 fig = plt.figure(figsize=(18,10))

 ax1 = plt.subplot2grid((2,4),(0,0),colspan=2,rowspan=2,projection='3d')
 ax1.set_xlabel('x')
 ax1.set_ylabel('y')
 ax1.set_zlabel('R')
 ax1.plot_surface(x, y, R, shade=True, rstride=stride,cstride=stride,
 cmap=plt.cm.CMRmap, linewidth=0, antialiased=True);

 ax3 = plt.subplot2grid((2,4), (0,2),colspan=2)
 ax3.set_title(r"$v(x,y) \;and\; \theta(x)$")
 a = ax3.contourf(x, y, v, cmap=plt.cm.jet)
 if (nmax<maxl) & (jmax<maxl):
   ax3.plot(x, y, '-k', x.transpose(), y.transpose(), '-k')
 ax3.plot(xlower,0.2*theta,'--',linewidth=2.0,color='blue')
 fig.colorbar(a, ax=ax3)

 ax4 = plt.subplot2grid((2,4), (1,2),colspan=2)
 ax4.set_title(r"$c_p(x,y) \;and\; ylower(x)$")
 a = ax4.contourf(x, y, cp, cmap=plt.cm.jet)
 if (nmax<maxl) & (jmax<maxl):
  ax4.plot(x, y, '-k', x.transpose(), y.transpose(), '-k')
 ax4.plot(xlower,ylower,linewidth=2.0,color='blue')
 fig.colorbar(a, ax=ax4)

 ax1.view_init(30, -120)
 plt.savefig('super_' + str(nmax) + '_' + str(q) + '.png',dpi=250)
 plt.show()

#------------------------------------------------------
#  All done
#------------------------------------------------------
print ("done")
