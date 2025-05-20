##================================================================= 
#
# AE2220-II: Computational Modelling 
# Main program for work session 2
#
# Line 95:  Definition of f for manufactured solution
# Line 117: Definition of Ke[i,j]
#
#
#=================================================================
# This code provides a base for computing the Laplace equation
# with a finite-element method based on triangles 
# with linear shape functions.
#=================================================================
import math
import numpy as np
import matplotlib.pyplot as plt
import TriFEMLibD_Prep

#=========================================================
# Input parameters
#=========================================================
n=1                   # Mesh refinement factor
a=2                   # Manufactured solution a
b=5                   # Manufactured solution b
xe1=1.0               # center of first electrode
xe2=4.0               # center of second electrode
le=0.5                # electrode length

#=========================================================
# Create the mesh 
#=========================================================
mesh = TriFEMLibD_Prep.TriMesh();
mesh.loadMesh(n)
#mesh.plotMesh(); quit(); 


#=========================================================
# Create a finite-element space.
# This object maps the degrees of freedom in an element
# to the degrees of freedom of the global vector.
#=========================================================
fes = TriFEMLibD_Prep.LinTriFESpace(mesh)


#=========================================================
# Prepare the global left-hand matrix, right-hand vector
# and solution vector
#=========================================================
sysDim = fes.sysDim
LHM    = np.zeros((sysDim,sysDim));
RHV    = np.zeros(sysDim);
solVec = np.zeros(sysDim);


#=========================================================
# Assemble the global left-hand matrix and
# right-hand vector by looping over the elements
print ("\nAssembling system of dimension",sysDim)
#=========================================================
for elemIndex in range(mesh.nElem):

  #----------------------------------------------------------------
  # Create a FiniteElement object for 
  # the element with index elemIndex
  #----------------------------------------------------------------
  elem = TriFEMLibD_Prep.LinTriElement(mesh,elemIndex)

  #----------------------------------------------------------------
  # Initialise the element vector and matrix to zero.
  # In this case we have only one unknown varible in the PDE (u),
  # So the element vector dimension is the same as
  # the number of shape functions (psi_i)  in the element.
  #----------------------------------------------------------------
  evDim   = elem.nFun
  elemVec = np.zeros((evDim))
  elemMat = np.zeros((evDim,evDim))

  #----------------------------------------------------------------
  # Evaluate the shape function integrals in the vector and matrix 
  # by looping over integration points (integration by quadrature)
  # int A = sum_ip (ipWeight*A_ip) where A is the function to be 
  # integrated and ipWeight is the weight of an integration point
  #----------------------------------------------------------------
  for ip in range(elem.nIP):

    # Retrieve the coordinates and weight of the integration point
    xIP      = elem.ipCoords[ip,0] 
    yIP      = elem.ipCoords[ip,1] 
    ipWeight = elem.ipWeights[ip];
  
    # Compute the local value of the source term, f
    # ***** For the manufactured solution add the appropriate value below
    # ***** For e.g. sin(z) use math.sin(z)
    fIP = -(a**2 + b**2) * math.sin(a * xIP) * math.sin(b * yIP);

    # Retrieve other values evaluated at this integration point (ip)
    # - perm is the value of permittivity at this ip
    # - psi[i] is the value of the function psi_i at this ip.
    # - gradPsi[i] is a vector contraining the x and y
    #   gradients of the function psi_i at this ip
    #   e.g.
    #     gradPsi[2][0] is the x gradient of shape 2 at point xIP,yIP
    #     gradPsi[2][1] is the y gradient of shape 2 at point xIP,yIP
    # perm    = mesh.getPerm(xIP,yIP);
    perm = 1.0;      
    psi     = elem.getShapes(xIP,yIP)
    gradPsi = elem.getShapeGradients(xIP,yIP)

    # Add this ip's contribution to the integrals in the
    # element vector and matrix
    for i in range(evDim):
      elemVec[i] += ipWeight*psi[i]*fIP;   # Right-hand side of weak form
      for j in range(evDim):
        # ***** Change the line below for the desired left-hand side
        # elemMat[i,j] += ipWeight*perm*psi[i]*psi[j]
        elemMat[i,j] += -ipWeight * perm * (gradPsi[i][0] * gradPsi[j][0] + gradPsi[i][1] * gradPsi[j][1])
 

  #----------------------------------------------------------------
  # Add the completed element matrix and vector to the system
  #----------------------------------------------------------------
  fes.addElemMat(elemIndex, elemMat, LHM )
  fes.addElemVec(elemIndex, elemVec, RHV ) 


#=========================================================
print ("Applying boundary conditions")
# Left boundary conditions
#=========================================================
#fes.printMatVec(LHM,RHV,"beforeConstraints")
for i in range(fes.nLeft):
   row = fes.leftDof[i];
   xy  = fes.leftCoords[i]; #x=xy[0],y=xy[1]
   LHM[row,:]   = 0.
   LHM[row,row] = 1.
   RHV[row]     = math.sin(a*xy[0])*math.sin(b*xy[1])


#=========================================================
# Right boundary conditions
#=========================================================
for i in range(fes.nRight):
   row = fes.rightDof[i];
   xy  = fes.rightCoords[i]; #x=xy[0],y=xy[1]
   LHM[row,:]   = 0.
   LHM[row,row] = 1.
   RHV[row]     = math.sin(a*xy[0])*math.sin(b*xy[1])


#=========================================================
# Lower boundary conditions
#=========================================================
for i in range(fes.nLower):
   row = fes.lowerDof[i];
   xy  = fes.lowerCoords[i]; #x=xy[0],y=xy[1]
   LHM[row,:]   = 0.
   LHM[row,row] = 1.
   RHV[row]     = math.sin(a*xy[0])*math.sin(b*xy[1])


#=========================================================
# Upper boundary conditions
#=========================================================
for i in range(fes.nUpper):
   row = fes.upperDof[i];
   xy  = fes.upperCoords[i]; #x=xy[0],y=xy[1]
   LHM[row,:]   = 0.
   LHM[row,row] = 1.
   RHV[row]     = math.sin(a*xy[0])*math.sin(b*xy[1])


#=========================================================
print ("Solving the system")
#=========================================================
#fes.printMatVec(LHM,RHV,"afterConstraints")
solVec = np.linalg.solve(LHM, RHV)


#=========================================================
# Compute the error by comparing the exact solution
# to the computed solution at the vertices
#=========================================================
sumsq = 0.;
uexact = np.zeros(fes.sysDim);
for i in range(mesh.nVert):
  xy        = mesh.getVertCoords(i);
  uexact[i] = math.sin(a*xy[0])*math.sin(b*xy[1]);
  sumsq    += (solVec[i]-uexact[i])*(solVec[i]-uexact[i])

print ("\n--------------------------------------------");
print ("Mesh: nVert=",mesh.nVert,"nElem=",mesh.nElem);
print ("Refinment ratio=",n);
print ("RMS Error =",math.sqrt(sumsq/mesh.nVert));
print ("--------------------------------------------\n");


#=========================================================
# Plot the results
#=========================================================
fig = plt.figure(figsize=(14,8))
ax1 = plt.subplot2grid((2,4), (0,0), rowspan=1, colspan=4)
sp1 = fes.plotSoln(ax1,solVec,"Solution")
ax2 = plt.subplot2grid((2,4), (1,0), rowspan=1, colspan=4)
sp2 = fes.plotSoln(ax2,uexact-solVec,"Error")
fig.colorbar(sp1,ax=ax1)
fig.colorbar(sp2,ax=ax2)
#plt.savefig('ndt.png',dpi=250)
plt.show()

#=========================================================
# Estimation of the order of accuracy
#=========================================================

# Mesh refinement factors
n_values = np.array([1, 2, 4, 8])

# The RMS errors you already computed (from your original code)
errors = np.array([0.00394, 0.0010160694164109234, 0.00026494222394222804, 6.647262133732484e-05])

# Calculate log values for plotting
log_n_inv = np.log10(1.0 / n_values)
log_errors = np.log10(errors)

# Print the values to verify
print("n values:", n_values)
print("Error values:", errors)
print("log(1/n) values:", log_n_inv)
print("log(Error) values:", log_errors)

# Linear regression to find the slope (order of accuracy)
slope, intercept = np.polyfit(log_n_inv, log_errors, 1)

# Create convergence plot
plt.figure(figsize=(10, 6))
plt.plot(log_n_inv, log_errors, 'o-', color='blue', linewidth=2, markersize=8, label='Numerical Results')
plt.plot(log_n_inv, slope * log_n_inv + intercept, '--', color='red', linewidth=2, 
         label=f'Slope = {slope:.2f}')

# Add reference slopes
x_range = np.linspace(min(log_n_inv) - 0.1, max(log_n_inv) + 0.1, 100)
for order in [1, 2, 3]:
    # Shift the reference line to pass through the last data point
    ref_intercept = log_errors[-1] - order * log_n_inv[-1]
    plt.plot(x_range, order * x_range + ref_intercept, ':', linewidth=1, 
             label=f'Reference Order {order}')

plt.xlabel('log₁₀(1/n)', fontsize=12)
plt.ylabel('log₁₀(RMS Error)', fontsize=12)
plt.title('Mesh Refinement Convergence Study', fontsize=14)
plt.grid(True)
plt.legend()

# Add text with the observed order
observed_order = round(slope)
plt.text(0.05, 0.05, f'Observed Order of Accuracy: {slope:.2f} ≈ {observed_order}', 
         transform=plt.gca().transAxes, fontsize=12, 
         bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('convergence_plot.png', dpi=250)
plt.show()

print(f"\nSlope of log-log plot: {slope:.4f}")
print(f"Estimated order of accuracy (to nearest integer): {observed_order}")