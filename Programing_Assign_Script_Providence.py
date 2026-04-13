#--- Programing Assignment---

import numpy as np
import matplotlib.pyplot as plt

#---SETTINGS---
L = 2 * np.pi         # Interval length [0, 2π]
N = 10        # 100   # Number of Fourier modes                                 
Points = 1001 # 2001  # Number of grid points (Odd Nr of points required for Simpson's rule)                         

#--- FUNCTIONS---
# Test functions:  
def function_linear(x):         # f(x) = x           
    return x 
 
def function_trig(x):           # f(x)=sin⁡(x)+0.5 cos⁡(3x)        
    return np.sin(x) + 0.5 * np.cos(3 * x)

#--- SIMPSON'S RULE---
def simpson(y, dx): 
 
    # Integral ≈ dx/3 [first + 4*odd + 2*even + last]                  
    return dx/3 * (y[0] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-1:2]) + y[-1])   


#--- COMPUTE FOURIER COEFFICIENTS --- 
# Analytical Fourier coefficients
def fourier_analytical(func):
    
    # Evenly spaced x-values on [0, L]. 
    x  = np.linspace(0, L, Points)           
    y  = func(x)
    
    # dx is the uniform spacing between x number of points.
    dx = x[1] - x[0]                        
    
    # Compute the constant coefficient a0:
    # a0 = (1/L) * integral from 0 to L of f(x) dx
    a0 = simpson(y, dx) / L
    
    # Create empty arrays to store the sine and cosine coefficients.
    A  = np.zeros(N)
    B  = np.zeros(N)
    
    for n in range(1, N + 1):
        
        # Angular frequency for this mode.
        w = 2 * np.pi * n / L
        
        # Compute An:
        # An = (2/L) * integral of f(x) sin(wx) dx
        A[n-1] = (2/L) * simpson(y * np.sin(w*x), dx)
        
        # Compute Bn:
        # Bn = (2/L) * integral of f(x) cos(wx) dx
        B[n-1] = (2/L) * simpson(y * np.cos(w*x), dx)
                 
    return x, y, a0, A, B


# Discrete Fourier coefficients
def fourier_discrete(y, dx):
    x = np.arange(len(y)) * dx
    a0 = np.sum(y) * dx / L
    
    A = np.zeros(N)
    B = np.zeros(N)
    
    # Compute the first N Fourier modes.
    for n in range(1, N + 1):
        w = 2*np.pi * n/L
        A[n-1] = (2/L) * np.sum(y * np.sin(w*x)) * dx
        B[n-1] = (2/L) * np.sum(y * np.cos(w*x)) * dx
        
    return x, a0, A, B


#--- RECONSTRUCTION OF FOURIER SERIES--- 
def reconstruct(x, a0, A, B):
    
    # Start with the constant term a0 at every x-value.
    y = np.full_like(x, a0)
    
    # Add each sine and cosine term from n = 1 to N. 
    for n in range(1, len(A)+1):
        
        # Angular frequency for the n-th mode.
        w = 2*np.pi * n/L
        y += A[n-1]*np.sin(w*x) + B[n-1]*np.cos(w*x)
     
    return y


#--- PLOT ORIGINAL FUNCTION AND FOURIER RECONSTRUCTION ---
def plot(x, y, y_rec, title):
    plt.figure(figsize=(10,5))
    plt.plot(x, y, label="Original function", linewidth=2)
    plt.plot(x, y_rec, "--", label=f"Fourier reconstruction (N={N})", linewidth=2)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()     
    
    
#--- MAIN EXECUTION ---
# Part 1A: Linear Function 
x1, y1, a0_1, A1, B1 = fourier_analytical(function_linear)
y1_rec = reconstruct(x1, a0_1, A1, B1)
plot(x1, y1, y1_rec, "Part 1A: f(x) = x")

# Part 1B: Trigonometric function
x2, y2, a0_2, A2, B2 = fourier_analytical(function_trig)
y2_rec = reconstruct(x2, a0_2, A2, B2)
plot(x2, y2, y2_rec, "Part 1B: f(x) = sin(x) + 0.5cos(3x)")

# Part 2: Discrete Noisy Data 
n_samples = 200   # 500                                    
x_d = np.linspace(0, L, n_samples, endpoint=False)
dx = x_d[1] - x_d[0]

# Use a fixed seed so results are reproducible.
rng = np.random.default_rng(42)

# Sampled trigonometric signal plus noise.
y_d = np.sin(x_d) + 0.5*np.cos(3*x_d) + 0.08*rng.normal(size=n_samples)

x_d, a0_d, A_d, B_d = fourier_discrete(y_d, dx)
y_d_rec = reconstruct(x_d, a0_d, A_d, B_d)

plot(x_d, y_d, y_d_rec, "Part 2: Discrete Data")  








   