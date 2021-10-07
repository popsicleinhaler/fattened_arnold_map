import matplotlib.pyplot as plt
import numpy as np
import math

print('working')

def one_iterate(t_n, T_n, tau_0, a, k, T_osc):
    t = (T_n + t_n) % T_osc
    T = tau_0 * (1 - a) + a * T_n + k * math.sin(2 * math.pi * t / T_osc)

    return T, t


def arnold_jacobian(a, k, tau_0, T_osc, x, y):
    dfdx = math.cos(2 * math.pi * x / T_osc) * 2 * math.pi / (T_osc)
    dfdy = a
    dgdx = 1
    dgdy = 1
    return [[dfdx, dfdy], [dgdx, dgdy]]


def arnold_tangent(a, k, tau_0, T_osc, dy, dx, x, y):
    c = math.pi * 2 / T_osc
    return [dy * a + dx * k * c * math.cos(c * x), dy + dx]


def arnold_find_lyapunov(transient=125, iterN=1000, a=0, k=4, tau_0=0, T_osc=3, initx=0, inity=2):
    # Estimate the LCEs
    # The number of iterations to throw away
    nTransients = transient
    # The number of iterations to over which to estimate
    nIterates = iterN
    # Initial condition

    xState = initx
    yState = inity
    # Initial tangent vectors
    e1x = 0
    e1y = 1
    # Iterate away transients and let the tangent vectors align
    #    with the global stable and unstable manifolds
    for n in range(0, nTransients):
        xState, yState = one_iterate(xState, yState, tau_0, a, k, T_osc)
        # Evolve tangent vector for maxLCE
        e1x, e1y = arnold_tangent(a, k, tau_0, T_osc, e1x, e1y, xState, yState)
        # Normalize the tangent vector's length
        d = math.sqrt(e1x * e1x + e1y * e1y)
        e1x = e1x / d
        e1y = e1y / d

    # Okay, now we're ready to begin the estimation
    # This is essentially the same as above, except we accumulate estimates
    # We have to set the min,max LCE estimates to zero, since they are sums
    maxLCE = 0.0
    for n in range(0, nIterates):
        # Get next state
        xState, yState = one_iterate(xState, yState, tau_0, a, k, T_osc)
        # Evolve tangent vector for maxLCE
        e1x, e1y = arnold_tangent(a, k, tau_0, T_osc, e1x, e1y, xState, yState)
        # Normalize the tangent vector's length
        d = math.sqrt(e1x * e1x + e1y * e1y)
        e1x = e1x / d
        e1y = e1y / d
        # Accumulate the stretching factor (tangent vector's length)
        maxLCE += math.log(d, 2)

    maxLCE = maxLCE / float(nIterates)

    return maxLCE

def heatmap2d(arr: np.ndarray,name='lyapunov image'):
    plt.imshow(arr, cmap='viridis')
    plt.colorbar()
    plt.ylabel("K")
    plt.xlabel("T_0")
    plt.xticks([0,len(arr[0])-1],[tau0,taun])
    plt.yticks([0,len(arr)-1],[kn,k0])
    plt.show()
    plt.savefig(name+'.png')


# control knobs
a = 0
k = 4
tau_0 = 0
T_osc = 3
initx = 0
inity = 2
transient = 125
iterN = 1000
k0=0
kn=1
tau0=1
taun=2
inter=500
taus = np.linspace(tau0, taun, inter)
ks = np.linspace(k0, kn, inter)

# color = []
# for k in ks:
#     rows = []
#     for tau in taus:
#         le = arnold_find_lyapunov(transient, iterN, a, k, tau, T_osc, initx, inity)
#         if le > 1:
#             le =1
#         elif le < -1:
#             le = -1
#         elif abs(le-0)<0.001:
#             le=0
#         rows.append(le)
#     color.append(rows)
# colorr = list(reversed(color))
# print('done')

toscs = np.linspace(0,24,48)
tdcolor=[]
for T_osc in toscs:
    color = []
    for k in ks:
        rows = []
        for tau in taus:
            rows.append(arnold_find_lyapunov(transient, iterN, a, k, tau, T_osc, initx, inity))
        color.append(rows)
    colorr = list(reversed(color))
    tdcolor.append(colorr)
    heatmap2d(colorr, name='lyapunov diagram for tosc='+str(a))
print('done')






