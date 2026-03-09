import numpy as np
from numpy import array, zeros

def EulerRichardson(f, t, y, dt, p):
    k1 = f(t, y, p)
    y_mid = y + 0.5 * dt * k1
    k2 = f(t + 0.5 * dt, y_mid, p)
    y_new = y + dt * k2

    return y_new

def total_energy(y, p):
    G = p['G']
    m = p['m']
    n = len(m)
    d = p['dimension']

    E = np.zeros(y.shape[0])

    for t in range(y.shape[0]):
        KE = 0.0
        PE = 0.0

        for i in range(n):
            # KE_i = 1/2 * m_i * |v_i|^2
            v = y[t, n*d + i*d : n*d + (i+1)*d]
            KE += 0.5 * m[i] * np.dot(v, v)

            # PE only counts each pair once (upper triangle)
            for j in range(i+1, n):
                ri = y[t, i*d:(i+1)*d]
                rj = y[t, j*d:(j+1)*d]
                rij = np.linalg.norm(ri - rj)
                PE -= G * m[i] * m[j] / rij

        E[t] = KE + PE

    return E

## Runge-Kutta 4 from last assignment ##
def RK4(f, t, y, dt, p):
    k1 = dt * f(t, y, p)                      # slope at start
    k2 = dt * f(t + 0.5*dt, y + 0.5*k1, p)    # slope at midpoint using k1
    k3 = dt * f(t + 0.5*dt, y + 0.5*k2, p)    # slope at midpoint using k2
    k4 = dt * f(t + dt, y + k3, p)            # slope at end using k3
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6.0  # midpoint slopes weighted double

## Runge-Kutta 45 ##
def solve_ode_rk45(f, t_span, y0, p, atol=1e-9, rtol=1e-9, h_init=0.01):

    # a_ij coefficients
    a21 = 1/5
    a31, a32 = 3/40, 9/40
    a41, a42, a43 = 44/45, -56/15, 32/9
    a51, a52, a53, a54 = 19372/6561, -25360/2187, 64448/6561, -212/729
    a61, a62, a63, a64, a65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656

    # 5th order weights (b_i), b7=0
    b1, b2, b3, b4, b5, b6 = 35/384, 0, 500/1113, 125/192, -2187/6784, 11/84

    # 4th order weights (b_i*)
    bs1, bs2, bs3, bs4, bs5, bs6, bs7 = 5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40

    # c values (time offsets for each stage)
    c2, c3, c4, c5, c6 = 1/5, 3/10, 4/5, 8/9, 1

    ## Parameters ##
    safety = 0.9
    minscale = 0.2      # step can't shrink by more than 5x
    maxscale = 10.0      # step can't grow by more than 10x
    beta = 0.0           # (textbook default's to zero) PI control?
    alpha = 0.2 - beta * 0.75  # = 1/5 for 5th order method when beta=0
    reject = False       # tracks whether the previous step was rejected
    errold = 1.0e-4      # (error from previous accepted step)

    ## Initialize ##
    t = t_span[0]
    t_end = t_span[1]
    y = np.array(y0, dtype=float)
    h = h_init

    t_list = [t]
    y_list = [y.copy()]

    ## Main integration loop ##
    while t < t_end:
        if t + h > t_end:
            h = t_end - t

        # 6 stages
        k1 = h * f(t, y, p)
        k2 = h * f(t + c2*h, y + a21*k1, p)
        k3 = h * f(t + c3*h, y + a31*k1 + a32*k2, p)
        k4 = h * f(t + c4*h, y + a41*k1 + a42*k2 + a43*k3, p)
        k5 = h * f(t + c5*h, y + a51*k1 + a52*k2 + a53*k3 + a54*k4, p)
        k6 = h * f(t + c6*h, y + a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5, p)

        # 5th order solution (in equation 17.2.4 b7=0 so k7 doesn't contribute)
        y_new = y + b1*k1 + b2*k2 + b3*k3 + b4*k4 + b5*k5 + b6*k6

        # 7th stage: evaluated at y_new 
        k7 = h * f(t + h, y_new, p)

        # Error estimate
        delta = (b1-bs1)*k1 + (b2-bs2)*k2 + (b3-bs3)*k3 + \
                (b4-bs4)*k4 + (b5-bs5)*k5 + (b6-bs6)*k6 - bs7*k7

        # Scale factor (17.2.8)
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))

        # RMS error norm (17.2.9)
        err = np.sqrt(np.mean((delta / scale)**2))

        if err <= 1.0:
            # Step accepted
            t = t + h
            y = y_new
            t_list.append(t)
            y_list.append(y.copy())

            # Calculate scale factor for next step size (17.2.12 & 17.2.13)
            if err == 0.0:
                s = maxscale
            else:
                s = safety * err**(-alpha) * errold**beta
                s = max(minscale, min(maxscale, s))

            # Don't let step increase if last step was rejected
            if reject:
                h = h * min(s, 1.0)
            else:
                h = h * s

            errold = max(err, 1.0e-4)  # bookkeeping for PI controller
            reject = False

        else:
            # Step rejected, reduce step size
            s = max(safety * err**(-alpha), minscale)
            h = h * s
            reject = True

    return np.array(t_list), np.array(y_list)


def solve_ode(f, t_span, y0, method, p, first_step=0.01):

    # If method is rk45, use the adaptive solver instead
    if method == 'rk45' or method == solve_ode_rk45:
        return solve_ode_rk45(f, t_span, y0, p, h_init=first_step)

    dt = first_step
    t_start, t_end = t_span

    n_steps = int((t_end - t_start) / dt)
    t_array = np.linspace(t_start, t_start + n_steps * dt, n_steps + 1)

    y_array = zeros((len(t_array), len(y0)))
    y_array[0] = y0

    y = array(y0, dtype=float)
    for i in range(1, len(t_array)):
        y = method(f, t_array[i - 1], y, dt, p)
        y_array[i] = y

    return t_array, y_array