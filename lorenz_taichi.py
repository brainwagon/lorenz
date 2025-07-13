import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

# Lorenz parameters
sigma = 10.0
beta = 8.0 / 3.0
rho = 28.0

XSIZE = 1280
YSIZE = 720

# Taichi fields for two color channels
imgR = ti.field(dtype=ti.f32, shape=(YSIZE, XSIZE))
imgG = ti.field(dtype=ti.f32, shape=(YSIZE, XSIZE))

# Store states as Taichi fields (P and Q)
state = ti.Vector.field(3, dtype=ti.f32, shape=2)  # state[0] = P, state[1] = Q

@ti.func
def lorenz(P):
    dP = ti.Vector([0.0, 0.0, 0.0])
    dP[0] = sigma * (P[1] - P[0])
    dP[1] = P[0] * (rho - P[2]) - P[1]
    dP[2] = P[0] * P[1] - beta * P[2]
    return dP

@ti.func
def rk4(P, h):
    k1 = lorenz(P)
    k2 = lorenz(P + 0.5 * h * k1)
    k3 = lorenz(P + 0.5 * h * k2)
    k4 = lorenz(P + h * k3)
    return P + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

@ti.func
def grand():
    t = 0.0
    for i in range(16):
        t += ti.random(ti.f32) - 0.5
    return t

@ti.kernel
def fade():
    for y, x in imgR:
        imgR[y, x] *= 0.9
        imgG[y, x] *= 0.9

@ti.kernel
def plot(x: ti.f32, y: ti.f32, img: ti.template()):
    for p in range(100):
        ix = int((x + 20) / 40.0 * XSIZE + grand())
        iy = int((y / 50.0) * YSIZE + grand())
        if 0 <= ix < XSIZE and 0 <= iy < YSIZE:
            img[iy, ix] += 1.0

@ti.kernel
def spinup_kernel(h: ti.f32, steps: ti.i32):
    for i in range(steps):
        state[0] = rk4(state[0], h)

@ti.kernel
def set_initial_states(P0: ti.types.vector(3, ti.f32), Q0: ti.types.vector(3, ti.f32)):
    state[0] = P0
    state[1] = Q0

@ti.kernel
def march_and_plot(h: ti.f32, l: ti.f32):
    t = 0.0
    P = state[0]
    Q = state[1]
    while t < l:
        P = rk4(P, h)
        plot(P[0], P[2], imgR)
        Q = rk4(Q, h)
        plot(Q[0], Q[2], imgG)
        t += h
    state[0] = P
    state[1] = Q

def main():
    gui = ti.GUI('Lorenz Attractor', res=(XSIZE, YSIZE), fast_gui=True)
    # Initial conditions
    P0 = ti.Vector([1.0, 1.0, 1.0000001])
    Q0 = ti.Vector([1.0, 1.0, 1.0 + 1e-4])
    # Set initial states
    set_initial_states(P0, Q0)
    h = 0.0001
    l = 0.05
    # Spin up to attractor
    spinup_kernel(0.1, 1000)
    while gui.running:
        fade()
        march_and_plot(h, l)
        # Normalize to [0,1] for display, gamma correct (≈pow(t, 0.4545))
        mR = np.max(imgR.to_numpy())
        mG = np.max(imgG.to_numpy())
        m = max(mR, mG, 1e-6)
        show = np.zeros((YSIZE, XSIZE, 3), dtype=np.float32)
        ir = imgR.to_numpy() / m
        ig = imgG.to_numpy() / m
        show[..., 0] = np.clip(ir, 0, 1) ** 0.4545
        show[..., 1] = np.clip(ig, 0, 1) ** 0.4545
        gui.set_image(show)
        gui.show()

if __name__ == '__main__':
    main()