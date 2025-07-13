import taichi as ti
import math
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
    # Approximate sum of [-0.5, 0.5], 16 times, like your C code
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

def main():
    gui = ti.GUI('Lorenz Attractor', res=(XSIZE, YSIZE), fast_gui=True)
    # Initial conditions
    P = np.array([1.0, 1.0, 1.0000001], dtype=np.float32)
    Q = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    h = 0.0001
    l = 0.05
    # "Spin up" to attractor
    for _ in range(1000):
        P = rk4(ti.Vector(P), 0.1).numpy()
    Q[:] = P
    Q[2] += 1e-4

    while gui.running:
        fade()
        # March both particles for l seconds in h steps
        PP = P.copy()
        QQ = Q.copy()
        t = 0.0
        while t < l:
            PP = rk4(ti.Vector(PP), h).numpy()
            plot(PP[0], PP[2], imgR)
            QQ = rk4(ti.Vector(QQ), h).numpy()
            plot(QQ[0], QQ[2], imgG)
            t += h
        P[:] = PP
        Q[:] = QQ
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