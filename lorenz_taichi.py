import argparse
import taichi as ti

p = argparse.ArgumentParser()
p.add_argument("-g", "--gpu", action="store_true", help="try to compile code for the GPU")
p.add_argument("-n", "--normalize", action="store_true", help="normalize screen brightness per frame")
args = p.parse_args()

ti.init(arch=ti.gpu if args.gpu else ti.cpu)

sigma = 10.0
beta = 8.0 / 3.0
rho = 28.0

XSIZE = 1280
YSIZE = 720

imgR = ti.field(dtype=ti.f32, shape=(XSIZE, YSIZE))
imgG = ti.field(dtype=ti.f32, shape=(XSIZE, YSIZE))

state = ti.Vector.field(3, dtype=ti.f32, shape=2)  # state[0] = P, state[1] = Q
vshow = ti.Vector.field(3, dtype=ti.f32, shape=(XSIZE, YSIZE))  # Taichi field for display

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

@ti.func
def plot_func(x: ti.f32, y: ti.f32, img: ti.template()):
    for p in range(100):
        ix = int((x + 20) / 40.0 * XSIZE + grand())
        iy = int((y / 50.0) * YSIZE + grand())
        if 0 <= ix < XSIZE and 0 <= iy < YSIZE:
            img[ix, iy] += 1.0

@ti.kernel
def plot(x: ti.f32, y: ti.f32, img: ti.template()):
    plot_func(x, y, img)

@ti.kernel
def spinup_kernel(h: ti.f32, steps: ti.i32):
    P = state[0]
    for i in range(steps):
        P = rk4(P, h)
    state[0] = P
    Q = state[1]
    for i in range(steps):
        Q = rk4(Q, h)
    state[1] = Q

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
        plot_func(P[0], P[2], imgR)
        Q = rk4(Q, h)
        plot_func(Q[0], Q[2], imgG)
        t += h
    state[0] = P
    state[1] = Q

@ti.kernel
def find_max(imgR: ti.template(), imgG: ti.template()) -> ti.f32:
    m = 0.0
    for y, x in imgR:
        m = max(m, imgR[y, x])
        m = max(m, imgG[y, x])
    return m

@ti.kernel
def normalize_and_gamma(imgR: ti.template(), imgG: ti.template(), vshow: ti.template(), m: ti.f32):
    for y, x in imgR:
        # Avoid division by zero
        r = min(max(imgR[y, x] / m, 0.0), 1.0) if m > 1e-6 else 0.0
        g = min(max(imgG[y, x] / m, 0.0), 1.0) if m > 1e-6 else 0.0
        # Gamma correction
        vshow[y, x][0] = r ** 0.4545
        vshow[y, x][1] = g ** 0.4545
        vshow[y, x][2] = 0.0

def main():
    global args
    gui = ti.GUI('Lorenz Attractor', res=(XSIZE, YSIZE), fast_gui=True)
    P0 = ti.Vector([1.0, 1.0, 1.0])
    Q0 = ti.Vector([1.0, 1.0, 1.000001])
    set_initial_states(P0, Q0)
    h = 0.001
    l = 0.05
    spinup_kernel(0.1, 100)
    while gui.running:
        fade()
        march_and_plot(h, l)
        if args.normalize:
            m = find_max(imgR, imgG)
        else:
            m = 50.
        normalize_and_gamma(imgR, imgG, vshow, m)
        gui.set_image(vshow)
        gui.show()

if __name__ == '__main__':
    main()
