import argparse
import taichi as ti

#
#    __                        __       _     __   _ 
#   / /__  _______ ___  ___   / /____ _(_)___/ /  (_)
#  / / _ \/ __/ -_) _ \/_ /  / __/ _ `/ / __/ _ \/ / 
# /_/\___/_/  \__/_//_//__/__\__/\_,_/_/\__/_//_/_/  
#                        /___/                       
#
# A port of my original C code which was pretty crufty to use python/taichi
# Ported via a conversation with copilot/chatgpt4.1 as an experiment.
# 
# I wasn't tremendously familiar with taichi, but while there were some false
# starts with copilot having an understanding which wasn't much beyond my own,
# it was not difficult to eventually reach a port of this code which operates
# entirely via taichi fields for maximal perfomance, and can be used on 
# either the CPU or GPU.
#
# Copyright 2025, Mark VandeWettering <mvandewettering@gmail.com>
#

p = argparse.ArgumentParser()
p.add_argument("-g", "--gpu", action="store_true", help="try to compile code for the GPU")
p.add_argument("-n", "--normalize", action="store_true", help="normalize screen brightness per frame")
p.add_argument("-f", "--fullscreen", action="store_true", help="create a full screen window")
p.add_argument("-s", "--spotsize", type=float, default=1.5, help="spot size (default %(default)s")
args = p.parse_args()

ti.init(arch=ti.gpu if args.gpu else ti.cpu)

sigma = 10.0
beta = 8.0 / 3.0
rho = 28.0

if args.fullscreen:
    XSIZE, YSIZE = 1920, 1080
else:
    XSIZE, YSIZE = 1280, 720


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

#
# This was the primitive version that I originally did in C
#
@ti.func
def grand():
    t = 0.0
    for i in range(16):
        t += ti.random(ti.f32) - 0.5
    return t

@ti.func
def gauss_box_muller(stddev: ti.f32):
    u1 = ti.random(ti.f32)
    u2 = ti.random(ti.f32)
    eps = 1e-8
    r = ti.sqrt(-2.0 * ti.log(u1 + eps))
    theta = 2.0 * ti.math.pi * u2
    z = r * ti.cos(theta)
    return z * stddev 

@ti.kernel
def fade():
    for y, x in imgR:
        imgR[y, x] *= 0.9
        imgG[y, x] *= 0.9

@ti.func
def erf(x):
    # Constants
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p  = 0.3275911

    sign = 1.0
    if x < 0:
        sign = -1.0
    x = ti.abs(x)

    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * ti.exp(-x * x)

    return sign * y

@ti.func
def gaussian_pixel_integral(xc: ti.f32, yc: ti.f32, px: ti.i32, py: ti.i32, sigma: ti.f32) -> ti.f32:
    # xc, yc: center of the gaussian in pixel coordinates
    # px, py: pixel indices
    # sigma: stddev in pixel units

    sqrt2 = ti.sqrt(2.0)
    # Pixel covers [px, px+1), [py, py+1)
    x0 = (px - xc) / sigma
    x1 = (px + 1 - xc) / sigma
    y0 = (py - yc) / sigma
    y1 = (py + 1 - yc) / sigma

    ix = 0.5 * (erf(x1 / sqrt2) - erf(x0 / sqrt2))
    iy = 0.5 * (erf(y1 / sqrt2) - erf(y0 / sqrt2))
    return ix * iy

@ti.func
def plot_func(x: ti.f32, y: ti.f32, img: ti.template(), sigma: ti.f32 = 2.0, window: ti.i32 = 6):
    # Compute Gaussian center in pixel coordinates
    cx = (x + 20) / 40.0 * XSIZE
    cy = (y / 50.0) * YSIZE
    # Integrate over a window of pixels around the center
    for dx in range(-window, window + 1):
        for dy in range(-window, window + 1):
            ix = int(cx + dx)
            iy = int(cy + dy)
            if 0 <= ix < XSIZE and 0 <= iy < YSIZE:
                g = gaussian_pixel_integral(cx, cy, ix, iy, sigma)
                img[ix, iy] += g

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
def march_and_plot(h: ti.f32, l: ti.f32, spotsize: ti.f32):
    t = 0.0
    P = state[0]
    Q = state[1]
    while t < l:
        P = rk4(P, h)
        plot_func(P[0], P[2], imgR, sigma=spotsize)
        Q = rk4(Q, h)
        plot_func(Q[0], Q[2], imgG, sigma=spotsize)
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
    mode = "GPU" if args.gpu else "CPU"
    gui = ti.GUI(f'Lorenz Attractor ({mode})', res=(XSIZE, YSIZE), fast_gui=True, fullscreen=args.fullscreen)
    P0 = ti.Vector([1.0, 1.0, 1.0])
    Q0 = ti.Vector([1.0, 1.0, 1.000001])
    set_initial_states(P0, Q0)
    h = 0.001
    l = 0.05
    spinup_kernel(0.1, 100)
    while gui.running:
        fade()
        march_and_plot(h, l, args.spotsize)
        if args.normalize:
            m = find_max(imgR, imgG)
        else:
            m = 1.
        normalize_and_gamma(imgR, imgG, vshow, m)
        gui.set_image(vshow)
        gui.show()

        if gui.get_event(ti.GUI.ESCAPE):
            break

if __name__ == '__main__':
    main()
