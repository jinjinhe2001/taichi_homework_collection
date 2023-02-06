# reference ==> https://www.shadertoy.com/view/3sySRKs

import taichi as ti
import handy_shader_functions as hsf

ti.init(arch = ti.metal)

res_x = 512
res_y = 512
screen_x = 6
screen_y = 6
resolution_rate = res_x / res_y
pixels = ti.Vector.field(3, ti.f32, shape=(res_x, res_y))

@ti.func
def opSmoothUnion(d1, d2, k):
    h = hsf.clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0)
    return hsf.lerp(d2, d1, h) - k * h * (1.0 - h)

@ti.func
def sdSphere(p, s):
    return p.norm() - s;

@ti.func
def mapPoint(p, t):
    d = 2.0
    for k in range(16):
        fk = float(k)
        time = t * (hsf.fract(fk * 412.531 + 0.513) - 0.5) * 2.0
        d = opSmoothUnion(sdSphere(p + ti.sin(time + fk * ti.Vector([52.5126, 64.62744, 632.25])) * \
                        ti.Vector([2.0, 2.0, 0.8]), hsf.lerp(0.5, 1.0, hsf.fract(fk * 412.531 + 0.5124))), \
                          d, 0.4)
    return d


@ti.func
def calcNormal(p, t):
    h = 1e-5
    xyy = ti.Vector([1, -1, -1])
    yyx = ti.Vector([-1, -1, 1])
    yxy = ti.Vector([-1, 1, -1])
    xxx = ti.Vector([1, 1, 1])
    retVector = (xyy * mapPoint(p + xyy * h, t) + \
            yyx * mapPoint(p + yyx * h, t) + \
            yxy * mapPoint(p + yxy * h, t) + \
            xxx * mapPoint(p + xxx * h, t))
    return retVector.normalized()

@ti.kernel
def render(t:ti.f32):
    # draw something on your canvas
    for i,j in pixels:
        u = i / float(res_x)
        v = j / float(res_y)
        rayOri = ti.Vector([(u - 0.5) * resolution_rate * screen_x, (v - 0.5) * resolution_rate * screen_y, 3.0])
        rayDir = ti.Vector([0.0, 0.0, -1.0])
        depth = 0.0;
        p = ti.Vector([0.0, 0.0, 0.0])
        for r in range(64):
            p = rayOri + rayDir * depth
            dist = mapPoint(p, t)
            depth += dist
            if dist < 1e-6:
                break

        depth = ti.min(6.0, depth)
        n = calcNormal(p, t)
        b = ti.max(0.0, n.dot(ti.Vector([0.577, 0.577, 0.577])))
        color = (0.5 + 0.5 * ti.cos(b + t * 3.0 + ti.Vector([u, v, u]) * 2.0 + ti.Vector([0.0, 2.0, 4.0]))) * (0.85 + b * 0.35)
        color *= ti.exp(-depth * 0.15)
        pixels[i,j] = color

gui = ti.GUI("Canvas", res=(res_x, res_y))

for i in range(100000):
    t = i * 0.03
    render(t)
    gui.set_image(pixels)
    gui.show()