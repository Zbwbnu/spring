import taichi as ti
import math

ti.init(arch=ti.gpu)

grid_size = 20
num_particles = grid_size * grid_size
dt = 0.01
ks = 1000.0
kd = 1
gravity = ti.Vector([0, -9.8, 0])
max_speed = 10.0

pos = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
vel = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
force = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)

max_springs = num_particles * 4
spring_a = ti.field(dtype=ti.i32, shape=max_springs)
spring_b = ti.field(dtype=ti.i32, shape=max_springs)
spring_len = ti.field(dtype=ti.f32, shape=max_springs)
num_springs = ti.field(dtype=ti.i32, shape=())

indices = ti.field(dtype=ti.i32, shape=6 * (grid_size - 1) * (grid_size - 1))

@ti.kernel
def init_particles():
    for i in range(num_particles):
        x = i % grid_size
        y = i // grid_size
        pos[i] = ti.Vector([x * 0.1, -y * 0.1, 0.0])
        vel[i] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def init_springs():
    num_springs[None] = 0
    for i in range(num_particles):
        x = i % grid_size
        y = i // grid_size
        if x < grid_size - 1:
            j = i + 1
            l = (pos[i] - pos[j]).norm()
            idx = ti.atomic_add(num_springs[None], 1)
            spring_a[idx] = i
            spring_b[idx] = j
            spring_len[idx] = l
        if y < grid_size - 1:
            j = i + grid_size
            l = (pos[i] - pos[j]).norm()
            idx = ti.atomic_add(num_springs[None], 1)
            spring_a[idx] = i
            spring_b[idx] = j
            spring_len[idx] = l

@ti.kernel
def init_indices():
    for i, j in ti.ndrange(grid_size - 1, grid_size - 1):
        idx = (i * (grid_size - 1) + j) * 6
        a = i * grid_size + j
        b = i * grid_size + j + 1
        c = (i + 1) * grid_size + j
        d = (i + 1) * grid_size + j + 1
        indices[idx + 0] = a
        indices[idx + 1] = b
        indices[idx + 2] = c
        indices[idx + 3] = b
        indices[idx + 4] = d
        indices[idx + 5] = c

@ti.func
def compute_forces_on(i: ti.i32):
    force[i] = gravity
    force[i] += -kd * vel[i]
    for k in range(num_springs[None]):
        a = spring_a[k]
        b = spring_b[k]
        if a == i or b == i:
            x1 = pos[a]
            x2 = pos[b]
            dir = x1 - x2
            dist = dir.norm()
            l0 = spring_len[k]
            f = -ks * (dist - l0) * dir / dist
            if a == i:
                force[i] += f
            else:
                force[i] -= f

@ti.func
def clamp_velocity(v: ti.template()):
    speed = v.norm()
    if speed > max_speed:
        v = v / speed * max_speed
    return v

@ti.kernel
def step_explicit():
    for i in range(num_particles):
        if i % grid_size == 0:
            continue
        compute_forces_on(i)
        acc = force[i]
        vel[i] += acc * dt
        vel[i] = clamp_velocity(vel[i])
        pos[i] += vel[i] * dt

@ti.kernel
def step_semi_implicit():
    for i in range(num_particles):
        if i % grid_size == 0:
            continue
        compute_forces_on(i)
        acc = force[i]
        vel[i] += acc * dt
        vel[i] = clamp_velocity(vel[i])
        pos[i] += vel[i] * dt

@ti.kernel
def step_implicit_iter():
    old_pos = pos
    old_vel = vel
    for _ in range(5):
        for i in range(num_particles):
            if i % grid_size == 0:
                continue
            compute_forces_on(i)
            acc = force[i]
            vel[i] = old_vel[i] + acc * dt
            vel[i] = clamp_velocity(vel[i])
            pos[i] = old_pos[i] + vel[i] * dt

init_particles()
init_springs()
init_indices()

window = ti.ui.Window("Mass-Spring Cloth", (800, 600), vsync=True)
canvas = window.get_canvas()
scene = window.get_scene()
camera = ti.ui.Camera()
camera.position(1, -1, 5)
camera.lookat(1, -1, 0)

current_method = 0
paused = False

while window.running:
    if not paused:
        if current_method == 0:
            step_explicit()
        elif current_method == 1:
            step_semi_implicit()
        else:
            step_implicit_iter()

    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.LMB)
    scene.set_camera(camera)
    scene.point_light(pos=(0, 3, 5), color=(1, 1, 1))
    scene.mesh(pos, indices, color=(0.1, 0.2, 0.8), two_sided=True)
    scene.particles(pos, radius=0.03, color=(1, 0, 0))
    canvas.scene(scene)
    window.show()