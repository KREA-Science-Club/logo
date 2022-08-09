from unittest import skip
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import animation

from crop import crop_to_square

def lorenz_solver(inic, t0 = 0, tf = 60, step = 0.001, sigma = 10, rho = 28, beta = 8/3):
    def xdot(x, y, z):
        return sigma * (y - x)

    def ydot(x, y, z):
        return x * (rho - z) - y

    def zdot(x, y, z):
        return x * y - beta * z

    def dot(t, vars):
        return [xdot(*vars), ydot(*vars), zdot(*vars)]

    t = np.arange(t0, tf, step)
    sol = solve_ivp(dot, [t0, tf], inic, t_eval = t)

    return sol.y

def generate_attractor(inics, filename="", crop_to_sq = False, show = True,
                        colors = ["gold", "saddlebrown"], show_text=True, text_color="goldenrod", 
                        bg_black = True, transparent = False):
    scale = 40

    if bg_black:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')
    
    plt.figure(figsize=(76.80, 43.20))

    plt.xlim(-scale, scale)
    plt.ylim(0, 50)
    plt.axis('off')

    sols = [lorenz_solver(inic) for inic in inics]

    for i, sol in enumerate(sols):
        x, y, z = sol
        plt.plot(x, z, color=colors[i % len(colors)], linewidth=0.5)

    if show_text:
        plt.text(0, 0, "THE SCIENCE CLUB", fontsize=30, fontweight=0, color=text_color, 
                    horizontalalignment="center", fontfamily="serif")

    if filename:
        plt.savefig(filename, transparent = transparent)

        if crop_to_sq:
            crop_to_square(filename)
    
    if show:
        plt.show()

def animate_logo(inics, filename="", show = True, colors = ["gold", "saddlebrown"], show_text=True, text_color="goldenrod", bg_black = True, 
                    frames = 6000, freeze_frames=150, batch_size = 20, interval = 1):
    scale = 40

    if bg_black:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')

    plt.figure(figsize=(19.20, 10.80))

    plt.xlim(-scale, scale)
    plt.ylim(0, 50)
    plt.axis('off')

    sols = [lorenz_solver(inic) for inic in inics]

    step = len(sols[0][0]) // frames # 60000/6000
    x_anims = [x[::step] for x, y, z in sols]
    z_anims = [z[::step] for x, y, z in sols]

    paths = [plt.plot([], [], color=colors[i % len(colors)], linewidth=0.5)[0] for i in range(len(inics))]

    if show_text:
        title_text = "THE SCIENCE CLUB"
        title_frame_multiple = (frames // batch_size) // len(title_text)

        title = plt.text(0, 0, "THE SCIENCE CLUB", fontsize=30, fontweight=0, 
                                color=text_color, horizontalalignment="center", fontfamily="serif")

        patches = [*paths, title]
    else:
        patches = [*paths]
    
    def init():
        for i in range(len(paths)):
            path = paths[i]
            path.set_data(x_anims[i][:1], z_anims[i][:1])

        return patches

    def animate(frame):
        if frame <= (frames // batch_size):
            for i in range(len(paths)):
                path = paths[i]
                path.set_data(x_anims[i][:frame * batch_size], z_anims[i][:frame * batch_size])
            
            if show_text:
                if frame % title_frame_multiple == 0:
                    title.set_text(title_text[:frame // title_frame_multiple])
        
        return patches

    anim = animation.FuncAnimation(plt.gcf(), animate, init_func=init, frames = (frames // batch_size) + freeze_frames, 
                                    interval = interval, blit = True)
    
    if filename:
        anim.save(filename, fps=60)

    if show:
        plt.show()

inics = []

# inic = np.random.uniform(0, 10, 3)
# print(f"Point: {list(inic)}")
# inics.append(inic)

# inics.append([9.49657258, -5.5307689, 8.94601363])
# inics.append([0.1884141, 3.18549037, 9.14596038])
inics.append([6.05681151749362, 5.4634096640009275, 9.255399642632])
inics.append([5.172675812338344, 4.359837599867456, 8.641490104471675])

def color_trial():
    colors = ["#000B8D", "#1D19AC", "#47C7FC"]

    import os
    
    if not os.path.exists("color_trial"):
        os.mkdir("color_trial")
    
    for color in colors:
        for color2 in colors:
            generate_attractor(inics, bg_black=False, colors=[color, color2],
                                filename=f"color_trial\\{color}-and-{color2}.png", show = False)

# color_trial()

def generate_logos():

    import os
    
    logo_path = "logo_images"
    if not os.path.exists(logo_path):
        os.mkdir(logo_path)

    generate_attractor(inics, filename=os.path.join(logo_path, "science_club_logo.png"), crop_to_sq=True)
    generate_attractor(inics, filename=os.path.join(logo_path, "science_club_logo_transparent.png"), 
                        crop_to_sq=True, transparent=True)

    generate_attractor(inics, filename=os.path.join(logo_path, "science_club_logo_light.png"), 
                        crop_to_sq=True, colors=["#000B8D", "#47C7FC"], 
                        text_color="#3494DE", bg_black=False)
    generate_attractor(inics, filename=os.path.join(logo_path, "science_club_logo_light_transparent.png"), 
                        crop_to_sq=True, colors=["#000B8D", "#47C7FC"], 
                        text_color="#3494DE", bg_black=False, transparent=True)

    generate_attractor(inics, filename=os.path.join(logo_path, "science_club_logo_light_on_black.png"), 
                        crop_to_sq=True, colors=["#000B8D", "#47C7FC"], 
                        text_color="#3494DE")

def generate_logos_without_text():

    import os
    
    logo_path = "logo_images_without_text"
    if not os.path.exists(logo_path):
        os.mkdir(logo_path)

    generate_attractor(inics, filename=os.path.join(logo_path, "science_club_logo.png"), crop_to_sq=True, show_text=False)
    generate_attractor(inics, filename=os.path.join(logo_path, "science_club_logo_transparent.png"), 
                        crop_to_sq=True, transparent=True, show_text=False)

    generate_attractor(inics, filename=os.path.join(logo_path, "science_club_logo_light.png"),
                        crop_to_sq=True, colors=["#000B8D", "#47C7FC"], 
                        text_color="#3494DE", bg_black=False, show_text=False)
    generate_attractor(inics, filename=os.path.join(logo_path, "science_club_logo_light_transparent.png"), 
                        crop_to_sq=True, colors=["#000B8D", "#47C7FC"], 
                        text_color="#3494DE", bg_black=False, transparent=True, show_text=False)

    generate_attractor(inics, filename=os.path.join(logo_path, "science_club_logo_light_on_black.png"), 
                        crop_to_sq=True, colors=["#000B8D", "#47C7FC"], 
                        text_color="#3494DE", show_text=False)


def generate_logo_animations():
    import os
    
    anim_path = "logo_animations" # add option to add text in animations and logo
    if not os.path.exists(anim_path):
        os.mkdir(anim_path)

    animate_logo(inics, filename=os.path.join(anim_path, "science_club_logo.mp4"), show=False)

    animate_logo(inics, filename=os.path.join(anim_path, "science_club_logo_light.mp4"), show=False,
                    colors=["#000B8D", "#47C7FC"], text_color="#3494DE", bg_black=False)

    animate_logo(inics, filename=os.path.join(anim_path, "science_club_logo_light_on_black.mp4"), show=False,
                    colors=["#000B8D", "#47C7FC"], text_color="#3494DE")

def generate_logo_animations_without_text():
    import os
    
    anim_path = "logo_animations_without_text" # add option to add text in animations and logo
    if not os.path.exists(anim_path):
        os.mkdir(anim_path)

    animate_logo(inics, filename=os.path.join(anim_path, "science_club_logo.mp4"), show=False, show_text=False, batch_size=1)

    animate_logo(inics, filename=os.path.join(anim_path, "science_club_logo_light.mp4"), show=False,
                    colors=["#000B8D", "#47C7FC"], text_color="#3494DE", bg_black=False, show_text=False, batch_size=1)

    animate_logo(inics, filename=os.path.join(anim_path, "science_club_logo_light_on_black.mp4"), show=False,
                    colors=["#000B8D", "#47C7FC"], text_color="#3494DE", show_text=False, batch_size=1)

# generate_logos()
# generate_logo_animations()

generate_logos_without_text()
# generate_logo_animations_without_text()