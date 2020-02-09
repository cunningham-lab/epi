import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], "ro")


def init():
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(-1, 1)
    return (ln,)


def update(frame):
    xdata.append(frame)
    ydata.append(np.sin(frame))
    ln.set_data(xdata, ydata)
    return (ln,)


ani = animation.FuncAnimation(
    fig, update, frames=np.linspace(0, 2 * np.pi, 128), init_func=init, blit=True
)

Writer = animation.writers["ffmpeg"]
writer = Writer(fps=30, metadata=dict(artist="Me"), bitrate=1800)

ani.save("temp.mp4", writer=writer)
