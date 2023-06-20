import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def plot(filename, xx, ff, tt, kk, rr):
    # Set the size of the figure to Full HD (1920x1080 pixels)
    fig = plt.figure(figsize=(19.2, 10.8))

    labels = ['focal length', 'theta', 'scale', 'residual']
    for i, y in enumerate([ff, tt, kk, rr]):
        ax = fig.add_subplot(230 + i + 1)
        ax.plot(xx, y)
        ax.set_xlabel('iteration')
        ax.set_ylabel(labels[i])

    # Adjust the spacing between the subplots
    plt.subplots_adjust(wspace=0.2, hspace=0.1)

    # Save the figure to a file in PNG format with a DPI of 300
    plt.savefig(filename, dpi=300)


def scatter(filename, truth, xs, ys, result, model):
    m = []
    N = len(truth)
    f, theta, ks = result[0], result[1], result[2:]
    for i in range(N):
        modelled = model.model(
            0, theta, f, np.pi, np.pi/4, xs[i], ys[i])
        m.append(modelled*ks[i])

        image = model.generate_image(0, theta, f, np.pi, np.pi/4)*ks[i]
        image = image / np.max(image) * 255
        print(i)
        Image.fromarray(image.astype(np.uint8)).save(
            f'../images/{filename}-{i}.jpg')

    m = np.array(m)
    truth = np.array(truth)

    fig = plt.figure(figsize=(19.2, 10.8))

    if len(m.shape) == 3:
        new_shape = [m.shape[0]*m.shape[1], m.shape[2]]
        m = m.reshape(new_shape)
        truth = truth.reshape(new_shape)

        for i in range(truth.shape[-1]):
            ax = fig.add_subplot(131 + i)
            ax.scatter(truth[:, i], m[:, i])
            ax.set_xlabel('true')
            ax.set_ylabel('modelled')

    if len(m.shape) == 2:
        ax = fig.add_subplot(131)
        ax.scatter(truth, m)
        ax.set_xlabel('true')
        ax.set_ylabel('modelled')

    plt.subplots_adjust(wspace=0.3)
    plt.savefig(filename, dpi=300)
