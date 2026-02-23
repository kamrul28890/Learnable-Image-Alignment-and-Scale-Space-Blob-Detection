import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def draw_all_circles(
    image: np.ndarray, 
    cx: list[int] | np.ndarray, 
    cy: list[int] | np.ndarray, 
    rad: list[int] | np.ndarray, 
    filename: str,
    color: str='r'
) -> None:
    """
    image: numpy array, representing the grayscsale image
    cx, cy: numpy arrays or lists, centers of the detected blobs
    rad: numpy array or list, radius of the detected blobs
    filename: output filename
    color: circle color, default to 'r' (red)
    """
    if image.shape[0] == 1:
        image = np.concatenate([image, image, image], 0)
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(image, cmap='gray')
    for x, y, r in zip(cx, cy, rad):
        circ = Circle((x, y), r, color=color, fill=False)
        ax.add_patch(circ)

    plt.title('%i circles' % len(cx))
    plt.savefig(filename)

