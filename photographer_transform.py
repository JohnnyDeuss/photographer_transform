from argparse import ArgumentParser
import os
import sys
from tkinter import filedialog

import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider
import numpy as np


DEFAULT_HUE = 136           # Roughly cyan.
DEFAULT_EXPONENT = 1.5      # More contrast, but not too much.
DEFAULT_SATURATION = 245    # Near full saturation.


def interface(bgr_image):
    """Create an interface for the given BGR image, providing the user with controls to update the model parameters with a live preview."""
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, bottom=0.25)
    ax.margins(x=0)
    image = photographer_transform(
        bgr_image,
        exp=DEFAULT_EXPONENT,
        hue=DEFAULT_HUE,
        saturation=DEFAULT_SATURATION)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_plot = plt.imshow(image)

    ax_hue = plt.axes((0.1, 0.15, 0.8, 0.03))
    ax_saturation = plt.axes((0.1, 0.11, 0.8, 0.03))
    ax_exponent = plt.axes((0.1, 0.07, 0.8, 0.03))

    slider_hue = Slider(ax_hue, 'Hue', 0, 180, DEFAULT_HUE, valstep=1)
    slider_saturation = Slider(ax_saturation, 'Sat', 0, 255, DEFAULT_SATURATION, valstep=1)
    slider_exponent = Slider(ax_exponent, 'Exp', 0, 20, DEFAULT_EXPONENT)

    def update(val):
        """Update the interface's image."""
        image = photographer_transform(
            bgr_image,
            exp=slider_exponent.val,
            hue=slider_hue.val,
            saturation=slider_saturation.val)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_plot.set_data(image)
        fig.canvas.draw_idle()
    
    slider_hue.on_changed(update)
    slider_saturation.on_changed(update)
    slider_exponent.on_changed(update)

    def save_image(val):
        """Show a save dialog and save the generated image."""
        file_name = filedialog.asksaveasfilename(filetypes=[('PNG', '.png')])
        if file_name:
            image = photographer_transform(
                bgr_image,
                exp=slider_exponent.val,
                hue=slider_hue.val,
                saturation=slider_saturation.val)
            cv2.imwrite(file_name, image)

    reset_ax = plt.axes([0.8, 0.02, 0.1, 0.04])
    button = Button(reset_ax, 'Save', hovercolor='0.975')
    button.on_clicked(save_image)
    plt.show()


def power_transform(a, exp):
    """Perform a power transformation on the given image. The image is expected to be a np.uint8 array, with values ranging between 0 and 255.
    The values will be scaled down to the range 0 to 1, transformed using the formula `p_{x,y}=p{x,y}^exp`, and then scaled back up again.
    """
    return np.round(255*(a/255)**exp).astype(np.uint8)


def remove_background(bgr_image, k=8):
    """Remove background from an image that has a somewhat solid background color, replacing it with
    solid black. k is the number of clusters to use in k-means.
    """
    # Quantize the image, so that we can find areas of solid colour, even when pixels aren't exactly the same.
    # Create a color feature vector from the image and perform k-means on it.
    z = bgr_image.reshape((-1, 3)).astype(np.float32)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 8
    _, labels_1d, center = cv2.kmeans(z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    labels_1d = labels_1d.flatten()
    labels_2d = labels_1d.reshape(bgr_image.shape[:2])

    # Make a mask to select 5 pixels along the edges and find the most common label, which we will assume to be the background.
    edge_mask = np.ones(bgr_image.shape[:2], dtype=bool)
    edge_mask[5:-5, 5:-5] = 0
    # Now find the most common label along the edges, which we assume is the background image.
    labels_2d[edge_mask]
    background_label = np.argmax(np.bincount(labels_2d[edge_mask]))
    foreground_mask = (labels_2d != background_label).astype(np.uint8)

    # Now pick the largest component in the mask as being the bivalve.
    contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    bivalve_mask = np.zeros(foreground_mask.shape, dtype=np.uint8)
    cv2.drawContours(bivalve_mask, [largest_contour], -1, 255, -1)
    # Now dilate and erode the mask, to close any openings seeping into the interior of the foreground image.
    bivalve_mask = cv2.dilate(bivalve_mask, None, iterations=10)
    bivalve_mask = cv2.erode(bivalve_mask, None, iterations=10)
    bgr_image[~bivalve_mask.astype(bool)] = (0, 0, 0)
    return bgr_image


def photographer_transform(bgr_image, exp=1, hue=DEFAULT_HUE, saturation=DEFAULT_SATURATION, k=8):
    """Perform a transformation of the given BGR image, using various image processing techniques.
    
    Parameters:
    - bgr_image: The BGR image to process.
    - exp: The exponent to be used in the power transformation step. See `power_transform`.
    - hue: The hue to use for every pixel in the HLS version of the image.
    - saturation: The saturation to use for every pixel in the HLS version of the image.
    - k: The number of clusters to use in the background removal process.
    """
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.equalizeHist(gray_image)
    gray_image = power_transform(gray_image, exp)

    # Use the grayscale as the L channel of HSL. 
    gray_image = np.expand_dims(gray_image, -1)
    hls_image = np.pad(gray_image, ((0, 0), (0, 0), (1, 1)))

    hls_image[:, :, 0] = hue
    hls_image[:, :, 2] = saturation

    # Convert HSL to BGR so we can save it.
    filename = os.path.splitext(sys.argv[1])[0]
    bgr_image = cv2.cvtColor(hls_image, cv2.COLOR_HLS2BGR_FULL)
    return bgr_image


def cli():
    """Provide a CLI to the pixelate function."""
    parser = ArgumentParser(description='Transform an image to be more aesthetically pleasing.')
    parser.add_argument('input', help='The path to the input image')
    parser.add_argument('-r', '--remove_background', help='Indicate whether to use background removal', action='store_true')
    args = parser.parse_args()
    if not os.path.exists(args.input):
        raise FileNotFoundError(f'Input file "{args.input}" could not be found.')
    return args


if __name__ == '__main__':
    args = cli()
    image = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if args.remove_background:
        image = remove_background(image)
    interface(image)
