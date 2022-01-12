import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, TextArea, AnchoredOffsetbox, VPacker


def plot_axhline(kwargs):
    """
    plot a circle in the image

    Parameters
    ----------
    kwargs: dict
        any valid attribute of plt.avhline function
    """
    plt.axhline(**kwargs)


def plot_axvline(kwargs):
    """
    plot a circle in the image

    Parameters
    ----------
    kwargs: dict
        any valid attribute of plt.avvline function
    """
    plt.avvline(**kwargs)


def plot_circle(kwargs):
    """
    plot a circle in the image

    Parameters
    ----------
    kwargs: dict
        any valid attribute of plt.circle function
    """
    circle1 = plt.Circle(**kwargs)
    ax = plt.gca()
    ax.add_patch(circle1)


def create_watermark(kwargs):
    """
    plot a circle in the image

    Parameters
    ----------
    kwargs: dict
        keys should be:
        image_path: path containing image for water mark
        label: label to add to image
        alpha: controls alpha parameter of the image
    """
    keys = [i for i in kwargs.keys()]
    image_path = kwargs['image_path']
    label = kwargs['label'] if 'label' in keys else 'image'
    alpha = kwargs['alpha'] if 'alpha' in keys else 0.3

    ax = plt.gca()
    img = plt.imread(image_path)
    imagebox = OffsetImage(img, alpha=alpha, zoom=0.2)
    textbox = TextArea(label, textprops=dict(alpha=alpha))
    packer = VPacker(children=[imagebox, textbox], mode='fixed',
                     pad=0, sep=0, align='center')
    ao = AnchoredOffsetbox('lower left', pad=0, borderpad=0, child=packer)
    ax.add_artist(ao)
