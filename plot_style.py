import matplotlib.pyplot as plt

def set_style():

    plt.style.use("default")

    plt.rcParams.update({

        "figure.figsize": (10,6),

        "figure.dpi":150,
        "savefig.dpi":300,

        "font.size":12,
        "axes.titlesize":14,
        "axes.labelsize":12,

        "legend.fontsize":10,

        "xtick.labelsize":10,
        "ytick.labelsize":10,

        "axes.grid":True,
        "grid.alpha":0.3,

        "lines.linewidth":2,

        "savefig.bbox":"tight"

    })