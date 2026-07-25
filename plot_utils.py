import os
import matplotlib.pyplot as plt

def save_figure(folder, filename):

    os.makedirs(folder, exist_ok=True)

    path = os.path.join(folder, filename)

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Saved -> {path}")