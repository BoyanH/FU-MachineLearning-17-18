import os


def save_plot(fig, path):
    fig.savefig(os.path.join(os.path.dirname(__file__), path))
