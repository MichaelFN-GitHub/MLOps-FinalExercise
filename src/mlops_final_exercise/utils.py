import matplotlib.pyplot as plt
import torch


def show_image_and_target(images: torch.Tensor, targets: torch.Tensor, show: bool = True) -> None:
    """Display a grid of images with their corresponding targets.

    Args:
        images: Tensor of images to display.
        targets: Tensor of targets corresponding to the images.
        show: Whether to display the images immediately.
    """
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].squeeze(), cmap="gray")
        ax.set_title(f"Label: {targets[i].item()}")
        ax.axis("off")
    if show:
        plt.show()
