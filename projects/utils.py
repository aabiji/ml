import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from IPython.display import clear_output
from IPython import display

def random_pad_crop(img_batch, pad, flip_percentage):
    padded = F.pad(img_batch, (pad, pad, pad, pad), mode="constant", value=0)

    img_size = (img_batch.shape[-2], img_batch.shape[-1])
    padded_size = (padded.shape[-2], padded.shape[-1])

    for sample in range(img_batch.shape[0]):
        xy = (
            torch.randint(padded_size[0] - img_size[0], ()).item(),
            torch.randint(padded_size[1] - img_size[1], ()).item())

        img_batch[sample] = padded[sample, :, xy[0]:xy[0]+img_size[0], xy[1]:xy[1]+img_size[1]]

        flip = torch.randint(100, ()).item() < (100 * flip_percentage)
        if flip:
            img_batch[sample] = torch.flip(img_batch[sample], dims=[2])


def random_cutout(img_batch, percentage, num_cuts, cut_size):
    img_size = (img_batch.shape[-2], img_batch.shape[-1])

    for sample in range(img_batch.shape[0]):
        cutout = torch.randint(100, ()).item() < (100 * percentage)
        if not cutout:
            continue

        for _ in range(num_cuts):
            xy = (
                torch.randint(img_size[0] - cut_size, ()).item(),
                torch.randint(img_size[1] - cut_size, ()).item())
            img_batch[sample, :, xy[0]:xy[0]+cut_size, xy[1]:xy[1]+cut_size] = 0.0


def classification_error(prediction, actual):
    probabilities = torch.softmax(prediction, dim=1)
    answer = torch.argmax(probabilities, dim=1)
    num_correct = (answer == actual).sum().item()
    batch_size = answer.shape[0]
    return 100 * (batch_size - num_correct) / batch_size


def plot_stats(fig, ax, title, errors=None, losses=None):
    clear_output(wait=True)
    ax.clear()

    if losses is not None:
        ax.plot(losses, "b-", label=f"Loss {losses[-1]:.3f}")

    if errors is not None:
        ax.plot(errors, "r-", label=f"Error {errors[-1]:.2f}%")

    ax.autoscale_view(scalex=True, scaley=True)
    ax.set_title(title)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Stats")
    plt.legend()
    display.display(fig)
