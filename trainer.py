import io
import os

import wandb

from PIL import Image
import matplotlib.animation as ani
import matplotlib.pyplot as plt
from rich.console import Console
import torch
import torch.optim as Optim
from torchvision import transforms
from torchvision.io import write_video

from integrator import Integrator
from model import DynamicsModel


def plot_to_tensor(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', )
    buf.seek(0)
    image = Image.open(buf)
    image = transforms.ToTensor()(image)
    plt.close()
    return image


def integrate(A: torch.Tensor, A_hat: torch.Tensor, initial: torch.Tensor):
    # dt = 1e-3
    dt, steps = 0.01, 50
    x, _ = Integrator.integrate(A, initial, method='rk4', dt=dt, steps=steps)
    x_hat, _ = Integrator.integrate(A_hat, initial, method='rk4', dt=dt, steps=steps)
    return x.cpu(), x_hat.cpu()


def plot_integrate(x: torch.Tensor, x_hat: torch.Tensor, initial: torch.Tensor, record: bool = False):

    fig = plt.figure(figsize=(10, 10))
    fig.suptitle('Learning Linear ODE')

    ax = fig.add_subplot()
    ax.grid()

    ax.scatter(initial.cpu()[:, 0], initial.cpu()[:, 1], color='cyan')
    x_plot = ax.plot(x[:, :, -2], x[:, :, -1], color='red')
    x_hat_plot = ax.plot(x_hat[:, :, -2], x_hat[:, :, -1], color='blue')

    def limits(x, x_hat):
        return min(x.min(), x_hat.min(), -2), max(x.max(), x_hat.max(), 2)

    frames = [plot_to_tensor(fig)] if record else None

    def animate(i: int, A: torch.Tensor, A_hat: torch.Tensor, initial: torch.Tensor):
        x, x_hat = integrate(A, A_hat, initial)
        for i in range(len(initial)):
            x_plot[i].set_data(x[:, i, -2], x[:, i, -1])
            x_hat_plot[i].set_data(x_hat[:, i, -2], x_hat[:, i, -1])
        plt.pause(1e-3)

        lim = limits(x, x_hat)
        ax.set_xlim(lim)
        ax.set_ylim(lim)

        if frames is not None:
            frames.append(plot_to_tensor(fig))

        return x_plot, x_hat_plot

    return fig, animate, frames


if __name__ == "__main__":

    log = record = False
    # log = record = True

    if log:
        wandb.init('ode')

    console = Console()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    console.log(device)

    steps = int(1e5)
    samples = 16
    batch_size = 256
    eps = 1e-5

    # A = torch.tensor([[0.5, -1], [1, -1]], device=device)
    A = torch.tensor([[4, -2], [3, -3]], device=device)
    # A = torch.tensor([[4, 2], [1, 3]], device=device)

    model = DynamicsModel(len(A), device=device)
    optim = Optim.Adam(model.parameters(), lr=1e-3)

    test_initial = 2 * (torch.rand(samples, len(A), device=device) - 0.5)
    x, x_hat = integrate(A, model.A.detach(), test_initial)
    fig, animate, frames = plot_integrate(x, x_hat, test_initial, record=record)

    plt.ion()
    plt.show(block=False)

    for step in range(steps):

        t = torch.rand(batch_size, device=device).sort().values
        initial = (torch.rand(samples, len(A), device=device) - 0.5)

        x = (A[None] * t[:, None, None]).exp()
        x = (x[None] @ initial[:, None, :, None]).squeeze()

        x_hat = model(t, initial)

        loss = ((x_hat - x)**2).mean()

        optim.zero_grad()
        loss.backward()
        optim.step()

        diff = ((model.A.detach() - A)**2).mean()

        if step % 100 == 0:
            console.log(f'{step} {loss.item():5.7f} {diff.item():5.7f}')
            animate(step, A, model.A.detach(), test_initial)
            if log:
                wandb.log({'loss': loss.item(), 'diff': diff.item()})

        if diff <= eps:
            break

    if frames is not None:
        frames = torch.stack(frames)
        console.log(frames.shape, frames.min(), frames.max())
        # torch.save(frames, f'animation/frames.pt')

        frames = (frames * 255).to(torch.uint8)
        frames = frames.detach().cpu().numpy()
        wandb.log({'video': wandb.Video(frames)})

    console.log(A)
    console.log(model.A.detach())
