from rich.console import Console
import torch
import torch.optim as Optim

import matplotlib.pyplot as plt
import matplotlib.animation as ani
from model import DynamicsModel
from integrator import Integrator


def integrate(A: torch.Tensor, A_hat: torch.Tensor, initial: torch.Tensor):
    # dt = 1e-3
    dt, steps = 0.05, 100
    x, _ = Integrator.integrate(A, initial, method='rk4', dt=dt, steps=steps)
    x_hat, _ = Integrator.integrate(A_hat, initial, method='rk4', dt=dt, steps=steps)
    return x.cpu(), x_hat.cpu()


def plot_integrate(x: torch.Tensor, x_hat: torch.Tensor, initial: torch.Tensor):

    fig = plt.figure()
    plt.grid()

    plt.scatter(initial.cpu()[:, 0], initial.cpu()[:, 1], color='cyan')
    x_plot = plt.plot(x[:, :, -2], x[:, :, -1], color='red')
    x_hat_plot = plt.plot(x_hat[:, :, -2], x_hat[:, :, -1], color='blue')

    def limits(x, x_hat):
        return min(x.min(), x_hat.min(), -2), max(x.max(), x_hat.max(), 2)

    def animate(i: int, A: torch.Tensor, A_hat: torch.Tensor, initial: torch.Tensor):
        x, x_hat = integrate(A, A_hat, initial)
        for i in range(len(initial)):
            x_plot[i].set_data(x[:, i, -2], x[:, i, -1])
            x_hat_plot[i].set_data(x_hat[:, i, -2], x_hat[:, i, -1])
        plt.pause(1e-3)

        plt.xlim(limits(x[:, :, -2], x_hat[:, :, -2]))
        plt.ylim(limits(x[:, :, -1], x_hat[:, :, -1]))

        return x_plot, x_hat_plot

    return fig, animate


if __name__ == "__main__":

    console = Console()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    console.log(device)

    steps = int(1e5)
    samples = 8
    batch_size = 256

    A = torch.tensor([[0.5, -1], [1, -1]], device=device)
    # A = torch.tensor([[4, -2], [3, -3]], device=device)
    # A = torch.tensor([[4, 2], [1, 3]], device=device)

    model = DynamicsModel(len(A), device=device)
    optim = Optim.Adam(model.parameters(), lr=1e-3)

    test_initial = 2 * (torch.rand(samples, len(A), device=device) - 0.5)
    x, x_hat = integrate(A, model.A.detach(), test_initial)
    fig, animate = plot_integrate(x, x_hat, test_initial)

    # plt.ion()
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

        if diff <= 1e-5:
            break

    console.log(A)
    console.log(model.A.detach())
