"""Train NFIST on single-well stochastic optimal control (SOC).

Task: p0 = N(0, I), terminal cost V(x) = |x - m1|^2 with the well m1 uniform in a
ball of radius 15. The prompts are the means of p0 and of the target, and training
is self-supervised: no reference solutions are used. The learned operator is then
checked against the closed-form optimal paths.
"""
import argparse
import torch
from nfist import NFIST, fp_energy

DIM, N_PT, N_TIME, GAMMA, RADIUS = 2, 100, 16, 0.1, 15.0


def sample_tasks(bsz, device):
    v = torch.randn(bsz, 1, DIM, device=device)
    r = RADIUS * torch.rand(bsz, 1, 1, device=device) ** (1 / DIM)
    m1 = r * v / v.norm(dim=-1, keepdim=True)              # well, uniform in the ball
    x0 = torch.randn(bsz, N_PT, DIM, device=device)        # particles from p0
    return x0, {'p0': torch.zeros_like(m1), 'p1': m1}


def log_p0(x):
    return -0.5 * x.pow(2).sum(-1)


def soc_loss(model, x0, prompts):
    t = torch.linspace(0, 1, N_TIME, device=x0.device)
    xt, energy = fp_energy(model(**prompts), x0, t, GAMMA, log_p0)
    terminal = (xt[:, -1] - prompts['p1']).pow(2).sum(-1).mean(-1)
    return (0.5 * energy + 0.5 * terminal).mean()


def exact_soc(x0, m1, t):
    """Optimal probability-flow paths for p0 = N(0, I): x_t = m_t + s_t * x0."""
    t = t[:, None, None]
    m_t = 0.5 * t * m1[:, None]
    s_t = (0.25 * (2 - t) ** 2 + GAMMA * t * (2 - t)).sqrt()
    return m_t + s_t * x0[:, None]


@torch.no_grad()
def rel_error(model, x0, prompts, n_time=32):
    """Relative trajectory L2 error vs. the exact solution, averaged over time and tasks."""
    t = torch.linspace(0, 1, n_time, device=x0.device)
    xt, ref = model(**prompts).traj(x0, t), exact_soc(x0, prompts['p1'], t)
    return ((xt - ref).norm(dim=-1).mean(-1) / ref.norm(dim=-1).mean(-1)).mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--iters', type=int, default=50000)
    p.add_argument('--bsz', type=int, default=16)
    p.add_argument('--out', default='nfist_soc.pt')
    args = p.parse_args()

    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = NFIST(DIM, prompt_dims={'p0': DIM, 'p1': DIM}).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    test = sample_tasks(32, device)

    for it in range(1, args.iters + 1):
        loss = soc_loss(model, *sample_tasks(args.bsz, device))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it % 1000 == 0 or it == args.iters:
            print(f'iter {it:6d}  loss {loss.item():.4f}  rel. error {rel_error(model, *test):.2e}')

    torch.save(model.state_dict(), args.out)


if __name__ == '__main__':
    main()
