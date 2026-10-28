# NFIST

**Normalizing Flow Invertible Solution Transformer** is a neural solution operator for
stochastic mean-field control. It comes from the paper [*Self-supervised In-context Operator
Learning for Stochastic Mean-Field Control*](https://arxiv.org/abs/2608.18282) (S. Gao, M. Zhou, R. Lai).

NFIST maps a task prompt to a time-dependent normalizing flow `x_t = F(x_0, t)`. The prompt
can be distribution parameters or particle clouds. The flow is the identity at `t = 0` and
is exactly invertible, with an analytic log-determinant, so the score `∇ log p_t` comes in
closed form. Training minimizes the stochastic control cost directly and needs no solution
data. After training, one forward pass solves a new task.

## Files

| file | content |
|---|---|
| `nfist.py` | model, score, Fokker–Planck energy |
| `train_soc.py` | example: single-well stochastic optimal control |

## Example

```bash
pip install -r requirements.txt
python train_soc.py
```

This trains on random target wells and prints the relative trajectory error against the
closed-form solution. Lower `--bsz` if you run out of GPU memory.

## Usage

```python
from nfist import NFIST

model = NFIST(dim=2, prompt_dims={'p0': 2, 'p1': 2})
op = model(p0=m0, p1=m1)            # prompt tokens, (bsz, n_tokens, 2)
xt = op.traj(x0, t)                 # x0: (bsz, n_pt, 2), t: (n_time,) -> (bsz, n_time, n_pt, 2)
s = op.score(y, t1, log_p0)         # grad log p_t at y, t1: (bsz,)
```

For particle-cloud prompts, use `NFIST(..., cloud_prompts=('p0', 'p1'))` and pass clouds
of shape `(bsz, n_samples, 2)`.

## Citation

If you use this code, please cite our paper:

```bibtex
@article{gao2026nfist,
  title   = {Self-supervised In-context Operator Learning for Stochastic Mean-Field Control},
  author  = {Gao, Suyi and Zhou, Mo and Lai, Rongjie},
  journal = {arXiv preprint arXiv:2608.18282},
  year    = {2026}
}
```