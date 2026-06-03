# Arena — Nico's Workthrough

> **For Claude**: this README exists so a fresh instance can immediately context-load and help effectively. Read everything before responding to any arena question.

---

## Who Nico is (relevant context)

- MIT freshman, Course 6-4 (AI & Decision Making), rising sophomore
- Currently in CDMX (June 1 – Aug 8) for MISTI at Universidad Panamericana, working on ML-based control for a simulated chameleon robot in MuJoCo (Python)
- Long-term goal: VLA research, targeting internship at Anthropic or Physical Intelligence by junior year
- Immediate goal: UROP with Tedrake's Robot Locomotion Group or Pulkit Agrawal's Improbable AI lab sophomore year
- MAIA officer and SPAR are fall 2026 targets — arena ch4 (alignment science) is the chapter that matters most for those
- Working block: 6–8pm CDMX time on weekdays (after MISTI lab)
- Machine: M4 Pro MacBook — use `device='mps'`, set `PYTORCH_ENABLE_MPS_FALLBACK=1` for pytorch. VastAI GPU needed for ch1+ heavy training

---

## Repo structure

```
arena/                          ← this repo (github.com/ncorios/arena)
├── README.md                   ← this file
├── ch0_fundamentals/
│   ├── notes.md
│   ├── prereqs.py
│   ├── ray_tracing.py
│   ├── cnns.py
│   ├── optimization.py
│   ├── backprop.py
│   └── vaes_gans.py
├── ch1_transformer_interp/
│   ├── notes.md
│   └── ...
├── ch2_rl/
│   ├── notes.md
│   └── ...
├── ch3_evals/
│   ├── notes.md
│   └── ...
└── ch4_alignment_science/
    ├── notes.md                ← highest priority notes file
    └── ...
```

**answer files** live here. the actual arena repo (callummcdougall/ARENA_3.0) is cloned separately on Nico's machine — this repo only contains his answers and notes, not callum's files (tests.py, solutions.py, utils.py etc).

---

## Git setup

- `origin` → `github.com/ncorios/arena` (push here)
- `upstream` → `github.com/callummcdougall/ARENA_3.0` (pull updates from here)

```bash
git pull upstream main   # get arena updates
git push origin main     # push nico's work
```

---

## How Nico is working through arena

**workflow per section:**
- read instructions on `learn.arena.education` (not the .ipynb files)
- write answers in a new `.py` file using `# %%` cell delimiters (VS Code interactive mode)
- import `tests`, `utils` from the cloned arena repo for test functions
- never edit existing arena repo files (tests.py, solutions.py, utils.py)

**option chosen:** VS Code + blank `.py` files (Option 2 from setup page) — not Colab, not notebooks

**mac-specific pytorch setup:**
```python
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
```
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python answers.py
```
Add `multiprocessing_context="fork"` to any DataLoader to avoid crashes.

---

## Chapter map & priority

| Chapter | Topic | Priority | Compute needed |
|---|---|---|---|
| **0.0** | Prerequisites — einops, einsum, tensor manipulation | do all of it | mac fine |
| **0.1** | Ray tracing — batched matrix ops, render 3D pikachu | do fully | mac fine |
| **0.2** | CNNs & ResNets — nn.Module, training loops, ResNet34 | do fully | mac mostly fine |
| **0.3** | Optimization — SGD/Adam from scratch, W&B, distributed training | do s1+s2, skip distributed | needs GPU for training |
| **0.4** | Backpropagation — build autograd from scratch | do fully, go slow | mac fine (numpy-based) |
| **0.5** | VAEs & GANs — generative models | do, use VastAI | needs GPU |
| **1.x** | Transformer interpretability — circuits, induction heads | go deep | needs GPU |
| **2.x** | RL | solid, not priority | needs GPU |
| **3.x** | LLM evals | solid, not priority | needs GPU |
| **4.x** | **Alignment science** | **highest priority destination** | needs A100 for some |

**target**: reach ch4 with real understanding by mid-July 2026.

---

## Nico's background coming in

**strong:**
- 18.03 (ODEs, Fourier, PDEs, Lyapunov) — strong performance
- 18.02 (multivariable calc) — solid, minor gaps in curl/divergence subscripts
- 8.02 (E&M) — fine
- 6.190 (intro ML/Python) — knows python well, some ML exposure
- PID control theory — built full PID crash course, coded from scratch (see `github.com/ncorios/pid-arm`)
- MuJoCo/robotics context from MISTI

**gaps coming into arena:**
- numpy: hasn't done explicit numpy work, needs 1-2hrs warmup before ch0 exercises
- pytorch: new — ch0 teaches it but he'll be slower at first
- einops/einsum: totally new, ch0.0 section 2 covers it

**prereq questions he should be able to answer before ch1:**
- what is a `torch.Tensor`? (n-dim array with autograd)
- what is `nn.Module`? (base class, handles params + forward pass)
- what does `.backward()` do? (computes + stores gradients via backprop)
- what is a loss function? (maps predictions + labels → scalar)
- what does an optimizer do? (uses gradients to update weights)
- what is a hyperparameter vs parameter?

---

## How Claude should help

**during exercises:**
- when Nico finishes a section, quiz him hard — conceptual + implementation questions strictly from what he just covered. no softballs. surface gaps.
- use the arena exercise format as reference: difficulty/importance ratings matter, don't let him skip high-importance exercises
- if he's stuck >15 mins on one exercise, tell him to read the solution and move on (arena's own advice)
- lowercase casual prose for chat; proper formatting + code blocks for technical content

**pacing:**
- don't encourage speedrunning — the goal is ch4 with real understanding by mid-July, not checking boxes
- ch4 alignment science is the destination; everything before it is foundation-building
- MAIA/SPAR applications need original artifacts (MISTI project + arena ch4 understanding), not just curriculum completion

**what not to do:**
- don't explain things he already knows (18.02/18.03 math, basic python, git basics)
- don't pad explanations — he moves fast and asks follow-ups when needed
- don't suggest he use Colab or notebooks — he's doing VS Code + .py files

---

## Current status

*Update this section as chapters are completed.*

- [ ] ch0.0 prereqs
- [ ] ch0.1 ray tracing
- [ ] ch0.2 CNNs & ResNets
- [ ] ch0.3 optimization
- [ ] ch0.4 backpropagation
- [ ] ch0.5 VAEs & GANs
- [ ] ch1 transformer interp
- [ ] ch2 RL
- [ ] ch3 evals
- [ ] ch4 alignment science

**last worked on:** setting up repo, git remotes configured, about to start ch0.0

---

## Other active projects (for context)

- **pid-arm** (`github.com/ncorios/pid-arm`, private) — PID controller built from scratch in numpy. theory notes in `notes/pid-theory.md`. next steps: low-pass filter, Ziegler-Nichols tuning, discretization.
- **MISTI chameleon robot** — MuJoCo simulation, ML control, at Universidad Panamericana in Hiram Ponce's lab. arc: PID baseline → ML controller → real robot deployment.
- **ncorios.github.io** — personal site, hand-coded HTML

---

*last updated: June 2026, start of MISTI*
