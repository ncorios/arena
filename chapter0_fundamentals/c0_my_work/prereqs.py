# %% [markdown]
# ============================================================================
#  ARENA ch0 — prereqs warmup
# ============================================================================
#  Goal: get fluent in the *tooling* before June 8. The math (linear algebra,
#  calculus, prob) is review for you, so it's not here. What IS here:
#       numpy  ->  torch  ->  einops/einsum  ->  a ray-tracing preview
#
#  How to use:
#   - run cell by cell with VS Code's interactive mode (Shift+Enter on a `# %%`)
#   - every `...` is a blank for you to replace. running a cell before you fill
#     it will error loudly — that's intended.
#   - asserts check you automatically. if a cell runs silently past its assert,
#     you got it right. compare prints against the EXPECTED OUTPUT block at the
#     bottom of the file.
#   - if you're stuck > a few mins on one drill, look it up and move on. the
#     point is reps, not suffering.
#
#  Run with MPS fallback on (your M4):  PYTORCH_ENABLE_MPS_FALLBACK=1 python prereqs.py
# ============================================================================

# %%
import numpy as np
import torch as t
import einops
from jaxtyping import Float, Int, Bool
from torch import Tensor

# warmup tensors live on CPU — keep it simple. we'll worry about device later.
device = "mps" if t.backends.mps.is_available() else "cpu"
print("device available:", device)


# %% [markdown]
# ============================================================================
#  SECTION 1 — NUMPY
# ============================================================================
#  This is your weakest area and it underpins everything (0.4 backprop is pure
#  numpy). Four ideas matter most: broadcasting, axis-based reductions,
#  boolean/fancy indexing, and reshape-vs-transpose. Drill them until they're
#  automatic.

# %% [markdown]
# --- 1.1  Broadcasting -------------------------------------------------------
#  Rule: numpy lines arrays up from the RIGHT. A dim is compatible if it's
#  equal, or one of them is 1 (it gets "stretched"). Shapes (3,) and (3,1)
#  broadcast to (3,3).

# %%
a = np.array([1, 2, 3])            # shape (3,)
b = np.array([[10], [20], [30]])   # shape (3, 1)

# TODO: add a and b, relying on broadcasting -> shape (3, 3)
broadcast_sum = a + b

assert broadcast_sum.shape == (3, 3)
assert broadcast_sum[1, 2] == 23
print("1.1", broadcast_sum.tolist())


# %% [markdown]
# --- 1.2  Pairwise distances (the single most useful broadcasting pattern) ---
#  You have 2 points and 3 points. You want a (2,3) matrix of distances between
#  every pair. Trick: insert size-1 axes so the two point-sets broadcast into a
#  (2, 3, 2) array of coordinate differences, then norm over the last axis.
#  This exact move shows up constantly in 0.1.

# %%
pts_a = np.array([[0.0, 0.0], [3.0, 0.0]])                # (2, 2)
pts_b = np.array([[0.0, 4.0], [0.0, 0.0], [6.0, 0.0]])    # (3, 2)

# TODO: differences of every a-point vs every b-point -> shape (2, 3, 2)
#       
diffs = pts_a[:, None, :] - pts_b[None, :, :]   # "a b c -> a 1 b c" minus "a b c -> 1 a b c"
# TODO: Euclidean distance over the last axis -> shape (2, 3)
#       
dists = np.linalg.norm(diffs, axis=-1)

assert diffs.shape == (2, 3, 2)
assert dists.shape == (2, 3)
assert np.allclose(dists, [[4, 0, 6], [5, 3, 3]])
print("1.2", dists.tolist())


# %% [markdown]
# --- 1.3  Axis reductions ----------------------------------------------------
#  `axis=0` collapses rows (gives you a per-column result), `axis=1` collapses
#  columns. `keepdims=True` keeps the collapsed axis as size 1 (great for
#  broadcasting the result back).

# %%
M = np.arange(12).reshape(3, 4)

# TODO: sum down each column (collapse the row axis) -> shape (4,)
col_sums = M.sum(axis= 0)
# TODO: mean across each row -> shape (3,)
row_means = M.mean(axis= 1)
# TODO: max of each column, KEEPING dims -> shape (1, 4)
col_max_keepdims = M.max(axis = 0, keepdims= True)

assert col_sums.shape == (4,) and row_means.shape == (3,)
assert col_max_keepdims.shape == (1, 4)
print("1.3", col_sums.tolist(), row_means.tolist(), col_max_keepdims.tolist())


# %% [markdown]
# --- 1.4  Boolean / fancy indexing -------------------------------------------
#  A boolean mask of the same shape selects elements. Summing a boolean array
#  counts Trues. You can also assign through a mask.

# %%
x = np.array([4, 1, 7, 3, 9, 2, 8])

# TODO: boolean mask of elements strictly greater than 4
mask = x > 4
# TODO: how many elements are > 4 (use the mask, no loop)
count_gt4 = mask.sum()
# TODO: a copy of x where every element > 4 is set to 0, others unchanged
#       hint: np.where(cond, a, b)  OR  copy then mask-assign
clipped = np.where(x > 4, 0, x)
assert int(count_gt4) == 3
assert clipped.tolist() == [4, 1, 0, 3, 0, 2, 0]
print("1.4", int(count_gt4), clipped.tolist())


# %% [markdown]
# --- 1.5  reshape vs transpose (people conflate these) -----------------------
#  reshape REFILLS the data row-by-row into a new shape. transpose SWAPS axes.
#  They give different arrays — make sure you see why below.

# %%
y = np.arange(6)

# TODO: reshape to (2, 3)
y23 = y.reshape(2,3)
# TODO: reshape to (3, 2), using -1 to let numpy infer one dimension
y32 = y.reshape(3,-1)
# TODO: the transpose of y23 (also shape (3,2)) — note it != y32
y23_T = y23.transpose()

assert y23.shape == (2, 3) and y32.shape == (3, 2) and y23_T.shape == (3, 2)
assert not np.array_equal(y32, y23_T)   # the teaching point
print("1.5", y23.tolist(), y32.tolist(), y23_T.tolist())


# %% [markdown]
# ============================================================================
#  SECTION 2 — PYTORCH
# ============================================================================
#  Mostly the same as numpy with renamed bits: torch uses `dim` where numpy
#  uses `axis`. The genuinely new thing is autograd.

# %% [markdown]
# --- 2.1  Tensor creation ----------------------------------------------------

# %%
# TODO: 1-D tensor [0, 1, ..., 9]
ar = t.arange(10)
# TODO: 5 evenly spaced values from 0 to 1 INCLUSIVE
ls = t.linspace(0,1,5)
# TODO: 3x3 identity matrix
I3 = t.eye(3,3)
np_arr = np.array([1.0, 2.0, 3.0])
# TODO: turn np_arr into a torch tensor (sharing memory)
from_np = t.from_numpy(np_arr)

assert ar.tolist() == list(range(10))
assert t.allclose(ls, t.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))
assert t.equal(I3, t.eye(3))
assert from_np.tolist() == [1.0, 2.0, 3.0]
print("2.1", ar.tolist(), ls.tolist(), from_np.tolist())


# %% [markdown]
# --- 2.2  dim vs axis --------------------------------------------------------

# %%
T = t.arange(12).reshape(3, 4).float()

# TODO: sum over dim 0 -> shape (4,)
sum_dim0 = t.sum(T, dim=0)
# TODO: index of the max along dim 1 -> shape (3,)
argmax_dim1 = t.argmax(T, dim=1)

assert sum_dim0.tolist() == [12.0, 15.0, 18.0, 21.0]
assert argmax_dim1.tolist() == [3, 3, 3]
print("2.2", sum_dim0.tolist(), argmax_dim1.tolist())


# %% [markdown]
# --- 2.3  Autograd by hand ---------------------------------------------------
#  f(x) = x^2 + 3x at x = 2. By hand df/dx = 2x + 3 = 7. Verify torch agrees.
#  This answers the prereq question "when you call .backward(), where are
#  gradients stored?" — they land in `x.grad`.

# %%
x = t.tensor(2.0, requires_grad=True)

# TODO: compute f = x^2 + 3x
f = x**2 + 3*x
# TODO: run backpropagation (one line)
f.backward()

assert x.grad.item() == 7.0
print("2.3", f.item(), x.grad.item())


# %% [markdown]
# --- 2.4  Concept check (write answers in the comments, no code) -------------
#  These are the 0.0 gate. You should be able to answer all of them cold before
#  week 1. (Quiz yourself, then ask me to check them.)
#
#   Q1. At a high level, what is a torch.Tensor?
#       A: array that can run on gpu
#   Q2. What is an nn.Parameter, and what is an nn.Module?
#       A: no clue
#   Q3. When you call .backward(), where do the gradients get stored?
#       A: t.grad
#   Q4. What is a loss function — what does it take in, what does it return?
#       A: idk u didnt explain
#   Q5. What does an optimizer do?
#       A: idk u didnt explain
#   Q6. Parameter vs hyperparameter — what's the difference?
#       A: idk u didnt explain
#   Q7. Give three examples of hyperparameters.
#       A: idk u didnt explain


# %% [markdown]
# ============================================================================
#  SECTION 3 — EINOPS & EINSUM
# ============================================================================
#  Highest leverage for week 1 specifically — 0.1 leans on these immediately.
#  einops names your axes so you never guess at .permute/.reshape/.squeeze
#  again. einsum (the einops version) does linear algebra by naming indices.

# %% [markdown]
# --- 3.1  rearrange ----------------------------------------------------------
#  Pattern is "input_axes -> output_axes". Parentheses (h w) merge axes.

# %%
img = t.arange(2 * 3 * 4).reshape(2, 3, 4)   # (batch=2, h=3, w=4)

# TODO: swap the last two axes -> (2, 4, 3)
swapped = einops.rearrange(img, "a b c -> a c b")                                # "b h w -> b w h"
# TODO: flatten h and w into one axis -> (2, 12)
flat = einops.rearrange(img,"a b c -> a (b c)")                                    # "b h w -> b (h w)"

assert tuple(swapped.shape) == (2, 4, 3)
assert tuple(flat.shape) == (2, 12)
print("3.1", tuple(swapped.shape), tuple(flat.shape))


# %% [markdown]
# --- 3.2  The channels pattern (you'll use this all of ch0.2) ----------------
#  Images often arrive as (batch, height, width, channel) but torch convs want
#  (batch, channel, height, width).

# %%
imgs = t.zeros(8, 32, 32, 3) # b h w c

# TODO: rearrange to (batch, channel, height, width)
imgs_chw = einops.rearrange(imgs, "b h w c -> b c h w")                              # "b h w c -> b c h w"

assert tuple(imgs_chw.shape) == (8, 3, 32, 32)
print("3.2", tuple(imgs_chw.shape))


# %% [markdown]
# --- 3.3  reduce -------------------------------------------------------------
#  reduce = rearrange + an aggregation ("mean", "max", "sum"). Any axis you drop
#  from the output pattern gets reduced over.

# %%
feat = t.arange(2 * 3 * 4).reshape(2, 3, 4).float()

# TODO: mean over the last axis -> (2, 3)
mean_last = einops.reduce(feat, " a b c -> a b", "mean")                             # einops.reduce(feat, "b h w -> b h", "mean")
# TODO: max over the MIDDLE axis -> (2, 4)
max_mid = einops.reduce(feat, " a b c -> a c", "max")                                 # "b h w -> b w", "max"

assert mean_last.tolist() == [[1.5, 5.5, 9.5], [13.5, 17.5, 21.5]]
assert max_mid.tolist() == [[8.0, 9.0, 10.0, 11.0], [20.0, 21.0, 22.0, 23.0]]
print("3.3", mean_last.tolist(), max_mid.tolist())


# %% [markdown]
# --- 3.4  repeat -------------------------------------------------------------
#  repeat introduces NEW axes by tiling. Opposite direction from reduce.

# %%
v = t.tensor([1, 2, 3])

# TODO: tile v into a (4, 3) matrix, each row = [1, 2, 3]
tiled = einops.repeat(v, " b -> a b", a = 4 )                                   # "w -> h w", h=4

assert tiled.tolist() == [[1, 2, 3]] * 4
print("3.4", tiled.tolist())


# %% [markdown]
# --- 3.5  einsum -------------------------------------------------------------
#  Name the indices of each input, then name the output. Repeated index that
#  doesn't appear in the output = summed over. "i j, j k -> i k" is matmul.

# %%
A = t.arange(6).reshape(2, 3).float()
B = t.arange(12).reshape(3, 4).float()

# TODO: matrix multiply A and B via einops.einsum -> (2, 4)
AB = einops.einsum(A, B, " a b, b c -> a c ")                                      # "i j, j k -> i k"
assert t.allclose(AB, A @ B)

u = t.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])   # (2, 3)
w = t.tensor([[1.0, 1.0, 1.0], [3.0, 4.0, 5.0]])   # (2, 3)
# TODO: row-wise dot product -> shape (2,) = [u0·w0, u1·w1]
dots = einops.einsum(u, w, " a b, a b -> a")                                   # "b i, b i -> b"

assert dots.tolist() == [1.0, 8.0]
print("3.5", AB.tolist(), dots.tolist())


# %% [markdown]
# ============================================================================
#  SECTION 4 — RAY TRACING PREVIEW
# ============================================================================
#  A taste of 0.1, minus the intersection math. The thing that trips people up
#  is the broadcast between a (n,) vector of lengths and a (n, 2) array of
#  vectors — you have to reshape lengths to (n, 1) first.

# %%
dirs = t.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0], [3.0, 4.0]])  # (5, 2)

# TODO: length (L2 norm) of each direction -> shape (5,)
lengths = t.linalg.norm(dirs, dim=1)                                # t.linalg.norm(dirs, dim=-1)
# TODO: normalize each direction to unit length -> shape (5, 2)
#       careful: lengths is (5,); you need (5, 1) to divide a (5, 2). Use
#       lengths[:, None]  or  lengths.unsqueeze(-1).
unit_dirs = dirs/lengths[:, None]

assert t.allclose(lengths, t.tensor([1.0, 1.0, 2.0**0.5, 2.0, 5.0]))
assert t.allclose(t.linalg.norm(unit_dirs, dim=-1), t.ones(5))
print("4.1", [round(v, 3) for v in lengths.tolist()])


# %% [markdown]
# ============================================================================
#  SECTION 5 — JAXTYPING (reading, not writing)
# ============================================================================
#  The arena setup code annotates tensors like Float[Tensor, "nrays 2 3"]. It's
#  shape+dtype documentation (and optional runtime checks). You mostly need to
#  READ it, not author it.
#
#   TODO (answer in comments):
#     a) Float[Tensor, "batch channels height width"]
#        -> what does this describe?  A: (4,) tensor image
#     b) write the annotation for "a boolean tensor of shape (n, n)"
#        -> A:  Bool[Tensor, NO CLUE U DIDNT TEACH]   (fill in the ...)
#
#  No asserts here — just get comfortable parsing the strings.


# %% [markdown]
# ============================================================================
#  EXPECTED OUTPUT  (compare your prints; floats may show more decimals —
#  the asserts use allclose so small float noise is fine)
# ============================================================================
#  device available: mps            (or "cpu" — either is fine)
#
#  1.1 [[11, 12, 13], [21, 22, 23], [31, 32, 33]]
#  1.2 [[4.0, 0.0, 6.0], [5.0, 3.0, 3.0]]
#  1.3 [12, 15, 18, 21] [1.5, 5.5, 9.5] [[8, 9, 10, 11]]
#  1.4 3 [4, 1, 0, 3, 0, 2, 0]
#  1.5 [[0, 1, 2], [3, 4, 5]] [[0, 1], [2, 3], [4, 5]] [[0, 3], [1, 4], [2, 5]]
#
#  2.1 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] [0.0, 0.25, 0.5, 0.75, 1.0] [1.0, 2.0, 3.0]
#  2.2 [12.0, 15.0, 18.0, 21.0] [3, 3, 3]
#  2.3 10.0 7.0
#
#  3.1 (2, 4, 3) (2, 12)
#  3.2 (8, 3, 32, 32)
#  3.3 [[1.5, 5.5, 9.5], [13.5, 17.5, 21.5]] [[8.0, 9.0, 10.0, 11.0], [20.0, 21.0, 22.0, 23.0]]
#  3.4 [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]
#  3.5 [[20.0, 23.0, 26.0, 29.0], [56.0, 68.0, 80.0, 92.0]] [1.0, 8.0]
#
#  4.1 [1.0, 1.0, 1.414, 2.0, 5.0]
# ============================================================================
