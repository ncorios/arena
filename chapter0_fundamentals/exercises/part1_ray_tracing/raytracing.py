#%%
import os
import sys
from functools import partial
from pathlib import Path
from typing import Callable

import einops
import plotly.express as px
import plotly.graph_objects as go
import torch as t
from IPython.display import display
from ipywidgets import interact
from jaxtyping import Bool, Float
from torch import Tensor
from tqdm import tqdm

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part1_ray_tracing"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part1_ray_tracing.tests as tests
from part1_ray_tracing.utils import (
    render_lines_with_plotly,
    setup_widget_fig_ray,
    setup_widget_fig_triangle,
)
from plotly_utils import imshow

MAIN = __name__ == "__main__"
#%%
def make_rays_1d(num_pixels: int, y_limit: float) -> Tensor:
    """
    num_pixels: The number of pixels in the y dimension. Since there is one ray per pixel, this is
        also the number of rays.
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both endpoints.

    Returns: shape (num_pixels, num_points=2, num_dim=3) where the num_points dimension contains
        (origin, direction) and the num_dim dimension contains xyz.

    Example of make_rays_1d(9, 1.0): [
        [[0, 0, 0], [1, -1.0, 0]],
        [[0, 0, 0], [1, -0.75, 0]],
        [[0, 0, 0], [1, -0.5, 0]],
        ...
        [[0, 0, 0], [1, 0.75, 0]],
        [[0, 0, 0], [1, 1, 0]],
    ]
    """
    
    
    rays = t.zeros(num_pixels, 2 ,3) # this creates the size we need
    rays[:,1,1] = t.linspace(-y_limit,y_limit, num_pixels) # index : = every pixel, index 1: the second point (non origin), index 1: y coord of xyz
    rays[:,1,0] = 1 # set every x coord to 1 in the second point ( non origin point)
    return rays



#shape(num_pixes, nump_points = 2, num_dim = 3) means there n # of pixels, with 2 points each, that hold 3 coords each

rays1d = make_rays_1d(9, 10.0)
fig = render_lines_with_plotly(rays1d)

#%%

fig: go.FigureWidget = setup_widget_fig_ray()
display(fig)


@interact(v=(0.0, 6.0, 0.01), seed=(0, 10, 1))
def update(v=0.0, seed=0):
    t.manual_seed(seed)
    L_1, L_2 = t.rand(2, 2)
    P = lambda v: L_1 + v * (L_2 - L_1)
    x, y = zip(P(0), P(6))
    with fig.batch_update():
        fig.update_traces({"x": x, "y": y}, 0)
        fig.update_traces({"x": [L_1[0], L_2[0]], "y": [L_1[1], L_2[1]]}, 1)
        fig.update_traces({"x": [P(v)[0]], "y": [P(v)[1]]}, 2)

# %%
def intersect_ray_1d(ray: Float[Tensor, "points dims"], segment: Float[Tensor, "points dims"]) -> bool:
    """
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    """
    O , D = ray[0,:2], ray[1,:2]
    L_1, L_2 = segment[0,:2], segment[1,:2]
    A = t.stack([D , L_1 - L_2], dim=-1)
    b = L_1 - O
    try: 
        solution = t.linalg.solve(A,b)
        u,v = solution
    except RuntimeError:
        return False
    assert A.shape == (2,2)
    return (u >= 0 and v >= 0 and v <= 1)


tests.test_intersect_ray_1d(intersect_ray_1d)
tests.test_intersect_ray_1d_special_case(intersect_ray_1d)
# %%
#start of batched operations
# bitwise operators for tensors (element by element)
#einops.repeat repeats a tensor a specified amount of times along a new dimension
#torch.any() or .any() accepts dimension to reduce over.
# using ... for indexing basically puts a : in every dimension up to the specified one
# im an actual idiot and forgot that indexing goes inwards
D = t.ones(2)  # Tensor with shape (2,) 2 items
E = t.zeros(3, 2)  # Tensor with shape (3, 2) 3 items, each with 2 elements
E[[True, False, True], :].shape  # Shape: (2, 2) of those 3 elements, keep the outer 2 and keep both elements inside them
E[[True, False, True], :] = D  # Assign values from D to selected rows
# output:
# tensor([[1., 1.],
#         [0., 0.],
#         [1., 1.]])

# F[[[True, True], [False, True]], :].shape  # Shape: (3, 2)
F = t.zeros(2,2,2)
print(F)
F[[[True,True],[False,True]],:].shape
# this collapses bc u have row 0 with 2 items and row 1 with 1 and they need to agree so 3,2
# the trick i guess is that whatever domain youre indexing over collapses to trues.
#mask ndim n will make a tensor have original dims - (n-1)
# end of notes

#general formula to think about when building tensors:
# (address, payload)
# payload = mathematical object

# %%
def intersect_rays_1d(
    rays: Float[Tensor, "nrays 2 3"], segments: Float[Tensor, "nsegments 2 3"]
) -> Bool[Tensor, " nrays"]:
    """
    For each ray, return True if it intersects any segment.
    Rays: (n_rays, points, (x,y,z) )
    Segments: (n_segments, points, (x,y,z))
    """
    # batch all matrices
    nrays = rays.shape[0] #number of rays, size of the first dimension
    nsegs = segments.shape[0]
    mats = t.zeros(nrays, nsegs, 2, 2) #initalize an array, 15 slots of a 2x2 matrix each (our system that we have to solve)
    mats_b = t.zeros(nrays, nsegs, 2) #b array, vector pair
    for i in range(nrays):
        for j in range(nsegs):
            O, D = rays[i,0,:2], rays[i,1,:2] # z coord doesnt matter :2
            L_1, L_2 = segments[j,0,:2], segments[j,1,:2]
            mats[i,j] = t.stack([D, L_1 - L_2], dim = -1) #dim =-1 stacks along columns so u get [[d_x ...], [d_y...]]
            mats_b[i,j] = L_1 - O # no stack this is just vector difference
    dets = t.linalg.det(mats)
    is_singular = dets.abs() < 1e-8 # boolean array of det = 0
    mats[is_singular] = t.eye(2) # boolean indexing, where true replaces 2x2 matrix entry with identity matrix. argument 2 makes a 2x2 square identity matrix
    sol = t.linalg.solve(mats, mats_b) 
    u = sol[..., 0] # the shape of our solution array is (nrays,nsegs,2) where the first 2 index the system and the last dim corresponds to u,v our solution
    v = sol[..., 1]
    return (((u >= 0) & (v >= 0) & (v <= 1) & ~is_singular).any(dim=1)) # conditons: u > 0, 0 < v < 1, and no singular entries alowed, so makes any true entry to is singular false and then &s with our other booleans. .any destroys segments axis and returns 1 bool per ray
    

tests.test_intersect_rays_1d(intersect_rays_1d)
tests.test_intersect_rays_1d_special_case(intersect_rays_1d)

# %%
def make_rays_2d(num_pixels_y: int, num_pixels_z: int, y_limit: float, z_limit: float) -> Float[Tensor, "nrays 2 3"]:
    """
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension

    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.

    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    """
    num_pixels = num_pixels_y * num_pixels_z # total pixels = combinations
    rays = t.zeros(num_pixels, 2 ,3) # this creates the shape we need
    z = t.linspace(-z_limit,z_limit, num_pixels_z) 
    y = t.linspace(-y_limit,y_limit, num_pixels_y) 
    y = einops.repeat(y, "y -> (y z)", z=num_pixels_z) # this creates a new dimension where the y values are repeated for however many slots of npz there are. will make combinations. (y z) flattens to 1d num_pixels size
    z = einops.repeat(z, "z -> (y z)", y=num_pixels_y )
    rays[:,1,2] = z
    rays[:,1,1] = y
    rays[:,1,0] = 1 # set every x coord to 1 in the second point ( non origin point)

    return rays

rays_2d = make_rays_2d(10, 10, 0.3, 0.3)
render_lines_with_plotly(rays_2d)

# %%
