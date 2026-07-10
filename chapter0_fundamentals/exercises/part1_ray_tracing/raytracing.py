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
# beginning of triangles
Point = Float[Tensor, "points=3"]


def triangle_ray_intersects(A: Point, B: Point, C: Point, O: Point, D: Point) -> bool:
    """
    A: shape (3,), one vertex of the triangle
    B: shape (3,), second vertex of the triangle
    C: shape (3,), third vertex of the triangle
    O: shape (3,), origin point
    D: shape (3,), direction point

    Return True if the ray and the triangle intersect.
    """
    # System
    # [-D * (B-A) * (C - A)][vector] = [O - A]
    mat_a = t.zeros(3,3)
    mat_a[:,0] = -D
    mat_a[:,1] = (B-A)
    mat_a[:,2] = (C-A)
    mat_b = (O - A)
    sol = t.linalg.solve(mat_a, mat_b)
    s,u,v = sol
    return ( ((s>= 0) & (u >= 0) & (v >= 0) & ((u + v) <= 1)).item())
    # could have used stack with dim = 1. dim = 1 puts them in a column


tests.test_triangle_ray_intersects(triangle_ray_intersects)
# %%
#notes
#storage() returns an object wrapping the undelrying c++ array. useful to see if 2 tensors share
# view shares memory with original tensor, copy doesnt. 
# you can access the original tensor with x._base
# %%
def raytrace_triangle(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"],
) -> Bool[Tensor, " nrays"]:
    """
    For each ray, return True if the triangle intersects that ray.
    """
    nrays = rays.shape[0]
    mat_a = t.zeros(nrays,3,3)
    mat_b = t.zeros(nrays,3)
    for i in range(nrays):
        ray = rays[i,:,:]
        O, D = ray
        A, B, C = triangle
        mat_a[i] = t.stack([(-D),(B-A),(C-A)], dim = 1)
        mat_b[i] = (O - A)
        
    sol = t.linalg.solve(mat_a,mat_b)
    s = sol[...,0]
    u = sol[...,1]
    v = sol[...,2]
    return ( ((s>= 0) & (u >= 0) & (v >= 0) & ((u + v) <= 1)))


A = t.tensor([1, 0.0, -0.5])
B = t.tensor([1, -0.5, 0.0])
C = t.tensor([1, 0.5, 0.5])
num_pixels_y = num_pixels_z = 15
y_limit = z_limit = 0.5

# Plot triangle & rays
test_triangle = t.stack([A, B, C], dim=0)
rays2d = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
triangle_lines = t.stack([A, B, C, A, B, C], dim=0).reshape(-1, 2, 3)
render_lines_with_plotly(rays2d, triangle_lines)

# Calculate and display intersections
intersects = raytrace_triangle(rays2d, test_triangle)
img = intersects.reshape(num_pixels_y, num_pixels_z).int()
imshow(img, origin="lower", width=600, title="Triangle (as intersected by rays)")


# %%
triangles = t.load(section_dir / "pikachu.pt", weights_only=True)

# %%
def raytrace_mesh(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"],
) -> Float[Tensor, " nrays"]:
    """
    For each ray, return the distance to the closest intersecting triangle, or infinity.

    """
    NR = rays.shape[0]
    NT = triangles.shape[0]

    mat_a = t.zeros(NR,NT,3,3) # solving batches of 3x3 systems, with indexing by ray and triamgle
    mat_b = t.zeros(NR,NT,3)
    for i in range(NR):
        for j in range(NT):
            ray = rays[i,...]
            triangle = triangles[j,...]
            O, D = ray
            A, B, C = triangle
            mat_a[i,j] = t.stack([(-D),(B-A),(C-A)], dim = 1) # dim = 1 stacking by columns 
            mat_b[i,j] = (O - A)
    sol = t.linalg.solve(mat_a,mat_b)
    s = sol[...,0] 
    u = sol[...,1]
    v = sol[...,2]
    mask = (((s>= 0) & (u >= 0) & (v >= 0) & ((u + v) <= 1))) # conditions for parameters
    s[~mask] = float('inf') # mask returns which parameters pass, flips boolean mask and replaces distances that dont have parameters that fit bounds with infinity
    return t.min(s,dim=1)[0] #finds minimum distance for each ray searching along triangle dimension. this returns a tuple of [min,index] so index 0 for just mins.
    # logic being that if theres no intersection, distance is inf so another minimum distance gets chosen. and if theres no intersections than inf is returned


num_pixels_y = 120
num_pixels_z = 120
y_limit = z_limit = 1

rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
rays[:, 0] = t.tensor([-2, 0.0, 0.0])
dists = raytrace_mesh(rays, triangles)
intersects = t.isfinite(dists).view(num_pixels_y, num_pixels_z)
dists_square = dists.view(num_pixels_y, num_pixels_z)
img = t.stack([intersects, dists_square], dim=0)

fig = px.imshow(img, facet_col=0, origin="lower", color_continuous_scale="magma", width=1000)
fig.update_layout(coloraxis_showscale=False)
for i, text in enumerate(["Intersects", "Distance"]):
    fig.layout.annotations[i]["text"] = text
fig.show()

# %%
def rotation_matrix(theta: Float[Tensor, ""]) -> Float[Tensor, "rows cols"]:
    """
    Creates a rotation matrix representing a counterclockwise rotation of `theta` around the y-axis.
    """
    return t.tensor(
        [
            [t.cos(theta), 0.0, t.sin(theta)],
            [0.0, 1.0, 0.0],
            [-t.sin(theta), 0.0, t.cos(theta)],
        ]
    )



tests.test_rotation_matrix(rotation_matrix)

#%% 
def raytrace_mesh_video(
    rays: Float[Tensor, "nrays points dim"],
    triangles: Float[Tensor, "ntriangles points dims"],
    rotation_matrix: Callable[[float], Float[Tensor, "rows cols"]],
    raytrace_function: Callable,
    num_frames: int,
) -> Bool[Tensor, "nframes nrays"]:
    """
    Creates a stack of raytracing results, rotating the triangles by `rotation_matrix` each frame.
    """
    result = []
    theta = t.tensor(2 * t.pi) / num_frames
    R = rotation_matrix(theta)
    for theta in tqdm(range(num_frames)):
        triangles = triangles @ R
        result.append(raytrace_function(rays, triangles))
        t.cuda.empty_cache()  # clears GPU memory (this line will be more important later on!)
    return t.stack(result, dim=0)


def display_video(distances: Float[Tensor, "frames y z"]):
    """
    Displays video of raytracing results, using Plotly. `distances` is a tensor where the [i, y, z]
    element is distance to the closest triangle for the i-th frame & the [y, z]-th ray in our 2D
    grid of rays.
    """
    px.imshow(
        distances,
        animation_frame=0,
        origin="lower",
        zmin=0.0,
        zmax=distances[distances.isfinite()].quantile(0.99).item(),
        color_continuous_scale="viridis_r",  # "Brwnyl"
    ).update_layout(coloraxis_showscale=False, width=550, height=600, title="Raytrace mesh video").show()


num_pixels_y = 250
num_pixels_z = 250
y_limit = z_limit = 0.8
num_frames = 50

rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
rays[:, 0] = t.tensor([-3.0, 0.0, 0.0])
dists = raytrace_mesh_video(rays, triangles, rotation_matrix, raytrace_mesh, num_frames)
dists = einops.rearrange(dists, "frames (y z) -> frames y z", y=num_pixels_y)

display_video(dists)

# %%
def raytrace_mesh_gpu(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"],
) -> Float[Tensor, " nrays"]:
    """
    For each ray, return the distance to the closest intersecting triangle, or infinity.

    All computations should be performed on the GPU.
    """
    device = t.device('mps')
    rays = rays.to(device)
    triangles = triangles.to(device)
    NR = rays.shape[0]
    NT = triangles.shape[0]

    mat_a = t.zeros(NR,NT,3,3, device=device) # solving batches of 3x3 systems, with indexing by ray and triamgle
    mat_b = t.zeros(NR,NT,3, device=device)
    for i in range(NR):
        for j in range(NT):
            ray = rays[i,...]
            triangle = triangles[j,...]
            O, D = ray
            A, B, C = triangle
            mat_a[i,j] = t.stack([(-D),(B-A),(C-A)], dim = 1) # dim = 1 stacking by columns 
            mat_b[i,j] = (O - A)
    sol = t.linalg.solve(mat_a,mat_b)
    s = sol[...,0] 
    u = sol[...,1]
    v = sol[...,2]
    mask = (((s>= 0) & (u >= 0) & (v >= 0) & ((u + v) <= 1))) # conditions for parameters
    s[~mask] = float('inf') # mask returns which parameters pass, flips boolean mask and replaces distances that dont have parameters that fit bounds with infinity
    return t.min(s,dim=1)[0].cpu() #finds minimum distance for each ray searching along triangle dimension. this returns a tuple of [min,index] so index 0 for just mins.
    # logic being that if theres no intersection, distance is inf so another minimum distance gets chosen. and if theres no intersections than inf is returned
    # moves back to cpu


dists = raytrace_mesh_video(rays, triangles, rotation_matrix, raytrace_mesh_gpu, num_frames)
dists = einops.rearrange(dists, "frames (y z) -> frames y z", y=num_pixels_y)
display_video(dists)

# %%
