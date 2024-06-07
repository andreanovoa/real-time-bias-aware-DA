import gmsh
import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm.autonotebook
from mpi4py import MPI
from petsc4py import PETSc
import basix

from dolfinx import default_real_type, fem, io, mesh
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block
from ufl import (
    CellDiameter,
    FacetNormal,
    TestFunction,
    TrialFunction,
    avg,
    conditional,
    div,
    dot,
    dS,
    ds,
    dx,
    grad,
    gt,
    inner,
    outer,
)

from dolfinx.cpp.mesh import to_type, cell_entity_type
from dolfinx.fem import (Constant, Function, FunctionSpace,
                         assemble_scalar, dirichletbc, form, locate_dofs_topological, set_bc)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_vector, create_matrix, set_bc)

from dolfinx.graph import adjacencylist
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from dolfinx.io import (VTXWriter, distribute_entity_data, gmshio)
from dolfinx.mesh import create_mesh, meshtags_from_entities

from ufl import (FacetNormal, Identity, Measure, TestFunction, TrialFunction,
                 as_vector, div, dot, ds, dx, inner, lhs, grad, nabla_grad, rhs, sym)

from basix.ufl import element

from ufl.pullback import identity_pullback
from ufl.sobolevspace import H1

import pyvista
from dolfinx import plot

if np.issubdtype(PETSc.ScalarType, np.complexfloating):  # type: ignore
    print("Demo should only be executed with DOLFINx real mode")
    exit(0)
# -

gmsh.initialize()

r = 0.5
D = 2 * r
L = 80 * D
H = 60 * D
c_x = 20 * D
c_y = H / 2
Dist = np.sqrt(3 / 2)
Dist = 1.5

c_x2 = c_x + Dist * np.cos(np.radians(30))
c_y2 = c_y + 0.5 * Dist

c_x3 = c_x + Dist * np.cos(np.radians(30))
c_y3 = c_y - 0.5 * Dist

# lc = 1e-1
gdim = 2
mesh_comm = MPI.COMM_WORLD
model_rank = 0
if mesh_comm.rank == model_rank:
    rectangle = gmsh.model.occ.addRectangle(0, 0, 0, L, H, tag=1)
    obstacle = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)
    obstacle2 = gmsh.model.occ.addDisk(c_x2, c_y2, 0, r, r)
    obstacle3 = gmsh.model.occ.addDisk(c_x3, c_y3, 0, r, r)

if mesh_comm.rank == model_rank:
    fluid = gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, obstacle), (gdim, obstacle2), (gdim, obstacle3)])
    # fluid = gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, obstacle)])
    gmsh.model.occ.synchronize()

fluid_marker = 1
if mesh_comm.rank == model_rank:
    volumes = gmsh.model.getEntities(dim=gdim)
    assert (len(volumes) == 1)
    gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
    gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid")

inlet_marker, outlet_marker, wall_marker, obstacle_marker, obstacle_marker2, obstacle_marker3, = 2, 3, 4, 5, 6, 7
inflow, outflow, walls, obstacle, obstacle2, obstacle3 = [], [], [], [], [], []

if mesh_comm.rank == model_rank:
    boundaries = gmsh.model.getBoundary(volumes, oriented=False)
    for boundary in boundaries:
        center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
        if np.allclose(center_of_mass, [0, H / 2, 0]):
            inflow.append(boundary[1])
        elif np.allclose(center_of_mass, [L, H / 2, 0]):
            outflow.append(boundary[1])
        elif np.allclose(center_of_mass, [L / 2, H, 0]) or np.allclose(center_of_mass, [L / 2, 0, 0]):
            walls.append(boundary[1])
        elif np.allclose(center_of_mass, [c_x2, c_y2, 0]):
            obstacle2.append(boundary[1])
        elif np.allclose(center_of_mass, [c_x3, c_y3, 0]):
            obstacle3.append(boundary[1])
        elif np.allclose(center_of_mass, [c_x, c_y, 0]):
            obstacle.append(boundary[1])

    gmsh.model.addPhysicalGroup(1, walls, wall_marker)
    gmsh.model.setPhysicalName(1, wall_marker, "Walls")
    gmsh.model.addPhysicalGroup(1, inflow, inlet_marker)
    gmsh.model.setPhysicalName(1, inlet_marker, "Inlet")
    gmsh.model.addPhysicalGroup(1, outflow, outlet_marker)
    gmsh.model.setPhysicalName(1, outlet_marker, "Outlet")
    gmsh.model.addPhysicalGroup(1, obstacle, obstacle_marker)
    gmsh.model.setPhysicalName(1, obstacle_marker, "Obstacle")
    gmsh.model.addPhysicalGroup(1, obstacle2, obstacle_marker2)
    gmsh.model.setPhysicalName(1, obstacle_marker2, "Obstacle2")
    gmsh.model.addPhysicalGroup(1, obstacle3, obstacle_marker3)
    gmsh.model.setPhysicalName(1, obstacle_marker3, "Obstacle3")

# Create distance field from obstacle.
# Add threshold of mesh sizes based on the distance field
# LcMax -                  /--------
#                      /
# LcMin -o---------/
#        |         |       |
#       Point    DistMin DistMax
res_min = r / 3
if mesh_comm.rank == model_rank:
    distance_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", [obstacle[0], obstacle2[0], obstacle3[0]])

    boxfield1 = gmsh.model.mesh.field.add("Box")
    gmsh.model.mesh.field.setNumber(boxfield1, "VIn", 1 * res_min)
    gmsh.model.mesh.field.setNumber(boxfield1, "XMin", c_x - 2 * r)
    gmsh.model.mesh.field.setNumber(boxfield1, "XMax", L)
    gmsh.model.mesh.field.setNumber(boxfield1, "YMin", c_y3 - 2 * r)
    gmsh.model.mesh.field.setNumber(boxfield1, "YMax", c_y2 + 2 * r)
    gmsh.model.mesh.field.setNumber(boxfield1, "Thickness", r)

    boxfield2 = gmsh.model.mesh.field.add("Box")
    gmsh.model.mesh.field.setNumber(boxfield2, "VIn", 5 * res_min)
    gmsh.model.mesh.field.setNumber(boxfield2, "XMin", c_x - 5 * r)
    gmsh.model.mesh.field.setNumber(boxfield2, "XMax", L)
    gmsh.model.mesh.field.setNumber(boxfield2, "YMin", c_y3 - 5 * r)
    gmsh.model.mesh.field.setNumber(boxfield2, "YMax", c_y2 + 5 * r)
    gmsh.model.mesh.field.setNumber(boxfield2, "Thickness", r)

    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", res_min)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", 0.25 * H)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", r)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 2 * H)

    min_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field, boxfield1, boxfield2])
    gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

# We also define some helper functions that will be used later


if mesh_comm.rank == model_rank:
    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
    gmsh.model.mesh.generate(gdim)
    gmsh.model.mesh.setOrder(2)
    gmsh.model.mesh.optimize("Netgen")

msh, _, ft = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
ft.name = "Facet markers"

pyvista.start_xvfb()
grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(msh, msh.topology.dim))
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True, show_scalar_bar=True)
plotter.view_xy()
plotter.show()

#########################################################

t = 0
T = 10  # Final time
dt = 0.05
num_steps = int(T / dt)
num_steps = 100

Rey = 60
k = Constant(msh, PETSc.ScalarType(dt))
mu = Constant(msh, PETSc.ScalarType(1 / Rey))  # Dynamic viscosity
rho = Constant(msh, PETSc.ScalarType(1))  # Density

kk = 1  # Polynomial degree

gdim = msh.geometry.dim
v_cg2 = element("Lagrange", msh.topology.cell_name(), kk + 1, shape=(gdim,))
s_cg1 = element("Lagrange", msh.topology.cell_name(), kk)
V = fem.functionspace(msh, v_cg2)
Q = fem.functionspace(msh, s_cg1)

# Funcion space for visualising the velocity field
gdim = msh.geometry.dim
fdim = msh.topology.dim - 1

W = fem.functionspace(msh, ("Discontinuous Lagrange", kk + 1, (gdim,)))
WP = fem.functionspace(msh, ("Discontinuous Lagrange", kk + 1))


# Define boundary conditions


class InletVelocity():
    def __init__(self, t):
        self.t = t

    def __call__(self, x):
        vals = np.zeros(shape=(gdim, x.shape[1]), dtype=PETSc.ScalarType)
        # vals[0] = 4 * 1.5 * np.sin(self.t * np.pi / 8) * x[1] * (0.41 - x[1]) / (0.41**2)

        vals[0] = 1

        return vals


class NoslipVelocity():
    def __init__(self, t):
        self.t = t

    def __call__(self, x):
        vals = np.zeros(shape=(gdim, x.shape[1]), dtype=PETSc.ScalarType)
        return vals


class OutPressure():
    def __init__(self, t):
        self.t = t

    def __call__(self, x):
        values = np.zeros((fdim, x.shape[1]), dtype=PETSc.ScalarType)
        return values


# Inlet
u_inlet = Function(V)
inlet_velocity = InletVelocity(t)

u_inlet.interpolate(inlet_velocity)
bcu_inflow = dirichletbc(u_inlet, locate_dofs_topological(V, fdim, ft.find(inlet_marker)))

# Walls

u_nonslip = Function(V)
noslip_velocity = NoslipVelocity(t)
u_nonslip.interpolate(noslip_velocity)

u_inlet.interpolate(inlet_velocity)
bcu_walls = dirichletbc(u_inlet, locate_dofs_topological(V, fdim, ft.find(wall_marker)))

# Obstacle
bcu_obstacle = dirichletbc(u_nonslip, locate_dofs_topological(V, fdim, ft.find(obstacle_marker)))
bcu_obstacle2 = dirichletbc(u_nonslip, locate_dofs_topological(V, fdim, ft.find(obstacle_marker2)))
bcu_obstacle3 = dirichletbc(u_nonslip, locate_dofs_topological(V, fdim, ft.find(obstacle_marker3)))

bcu = [bcu_inflow, bcu_obstacle, bcu_obstacle2, bcu_obstacle3, bcu_walls]

# Outlet
p_outlet = Function(Q)
press_outlet = OutPressure(t)
p_outlet.interpolate(press_outlet)

bcp_outlet = dirichletbc(p_outlet, locate_dofs_topological(Q, fdim, ft.find(outlet_marker)))
bcp = [bcp_outlet]

##################################################################

u = TrialFunction(V)
v = TestFunction(V)
u_ = Function(V)
u_.name = "u"

u_save = Function(W)
u_save.name = "usav"

p_save = Function(WP)
p_save.name = "psav"

u_s = Function(V)
u_n = Function(V)
u_n1 = Function(V)
p = TrialFunction(Q)
q = TestFunction(Q)

p_ = Function(Q)
p_.name = "p"

phi = Function(Q)

##################################################################


f = Constant(msh, PETSc.ScalarType((0, 0)))
F1 = rho / k * dot(u - u_n, v) * dx
F1 += inner(dot(1.5 * u_n - 0.5 * u_n1, 0.5 * nabla_grad(u + u_n)), v) * dx
F1 += 0.5 * mu * inner(grad(u + u_n), grad(v)) * dx
F1 += - dot(p_, div(v)) * dx
F1 += dot(f, v) * dx
a1 = form(lhs(F1))
L1 = form(rhs(F1))
A1 = create_matrix(a1)
b1 = create_vector(L1)

a2 = form(dot(grad(p), grad(q)) * dx)
L2 = form(-rho / k * dot(div(u_s), q) * dx)
A2 = assemble_matrix(a2, bcs=bcp)
A2.assemble()
b2 = create_vector(L2)

a3 = form(rho * dot(u, v) * dx)
L3 = form(rho * dot(u_s, v) * dx - k * dot(nabla_grad(phi), v) * dx)
A3 = assemble_matrix(a3)
A3.assemble()
b3 = create_vector(L3)

# Solver for step 1
solver1 = PETSc.KSP().create(msh.comm)
solver1.setOperators(A1)
solver1.setType(PETSc.KSP.Type.BCGS)
pc1 = solver1.getPC()
pc1.setType(PETSc.PC.Type.JACOBI)

# Solver for step 2
solver2 = PETSc.KSP().create(msh.comm)
solver2.setOperators(A2)
solver2.setType(PETSc.KSP.Type.MINRES)
pc2 = solver2.getPC()
pc2.setType(PETSc.PC.Type.HYPRE)
pc2.setHYPREType("boomeramg")

# Solver for step 3
solver3 = PETSc.KSP().create(msh.comm)
solver3.setOperators(A3)
solver3.setType(PETSc.KSP.Type.CG)
pc3 = solver3.getPC()
pc3.setType(PETSc.PC.Type.SOR)

##################################################################

##################################################################
from pathlib import Path

folder = Path("results")
folder.mkdir(exist_ok=True, parents=True)
u_save.interpolate(u_)

progress = tqdm.autonotebook.tqdm(desc="Solving PDE", total=num_steps)

import os

snapshot_directory = "snapshots"
if not os.path.exists(snapshot_directory):
    os.makedirs(snapshot_directory)

snapshot_interval = 1  # Define how often to save snapshots
snapshots_u = []  # Lists to store velocity snapshots
snapshots_p = []  # Lists to store pressure snapshots

for i in range(num_steps):
    progress.update(1)
    # Update current time step
    t += dt
    # Update inlet velocity
    inlet_velocity.t = t
    u_inlet.interpolate(inlet_velocity)

    # Step 1: Tentative velocity step
    A1.zeroEntries()
    assemble_matrix(A1, a1, bcs=bcu)
    A1.assemble()
    with b1.localForm() as loc:
        loc.set(0)
    assemble_vector(b1, L1)
    apply_lifting(b1, [a1], [bcu])
    b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b1, bcu)
    solver1.solve(b1, u_s.vector)
    u_s.x.scatter_forward()

    # Step 2: Pressure corrrection step
    with b2.localForm() as loc:
        loc.set(0)
    assemble_vector(b2, L2)
    apply_lifting(b2, [a2], [bcp])
    b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b2, bcp)
    solver2.solve(b2, phi.vector)
    phi.x.scatter_forward()

    p_.vector.axpy(1, phi.vector)
    p_.x.scatter_forward()

    # Step 3: Velocity correction step
    with b3.localForm() as loc:
        loc.set(0)
    assemble_vector(b3, L3)
    b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    solver3.solve(b3, u_.vector)
    u_.x.scatter_forward()

    u_save.interpolate(u_)
    p_save.interpolate(p_)

    # Save snapshots at the specified interval
    if i % snapshot_interval == 0:
        snapshots_u.append(u_save.vector.copy())  # Make sure to copy the data if storing in memory
        snapshots_p.append(p_save.vector.copy())

    # Update variable with solution form this time step
    with u_.vector.localForm() as loc_, u_n.vector.localForm() as loc_n, u_n1.vector.localForm() as loc_n1:
        loc_n.copy(loc_n1)
        loc_.copy(loc_n)

num_velocity_dofs = V.dofmap.index_map_bs * V.dofmap.index_map.size_global
num_pressure_dofs = Q.dofmap.index_map_bs * V.dofmap.index_map.size_global

topology, cell_types, geometry = plot.vtk_mesh(V)

values = np.zeros((geometry.shape[0], 3), dtype=np.float64)
values[:, :len(u_)] = u_.x.array.real.reshape((geometry.shape[0], len(u_)))

pyvista.start_xvfb()

# Create a point cloud of glyphs
function_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

# Create a pyvista-grid for the mesh
grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(msh, msh.topology.dim))
grid.point_data["u"] = values[:, 0]
# Create plotter
plotter = pyvista.Plotter()

plotter.add_mesh(grid, show_edges=False, show_scalar_bar=True)

plotter.view_xy()
plotter.show()
