from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from cbf_control import quadrotorModel
from casadi import MX
import numpy as np

def create_ocp_solver(x0, N_horizon, t_horizon, L, ts)->AcadosOcp:
    # Creation of the optimal control problem
    # Optimal control problem class
    ocp = AcadosOcp()

    # Model of the system
    model, get_trans, get_quat, constraint = quadrotorModel(L)

    # Constructing the optimal control problem
    ocp.model = model

    # Dimension of the problem
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu

    # Set the dimension of the problem
    ocp.p = model.p
    ocp.dims.N = N_horizon



    # Auxiliary variable initialization
    ocp.parameter_values = np.zeros(nx + nu)

    # Constraints
    ocp.constraints.constr_type = 'BGH'

    # Set constraints
    ocp.constraints.x0 = x0

    ## Nonlinear constraints
    ocp.model.con_h_expr = constraint.expr
    nsbx = 0
    nh = constraint.expr.shape[0]
    nsh = nh
    ns = nsh + nsbx

    ##### Gains over the Horizon for the nonlinear constraint
    ocp.cost.zl = 100*np.ones((ns, ))
    ocp.cost.Zl = 100*np.ones((ns, ))
    ocp.cost.Zu = 100*np.ones((ns, ))
    ocp.cost.zu = 100*np.ones((ns, ))
#
    ##### Norm of a quaternion should be one
    ocp.constraints.lh = np.array([constraint.min])
    ocp.constraints.uh = np.array([constraint.max])

    ocp.constraints.lsh = np.zeros(nsh)
    ocp.constraints.ush = np.zeros(nsh)
    ocp.constraints.idxsh = np.array(range(nsh))
#
#
    # Set options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM" 
    ocp.solver_options.qp_solver_cond_N = N_horizon // 4
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"  
    ocp.solver_options.regularize_method = "CONVEXIFY"  
    ocp.solver_options.integrator_type = "IRK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.Tsim = ts
    ocp.solver_options.tf = t_horizon
    return ocp