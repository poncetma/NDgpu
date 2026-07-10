"""First-order perturbation theory for the k-eigenvalue problem.

Given a reference configuration's forward flux phi and adjoint flux phi*, the
reactivity change caused by perturbing the cross sections is estimated *without
re-solving the eigenproblem* by the standard Rayleigh-quotient formula

    1/k' ~= <phi*, A' phi> / <phi*, F' phi>,     d_rho = 1/k - 1/k',

where A is the loss operator (leakage + removal - in-scatter) and F the fission
production operator, primes denoting the perturbed configuration. Because phi*
weights the perturbation by neutron importance, the estimate is first-order
accurate: its error is second order in the flux perturbation.

That last clause is the catch, and it matters for control drums. First-order PT
is excellent for *weak* perturbations -- temperature/density feedback, cross-
section sensitivities, small enrichment changes -- where it matches a direct
re-solve to fractions of a percent. It is *not* reliable for a strong, localized
black absorber such as the HP-MR B4C arc: covering a previously bare cell with
full-strength absorber depresses the local flux by an order-one factor, so the
neglected flux perturbation dominates and the estimate over-predicts the worth
several-fold. That breakdown is the diffusion-homogenization (SPH / rod-cusping)
problem in another guise -- see the tri HP-MR analysis. Use this tool for
sensitivities and weak perturbations; use a direct re-solve (or SPH-corrected
constants) for drum worth.

Works for the diffusion-family solvers (DiffusionEigenSolver and its Hex/Tri
subclasses), whose group state is the scalar flux; SP3's moment-pair state is
not supported here.
"""

from __future__ import annotations


def _loss_apply(solver, flux):
    """(A phi)_g = op_g.apply(phi_g) - sum_{g'!=g} Sigma_s[g'->g] phi_{g'}."""
    G = solver.n_groups
    out = []
    for g in range(G):
        a = solver.ops[g].apply(flux[g])
        for gf in range(G):
            s = solver.sigma_s[gf][g]
            if gf != g and s is not None:
                a = a - s * flux[gf]
        out.append(a)
    return out


def _fission_apply(solver, flux):
    """(F phi)_g = chi_g * sum_{g'} nuSigma_f,g' phi_{g'}."""
    G = solver.n_groups
    production = solver.nu_sigma_f[0] * flux[0]
    for g in range(1, G):
        production = production + solver.nu_sigma_f[g] * flux[g]
    return [solver.chi[g] * production for g in range(G)]


def first_order_reactivity(ref_solver, forward, adjoint, perturbed_solver) -> float:
    """First-order reactivity change d_rho = rho' - rho from a perturbation.

    ref_solver / perturbed_solver : two DiffusionEigenSolver-family solvers on
        the *same* grid and group structure, differing only in their cross
        sections (materials, material_map, or the mix arrays -- e.g. a rotated
        control drum). Only ``perturbed_solver``'s assembled operators and
        fields are used, so no eigen-solve of the perturbed problem is needed.
    forward, adjoint : the reference solver's forward and adjoint Results
        (adjoint from ``ref_solver.solve(adjoint=True)``).

    Returns d_rho in absolute reactivity units (multiply by 1e5 for pcm). The
    estimate captures every cross-section change -- absorption, scattering, and
    the diffusion coefficient's effect on leakage -- because it applies the
    perturbed operators directly to the reference flux.
    """
    if perturbed_solver.n_groups != ref_solver.n_groups:
        raise ValueError("reference and perturbed solvers differ in group count")
    if perturbed_solver.grid.shape != ref_solver.grid.shape:
        raise ValueError("reference and perturbed solvers differ in grid shape")

    xp = ref_solver.xp
    phi = [xp.asarray(forward.flux[g]) for g in range(ref_solver.n_groups)]
    star = [xp.asarray(adjoint.flux[g]) for g in range(ref_solver.n_groups)]

    def dot(a, b):
        return float(sum(xp.sum(a[g] * b[g]) for g in range(len(a))))

    loss = dot(star, _loss_apply(perturbed_solver, phi))
    prod = dot(star, _fission_apply(perturbed_solver, phi))
    inv_k_pert = loss / prod
    return 1.0 / forward.k_eff - inv_k_pert
