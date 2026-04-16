// =================================================================================
// Set the attributes of the primary field variables
// =================================================================================
// u residuals   -> explicitEquationRHS
// phi residual  -> explicitEquationRHS
// xi1 residuals -> nonExplicitEquationRHS

void
customAttributeLoader::loadVariableAttributes()
{
  // ---------------------------------------------------------------------------
  // Variable 0: u
  // ---------------------------------------------------------------------------
  // This is the supersaturation / diffusion field.
  //
  // In the derived weak form:
  //
  //   r_u   = u^n - dt * (Lsat/2) * (B_n / A_n^2) * xi1^n
  //   r_u_x = -dt * F2^n
  //
  // So for the value residual, u depends on:
  //   - u itself
  //   - xi1
  //   - phi
  //   - grad(phi)
  //   - grad(u)
  //
  // And for the gradient residual, u depends on:
  //   - phi
  //   - grad(u)
  //
  // This keeps the field wiring aligned with the intended weak-form structure.
  set_variable_name(0, "u");
  set_variable_type(0, SCALAR);
  set_variable_equation_type(0, EXPLICIT_TIME_DEPENDENT);

  set_dependencies_value_term_RHS(0, "u,xi1,phi,grad(phi),grad(u)");
  set_dependencies_gradient_term_RHS(0, "phi,grad(u)");

  // ---------------------------------------------------------------------------
  // Variable 1: phi
  // ---------------------------------------------------------------------------
  // This is the phase-field / order parameter.
  //
  // The explicit residual used for phi is:
  //
  //   r_phi = phi^n + dt * xi1^n / A_n^2
  //
  // So phi depends on:
  //   - phi
  //   - xi1
  //   - grad(phi)
  //
  // There is no separate gradient residual term for phi in this explicit update.
  set_variable_name(1, "phi");
  set_variable_type(1, SCALAR);
  set_variable_equation_type(1, EXPLICIT_TIME_DEPENDENT);

  set_dependencies_value_term_RHS(1, "phi,xi1,grad(phi)");
  set_dependencies_gradient_term_RHS(1, "");

  // ---------------------------------------------------------------------------
  // Variable 2: xi1
  // ---------------------------------------------------------------------------
  // This is the auxiliary field used to hold the phase-equation RHS structure.
  //
  // The intended residuals are:
  //
  //   r_xi   = f1^n
  //   r_xi_x = -F1^n
  //
  // So xi1 is treated as an AUXILIARY variable rather than an explicitly
  // time-advanced primary evolution field.
  set_variable_name(2, "xi1");
  set_variable_type(2, SCALAR);
  set_variable_equation_type(2, AUXILIARY);

  set_dependencies_value_term_RHS(2, "phi,u,grad(phi)");
  set_dependencies_gradient_term_RHS(2, "grad(phi)");
}


// =================================================================================
// Supplementary helper functions for 3D anisotropy regularization
// =================================================================================

namespace
{
  constexpr double snow_pi = 3.14159265358979323846;

  inline double
  map_theta_to_principal_sector(const double theta_raw)
  {
    double theta_local = std::remainder(theta_raw, snow_pi / 3.0);

    // Keep theta in (-pi/6, pi/6]
    if (theta_local <= -snow_pi / 6.0)
      {
        theta_local += snow_pi / 3.0;
      }
    else if (theta_local > snow_pi / 6.0)
      {
        theta_local -= snow_pi / 3.0;
      }

    return theta_local;
  }

  inline double
  map_psi_to_principal_sector(const double psi_raw)
  {
    // Period for cos(2 psi) is pi
    double psi_wrap = std::fmod(psi_raw, snow_pi);
    if (psi_wrap < 0.0)
      {
        psi_wrap += snow_pi;
      }

    // Fold into [0, pi/2]
    return (psi_wrap <= snow_pi / 2.0) ? psi_wrap : (snow_pi - psi_wrap);
  }

  inline double
  raw_snow_A(const double theta_local,
             const double psi_local,
             const double eps_xy,
             const double eps_z)
  {
    return 1.0 +
           eps_xy * std::cos(6.0 * theta_local) +
           eps_z  * std::cos(2.0 * psi_local);
  }

  inline double
  solve_theta_m(const double psi_local,
                const double eps_xy,
                const double eps_z)
  {
    const double small = 1.0e-12;
    double       lo    = 1.0e-10;
    double       hi    = snow_pi / 6.0 - 1.0e-10;

    auto f_theta = [&](const double th) -> double {
      const double lhs =
        6.0 * eps_xy * std::sin(6.0 * th) * std::cos(th) / (std::sin(th) + small);
      const double rhs =
        1.0 + eps_xy * std::cos(6.0 * th) + eps_z * std::cos(2.0 * psi_local);
      return lhs - rhs;
    };

    double flo = f_theta(lo);
    double fhi = f_theta(hi);

    // If no bracket is found, return zero so caller can safely fall back.
    if (flo * fhi > 0.0)
      {
        return 0.0;
      }

    for (unsigned int iter = 0; iter < 80; ++iter)
      {
        const double mid = 0.5 * (lo + hi);
        const double fmid = f_theta(mid);

        if (flo * fmid <= 0.0)
          {
            hi = mid;
            fhi = fmid;
          }
        else
          {
            lo = mid;
            flo = fmid;
          }
      }

    return 0.5 * (lo + hi);
  }

  inline double
  solve_psi_m(const double theta_local,
              const double eps_xy,
              const double eps_z)
  {
    const double small = 1.0e-12;
    double       lo    = 1.0e-10;
    double       hi    = snow_pi / 2.0 - 1.0e-10;

    auto f_psi = [&](const double ps) -> double {
      const double lhs =
        2.0 * eps_z * std::sin(2.0 * ps) * std::cos(ps) / (std::sin(ps) + small);
      const double rhs =
        1.0 + eps_xy * std::cos(6.0 * theta_local) + eps_z * std::cos(2.0 * ps);
      return lhs - rhs;
    };

    double flo = f_psi(lo);
    double fhi = f_psi(hi);

    if (flo * fhi > 0.0)
      {
        return 0.0;
      }

    for (unsigned int iter = 0; iter < 80; ++iter)
      {
        const double mid = 0.5 * (lo + hi);
        const double fmid = f_psi(mid);

        if (flo * fmid <= 0.0)
          {
            hi = mid;
            fhi = fmid;
          }
        else
          {
            lo = mid;
            flo = fmid;
          }
      }

    return 0.5 * (lo + hi);
  }

  inline double
  eval_regularized_snow_A(const double theta_raw,
                          const double psi_raw,
                          const double eps_xy,
                          const double eps_z)
  {
    const double small = 1.0e-12;

    const double theta_local = map_theta_to_principal_sector(theta_raw);
    const double psi_local   = map_psi_to_principal_sector(psi_raw);
    const double theta_abs   = std::abs(theta_local);

    const double A_raw = raw_snow_A(theta_local, psi_local, eps_xy, eps_z);

    // Missing-orientation thresholds from supplementary material
    const double eps_xy_m = (1.0 + eps_z * std::cos(2.0 * psi_local)) / 35.0;
    const double eps_z_m  = (1.0 + eps_xy * std::cos(6.0 * theta_local)) / 3.0;

    const bool theta_missing = (eps_xy > eps_xy_m);
    const bool psi_missing   = (eps_z  > eps_z_m);

    double theta_m = 0.0;
    double psi_m   = 0.0;

    if (theta_missing)
      {
        theta_m = solve_theta_m(psi_local, eps_xy, eps_z);
      }

    if (psi_missing)
      {
        psi_m = solve_psi_m(theta_local, eps_xy, eps_z);
      }

    // Regularized branch in theta
    double A_theta = A_raw;
    if (theta_missing && theta_m > 0.0)
      {
        const double B1 =
          6.0 * eps_xy * std::sin(6.0 * theta_m) / (std::sin(theta_m) + small);

        const double A1 =
          1.0 +
          eps_xy * std::cos(6.0 * theta_m) +
          eps_z  * std::cos(2.0 * psi_local) -
          B1 * std::cos(theta_m);

        A_theta = A1 + B1 * std::cos(theta_local);
      }

    // Regularized branch in psi
    double A_psi = A_raw;
    if (psi_missing && psi_m > 0.0)
      {
        const double B2 =
          2.0 * eps_z * std::sin(2.0 * psi_m) / (std::sin(psi_m) + small);

        const double A2 =
          1.0 +
          eps_xy * std::cos(6.0 * theta_local) +
          eps_z  * std::cos(2.0 * psi_m) -
          B2 * std::cos(psi_m);

        A_psi = A2 + B2 * std::cos(psi_local);
      }

    // Piecewise supplementary regularization
    if (theta_missing && psi_missing && theta_m > 0.0 && psi_m > 0.0)
      {
        if (theta_abs < theta_m && psi_local >= psi_m)
          {
            return A_theta;
          }
        else if (theta_abs >= theta_m && psi_local < psi_m)
          {
            return A_psi;
          }
        else if (theta_abs < theta_m && psi_local < psi_m)
          {
            const double dtheta = theta_abs - theta_m;
            const double dpsi   = psi_local - psi_m;
            const double denom  = std::sqrt(dtheta * dtheta + dpsi * dpsi) + small;
            const double alpha  = std::abs(dtheta) / denom;

            return alpha * A_theta + (1.0 - alpha) * A_psi;
          }
        else
          {
            return A_raw;
          }
      }
    else if (theta_missing && theta_m > 0.0)
      {
        return (theta_abs < theta_m) ? A_theta : A_raw;
      }
    else if (psi_missing && psi_m > 0.0)
      {
        return (psi_local < psi_m) ? A_psi : A_raw;
      }

    return A_raw;
  }
}


// =============================================================================================
// explicitEquationRHS
// =============================================================================================

template <int dim, int degree>
void
customPDE<dim, degree>::explicitEquationRHS(
  [[maybe_unused]] variableContainer<dim, degree, VectorizedArray<double>> &variable_list,
  [[maybe_unused]] const Point<dim, VectorizedArray<double>>                q_point_loc,
  [[maybe_unused]] const VectorizedArray<double>                            element_volume) const
{
  // ---------------------------------------------------------------------------
  // This implementation is intentionally restricted to 3D.
  //
  // The current snow-crystal model uses the derived 3D formulas for:
  //   - the interface normal n
  //   - theta
  //   - psi
  //   - A(n)
  //   - B(n)
  //
  //
  // IMPORTANT:
  // The main application still instantiates both dim=2 and dim=3 template
  // versions during compilation, so this function must remain compile-safe for
  // dim=2 as well. Therefore, the actual 3D snow physics is placed inside
  // if constexpr (dim == 3), while a compile-only fallback is provided for
  // dim=2.
  // ---------------------------------------------------------------------------

  // ---------------------------------------------------------------------------
  // Get the values and gradients of the model variables at the quadrature point
  // ---------------------------------------------------------------------------

  // u and grad(u)
  scalarvalueType u  = variable_list.get_scalar_value(0);
  scalargradType  ux = variable_list.get_scalar_gradient(0);

  // phi and grad(phi)
  scalarvalueType phi  = variable_list.get_scalar_value(1);
  scalargradType  phix = variable_list.get_scalar_gradient(1);

  // xi1
  scalarvalueType xi1 = variable_list.get_scalar_value(2);

  // ---------------------------------------------------------------------------
  // Interface geometry: build the unit normal vector
  //
  // n = -grad(phi) / |grad(phi)|
  //
  // regval is used to prevent division by zero near vanishing gradients.
  // ---------------------------------------------------------------------------

  scalarvalueType normgradn = std::sqrt(phix.norm_square());
  scalargradType  normal    = (-phix) / (normgradn + constV(regval));

  scalarvalueType A2_n;
  scalarvalueType B_n;
  scalargradType  F2;

  if constexpr (dim == 3)
    {
      scalarvalueType nx = normal[0];
      scalarvalueType ny = normal[1];
      scalarvalueType nz = normal[2];

      // ---------------------------------------------------------------------------
      // Convert the interface normal into the angular representation used in the
      // derived anisotropy model:
      //
      //   theta = atan2(ny, nx)
      //   psi   = atan2(sqrt(nx^2 + ny^2), -nz)
      // ---------------------------------------------------------------------------

      scalarvalueType rho_n = std::sqrt(nx * nx + ny * ny);

      scalarvalueType theta;
      scalarvalueType psi;
      scalarvalueType A_n;

      for (unsigned int i = 0; i < theta.size(); ++i)
        {
          theta[i] = std::atan2(ny[i], nx[i]);
          psi[i]   = std::atan2(rho_n[i], -nz[i]);

          // Supplementary regularized anisotropy A(n)
          A_n[i] = eval_regularized_snow_A(theta[i], psi[i], eps_xy, eps_z);
        }

      // A(n)^2 appears in the explicit time-discretized phase update
      // A small floor is added for numerical stabilization.
      A2_n = A_n * A_n + constV(1.0e-8);

      // B(n) = sqrt(nx^2 + ny^2 + Gamma^2 * nz^2)
      //
      // This is the anisotropy factor used in the kinetic/coupling term.
      B_n =
        std::sqrt(nx * nx + ny * ny + constV(Gamma) * constV(Gamma) * nz * nz);

      // q(phi) used in the supersaturation diffusion flux F2
      scalarvalueType q_phi = constV(1.0) - phi;

      // ---------------------------------------------------------------------------
      // F2 = D_tilde * Gamma^T * q(phi) * Gamma * grad(u)
      //
      // In this 3D implementation, Gamma = diag(1, 1, Gamma).
      // Therefore:
      //   x-component: unchanged
      //   y-component: unchanged
      //   z-component: scaled by Gamma^2
      // ---------------------------------------------------------------------------

      F2[0] = constV(D_tilde) * q_phi * ux[0];
      F2[1] = constV(D_tilde) * q_phi * ux[1];
      F2[2] = constV(D_tilde) * q_phi * constV(Gamma) * constV(Gamma) * ux[2];
    }
  else
    {
      // ---------------------------------------------------------------------------
      // Compile-only fallback for dim = 2
      //
      // This branch is not the intended snow-crystal physics. It exists only so
      // that the dim=2 templates can still compile, because the main application
      // instantiates both 2D and 3D versions.
      // ---------------------------------------------------------------------------

      scalarvalueType q_phi = constV(1.0) - phi;

      A2_n = constV(1.0);
      B_n  = constV(1.0);
      F2   = constV(D_tilde) * q_phi * ux;
    }

  // ---------------------------------------------------------------------------
  // Explicit residual terms
  // ---------------------------------------------------------------------------

  // r_phi = phi^n + dt * xi1^n / A_n^2
  //
  // This advances phi explicitly using xi1 as the already-computed auxiliary RHS.
  scalarvalueType eq_phi =
    phi + constV(userInputs.dtValue) * xi1 / A2_n;

  // r_u = u^n - dt * (Lsat/2) * (B_n / A_n^2) * xi1^n
  //
  // This is the scalar/value residual part of the u equation.
  scalarvalueType eq_u =
    u - constV(userInputs.dtValue) * constV(Lsat / 2.0) * (B_n / A2_n) * xi1;

  // r_u_x = -dt * F2^n
  //
  // This is the gradient/flux residual part of the u equation.
  scalargradType eqx_u =
    constV(-1.0) * constV(userInputs.dtValue) * F2;

  // ---------------------------------------------------------------------------
  // Submit the explicit RHS terms
  // ---------------------------------------------------------------------------

  // Terms for the equation to evolve u
  variable_list.set_scalar_value_term_RHS(0, eq_u);
  variable_list.set_scalar_gradient_term_RHS(0, eqx_u);

  // Terms for the equation to evolve phi
  variable_list.set_scalar_value_term_RHS(1, eq_phi);
}

// =============================================================================================
// nonExplicitEquationRHS
// =============================================================================================

template <int dim, int degree>
void
customPDE<dim, degree>::nonExplicitEquationRHS(
  [[maybe_unused]] variableContainer<dim, degree, VectorizedArray<double>> &variable_list,
  [[maybe_unused]] const Point<dim, VectorizedArray<double>>                q_point_loc,
  [[maybe_unused]] const VectorizedArray<double>                            element_volume) const
{
  // ---------------------------------------------------------------------------
  // This auxiliary-field implementation is also restricted to 3D for the same
  // reason as the explicitEquationRHS: the anisotropy formulas are written in
  // terms of 3D interface angles theta and psi.
  //
  // IMPORTANT:
  // The main application still instantiates both dim=2 and dim=3 template
  // versions during compilation, so this function must remain compile-safe for
  // dim=2 as well. Therefore, the actual 3D snow physics is placed inside
  // if constexpr (dim == 3), while a compile-only fallback is provided for
  // dim=2.
  // ---------------------------------------------------------------------------

  // ---------------------------------------------------------------------------
  // Get the values and derivatives of the model variables
  // ---------------------------------------------------------------------------

  // supersaturation
  scalarvalueType u = variable_list.get_scalar_value(0);

  // phase field and its gradient
  scalarvalueType phi  = variable_list.get_scalar_value(1);
  scalargradType  phix = variable_list.get_scalar_gradient(1);

  // ---------------------------------------------------------------------------
  // Build the interface normal and spherical-angle representation again
  // ---------------------------------------------------------------------------

  scalarvalueType normgradn = std::sqrt(phix.norm_square());
  scalargradType  normal    = (-phix) / (normgradn + constV(regval));

  scalarvalueType A2_n;
  scalarvalueType B_n;
  scalargradType  F1;

  if constexpr (dim == 3)
    {
      scalarvalueType nx = normal[0];
      scalarvalueType ny = normal[1];
      scalarvalueType nz = normal[2];

      scalarvalueType rho_n = std::sqrt(nx * nx + ny * ny);

      scalarvalueType theta;
      scalarvalueType psi;
      scalarvalueType A_n;

      // Numerical derivatives of the supplementary regularized A
      scalarvalueType dA2_dtheta;
      scalarvalueType dA2_dpsi;

      constexpr double fd_step = 1.0e-6;

      for (unsigned int i = 0; i < theta.size(); ++i)
        {
          theta[i] = std::atan2(ny[i], nx[i]);
          psi[i]   = std::atan2(rho_n[i], -nz[i]);

          const double A_here =
            eval_regularized_snow_A(theta[i], psi[i], eps_xy, eps_z);
          const double A_theta_p =
            eval_regularized_snow_A(theta[i] + fd_step, psi[i], eps_xy, eps_z);
          const double A_theta_m =
            eval_regularized_snow_A(theta[i] - fd_step, psi[i], eps_xy, eps_z);
          const double A_psi_p =
            eval_regularized_snow_A(theta[i], psi[i] + fd_step, eps_xy, eps_z);
          const double A_psi_m =
            eval_regularized_snow_A(theta[i], psi[i] - fd_step, eps_xy, eps_z);

          const double dA_dtheta = (A_theta_p - A_theta_m) / (2.0 * fd_step);
          const double dA_dpsi   = (A_psi_p   - A_psi_m)   / (2.0 * fd_step);

          A_n[i]         = A_here;
          dA2_dtheta[i]  = 2.0 * A_here * dA_dtheta;
          dA2_dpsi[i]    = 2.0 * A_here * dA_dpsi;
        }

      // A(n)^2
      A2_n = A_n * A_n + constV(1.0e-8);

      // B(n)
      B_n =
        std::sqrt(nx * nx + ny * ny + constV(Gamma) * constV(Gamma) * nz * nz);

      // -----------------------------------------------------------------------------
      // d(A^2)/d(grad(phi)) via chain rule
      //
      //   d(A^2)/d(grad(phi))
      //     = d(A^2)/d(theta) * d(theta)/d(grad(phi))
      //     + d(A^2)/d(psi)   * d(psi)/d(grad(phi))
      //
      // Here d(A^2)/d(theta) and d(A^2)/d(psi) are taken from the supplementary
      // regularized A(n) using centered finite differences so the derivative stays
      // consistent with the piecewise faceting treatment.
      // -----------------------------------------------------------------------------

      // gradxy = sqrt(phi_x^2 + phi_y^2)
      scalarvalueType gradxy2 = phix[0] * phix[0] + phix[1] * phix[1];
      scalarvalueType gradxy  = std::sqrt(gradxy2);

      // Regularized denominators to avoid division by zero
      scalarvalueType denom_theta = gradxy2 + constV(regval);
      scalarvalueType denom_psi   = gradxy2 + phix[2] * phix[2] + constV(regval);
      scalarvalueType safe_gradxy = gradxy + constV(regval);

      // dtheta / d(grad(phi)) for theta = atan2(phi_y, phi_x)
      scalargradType dtheta_dgradphi;
      dtheta_dgradphi[0] = -phix[1] / denom_theta;
      dtheta_dgradphi[1] =  phix[0] / denom_theta;
      dtheta_dgradphi[2] =  constV(0.0);

      // dpsi / d(grad(phi)) for psi = atan2(sqrt(phi_x^2 + phi_y^2), -phi_z)
      scalargradType dpsi_dgradphi;
      dpsi_dgradphi[0] = (-phix[2] * phix[0]) / (safe_gradxy * denom_psi);
      dpsi_dgradphi[1] = (-phix[2] * phix[1]) / (safe_gradxy * denom_psi);
      dpsi_dgradphi[2] = gradxy / denom_psi;

      // Full chain-rule derivative
      scalargradType dA2_dgradphi =
        dA2_dtheta * dtheta_dgradphi + dA2_dpsi * dpsi_dgradphi;

      // -----------------------------------------------------------------------------
      // F1 = (Gamma^T / 2) * ( |grad(phi)|^2 d(A^2)/d(grad(phi)) + A^2 Gamma grad(phi) )
      //
      // Gamma = diag(1,1,Gamma)
      // -----------------------------------------------------------------------------

      // |grad(phi)|^2
      scalarvalueType gradphi2 = phix.norm_square();

      // Gamma * grad(phi)
      scalargradType Gamma_gradphi;
      Gamma_gradphi[0] = phix[0];
      Gamma_gradphi[1] = phix[1];
      Gamma_gradphi[2] = constV(Gamma) * phix[2];

      // Inner quantity:
      // |grad(phi)|^2 d(A^2)/d(grad(phi)) + A^2 Gamma grad(phi)
      scalargradType inside_F1;
      inside_F1[0] = gradphi2 * dA2_dgradphi[0] + A2_n * Gamma_gradphi[0];
      inside_F1[1] = gradphi2 * dA2_dgradphi[1] + A2_n * Gamma_gradphi[1];
      inside_F1[2] = gradphi2 * dA2_dgradphi[2] + A2_n * Gamma_gradphi[2];

      // Apply Gamma^T / 2
      F1[0] = constV(0.5) * inside_F1[0];
      F1[1] = constV(0.5) * inside_F1[1];
      F1[2] = constV(0.5 * Gamma) * inside_F1[2];
    }
  else
    {
      // ---------------------------------------------------------------------------
      // Compile-only fallback for dim = 2
      //
      // This branch is not the intended snow-crystal physics. It exists only so
      // that the dim=2 templates can still compile, because the main application
      // instantiates both 2D and 3D versions.
      // ---------------------------------------------------------------------------

      A2_n = constV(1.0);
      B_n  = constV(1.0);
      F1   = constV(0.0) * phix;
    }

  // ---------------------------------------------------------------------------
  // Local scalar term f1(phi, grad(phi), u)
  //
  //
  //   r_xi = f1^n = -f'(phi) + lambda * B(n) * g'(phi) * u
  //
  // Here:
  //   -f'(phi) = phi - phi^3
  //   g'(phi)  = (1 - phi^2)^2
  // ---------------------------------------------------------------------------

  scalarvalueType minus_fprime = phi - phi * phi * phi;

  scalarvalueType gprime =
    (constV(1.0) - phi * phi) * (constV(1.0) - phi * phi);

  scalarvalueType eq_xi1 =
    minus_fprime + constV(lambda) * B_n * gprime * u;

  // r_xi_x = -F1
  scalargradType eqx_xi1 = -F1;

  // ---------------------------------------------------------------------------
  // Submit the auxiliary residual terms
  // ---------------------------------------------------------------------------

  variable_list.set_scalar_value_term_RHS(2, eq_xi1);
  variable_list.set_scalar_gradient_term_RHS(2, eqx_xi1);
}

// =============================================================================================
// equationLHS
// =============================================================================================

template <int dim, int degree>
void
customPDE<dim, degree>::equationLHS(
  [[maybe_unused]] variableContainer<dim, degree, VectorizedArray<double>> &variable_list,
  [[maybe_unused]] const Point<dim, VectorizedArray<double>>                q_point_loc,
  [[maybe_unused]] const VectorizedArray<double>                            element_volume) const
{
  // No separate time-independent left-hand-side terms are added here.
  // This function remains empty for the current explicit/auxiliary formulation.
}
