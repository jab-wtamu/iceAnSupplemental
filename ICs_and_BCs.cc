#include <algorithm>
#include <cmath>

// ===========================================================================
// FUNCTION FOR INITIAL CONDITIONS
// ===========================================================================

template <int dim, int degree>
void
customPDE<dim, degree>::setInitialCondition([[maybe_unused]] const Point<dim>  &p,
                                            [[maybe_unused]] const unsigned int index,
                                            [[maybe_unused]] double            &scalar_IC,
                                            [[maybe_unused]] Vector<double>    &vector_IC)
{
  // ---------------------------------------------------------------------
  // Initial condition:
  //   u   = u0 everywhere
  //   phi = one centered seed
  //         - in 3D: thin circular disc / short cylinder
  //         - in 2D: circular seed fallback
  //   xi1 = 0 everywhere initially
  // ---------------------------------------------------------------------

  scalar_IC = 0.0;

  // ---------------------------------------------------------------------
  // Initial condition for the supersaturation field u
  // ---------------------------------------------------------------------
  if (index == 0)
    {
      scalar_IC = u0;
    }

  // ---------------------------------------------------------------------
  // Initial condition for the order parameter field phi
  // ---------------------------------------------------------------------
  else if (index == 1)
    {
      const double cx = 0.5 * userInputs.domain_size[0];
      const double cy = 0.5 * userInputs.domain_size[1];

      // Keep the in-plane seed size close to your previous choice
      const double seed_radius = 5.0;

      if constexpr (dim == 3)
        {
          const double cz = 0.5 * userInputs.domain_size[2];

          // Thin-disc / short-cylinder half thickness
          const double half_thickness = 1.0;

          const double dx = p[0] - cx;
          const double dy = p[1] - cy;
          const double dz = p[2] - cz;

          const double r_xy = std::sqrt(dx * dx + dy * dy);

          // Signed distance to a finite cylinder
          const double dr     = r_xy - seed_radius;
          const double dz_cap = std::abs(dz) - half_thickness;

          const double outside_r = std::max(dr, 0.0);
          const double outside_z = std::max(dz_cap, 0.0);

          const double outside_dist =
            std::sqrt(outside_r * outside_r + outside_z * outside_z);

          const double inside_dist = std::min(std::max(dr, dz_cap), 0.0);

          const double signed_dist = outside_dist + inside_dist;

          scalar_IC = -std::tanh(signed_dist / std::sqrt(2.0));
        }
      else
        {
          // 2D compile-safe fallback: circular seed
          const double dx   = p[0] - cx;
          const double dy   = p[1] - cy;
          const double dist = std::sqrt(dx * dx + dy * dy);

          scalar_IC = -std::tanh((dist - seed_radius) / std::sqrt(2.0));
        }
    }

  // ---------------------------------------------------------------------
  // Initial condition for the auxiliary field xi1
  // ---------------------------------------------------------------------
  else if (index == 2)
    {
      scalar_IC = 0.0;
    }
}

// ===========================================================================
// FUNCTION FOR NON-UNIFORM DIRICHLET BOUNDARY CONDITIONS
// ===========================================================================

template <int dim, int degree>
void
customPDE<dim, degree>::setNonUniformDirichletBCs(
  [[maybe_unused]] const Point<dim>  &p,
  [[maybe_unused]] const unsigned int index,
  [[maybe_unused]] const unsigned int direction,
  [[maybe_unused]] const double       time,
  [[maybe_unused]] double            &scalar_BC,
  [[maybe_unused]] Vector<double>    &vector_BC)
{
  // --------------------------------------------------------------------------
  // This function is intentionally left blank because the parameter file uses
  // NATURAL boundary conditions for u, phi, and xi1.
  // --------------------------------------------------------------------------
}
