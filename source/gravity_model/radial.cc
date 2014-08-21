/*
  Copyright (C) 2011 - 2014 by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.

  ASPECT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ASPECT; see the file doc/COPYING.  If not see
  <http://www.gnu.org/licenses/>.
*/


#include <aspect/gravity_model/radial.h>

#include <deal.II/base/tensor.h>

namespace aspect
{
  namespace GravityModel
  {
// ------------------------------ RadialConstant -------------------
    template <int dim>
    Tensor<1,dim>
    RadialConstant<dim>::gravity_vector (const Point<dim> &p) const
    {
      const double r = p.norm();
      return -magnitude * p/r;
    }



    template <int dim>
    void
    RadialConstant<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Gravity model");
      {
        prm.enter_subsection("Radial constant");
        {
          prm.declare_entry ("Magnitude", "9.81",
                             Patterns::Double (0),
                             "Magnitude of the gravity vector in $m/s^2$. The direction is "
                             "always radially inward towards the center of the earth.");
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();
    }


    template <int dim>
    void
    RadialConstant<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Gravity model");
      {
        prm.enter_subsection("Radial constant");
        {
          magnitude = prm.get_double ("Magnitude");
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();
    }

// ------------------------------ RadialEarthLike -------------------

    template <int dim>
    Tensor<1,dim>
    RadialEarthLike<dim>::gravity_vector (const Point<dim> &p) const
    {
      const double r = p.norm();
      return -(1.245e-6 * r + 7.714e13/r/r) * p / r;
    }


// ----------------------------- Glisovic Earth Profile ------------

    template <int dim>
    Tensor<1,dim>
    RadialGlisovic<dim>::gravity_vector (const Point<dim> &p) const
    {
      const double r = p.norm();
      const double d_km = 6371 - r/1000;
      Tensor<1,dim> grav;
      if (r > 5951000) //420km depth
         grav = -(0.00038 * d_km + 9.82) * p / r;

      if (r <= 5951000)
       if (r > 5701000) // between 420km and 670km
         grav = -(8e-5 * d_km + 9.9464) * p / r;

      if (r <= 5701000) // down to CMB
         grav = -(1.3317e-10 * d_km*d_km*d_km - 3.9768e-7 * d_km*d_km + 2.9088e-4 * d_km + 9.9436) * p / r;

      return grav;
    }

// ----------------------------- RadialLinear ----------------------


    template <int dim>
    Tensor<1,dim>
    RadialLinear<dim>::gravity_vector (const Point<dim> &p) const
    {
      if (p.norm() == 0.0)
        return Tensor<1,dim>();

      const double depth = this->get_geometry_model().depth(p);
      return  (-magnitude_at_surface * p/p.norm() *
               (1.0 - depth/this->get_geometry_model().maximal_depth()));
    }


    template <int dim>
    void
    RadialLinear<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Gravity model");
      {
        prm.enter_subsection("Radial linear");
        {
          prm.declare_entry ("Magnitude at surface", "9.8",
                             Patterns::Double (0),
                             "Magnitude of the radial gravity vector "
                             "at the surface of the domain. Units: $m/s^2$");
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();
    }


    template <int dim>
    void
    RadialLinear<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Gravity model");
      {
        prm.enter_subsection("Radial linear");
        {
          magnitude_at_surface = prm.get_double ("Magnitude at surface");
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();
    }

  }
}

// explicit instantiations
namespace aspect
{
  namespace GravityModel
  {
    ASPECT_REGISTER_GRAVITY_MODEL(RadialConstant,
                                  "radial constant",
                                  "A gravity model in which the gravity direction is radially inward "
                                  "and at constant magnitude. The magnitude is read from the parameter "
                                  "file in subsection 'Radial constant'.")

    ASPECT_REGISTER_GRAVITY_MODEL(RadialEarthLike,
                                  "radial earth-like",
                                  "A gravity model in which the gravity direction is radially inward "
                                  "and with a magnitude that matches that of the earth at "
                                  "the core-mantle boundary as well as at the surface and "
                                  "in between is physically correct under the assumption "
                                  "of a constant density.")

    ASPECT_REGISTER_GRAVITY_MODEL(RadialLinear,
                                  "radial linear",
                                  "A gravity model which is radially inward, where the magnitude"
                                  "decreases linearly with depth down to zero at the maximal depth "
                                  "the geometry returns, as you would get with a constant"
                                  "density spherical domain. (Note that this would be for a full "
                                  "sphere, not a spherical shell.) The magnitude of gravity at the "
                                  "surface is read from the input file in a section "
                                  "``Gravity model/Radial linear''.")
    ASPECT_REGISTER_GRAVITY_MODEL(RadialGlisovic,
                                  "radial glisovic",
                                  "A gravity model which follows the model shown in figure 1 of "
                                  "Glisovic and Forte, Geoscience Frontiers, 2014.")
  }
}
