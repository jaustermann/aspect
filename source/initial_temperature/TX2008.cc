/*
  Copyright (C) 2011 - 2017 by the authors of the ASPECT code.

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
  along with ASPECT; see the file LICENSE.  If not see
  <http://www.gnu.org/licenses/>.
 */


#include <aspect/global.h>
#include <aspect/initial_temperature/TX2008.h>

#include <aspect/simulator_access.h>
#include <aspect/initial_composition/interface.h>
#include <aspect/material_model/interface.h>
#include <aspect/adiabatic_conditions/interface.h>

#include <boost/lexical_cast.hpp>

namespace aspect
{
  namespace InitialTemperature
  {
    template <int dim>
    TX2008<dim>::TX2008 ()
    {}


    template <int dim>
    void
    TX2008<dim>::initialize ()
    {
      Utilities::AsciiDataInitial<dim>::initialize(1);
    }

    template <>
    double
    TX2008<2>::
    initial_temperature (const Point<2> &) const
    {
      // we shouldn't get here but instead should already have been
      // kicked out by the assertion in the parse_parameters()
      // function
      Assert (false, ExcNotImplemented());
      return 0;
    }


    template <int dim>
    double
    TX2008<dim>::
    initial_temperature (const Point<dim> &position) const
    {
      // use either the user-input reference temperature as background temperature
      // (incompressible model) or the adiabatic temperature profile (compressible model)
      const double background_temperature = this->get_material_model().is_compressible() ?
                                            this->get_adiabatic_conditions().temperature(position) :
                                            reference_temperature;

      const double density_perturbation  =  0.01 * Utilities::AsciiDataInitial<dim>::get_data_component(position,0);

      const double depth = this->get_geometry_model().depth(position);
      double temperature_perturbation;
      if (depth > no_perturbation_depth)
        // scale the density perturbation into a temperature perturbation
        {
          // scale the density perturbation into a temperature perturbation
          // see if we need to ask material model for the thermal expansion coefficient
          if (use_material_model_thermal_alpha)
            {
              MaterialModel::MaterialModelInputs<3> in(1, this->n_compositional_fields());
              MaterialModel::MaterialModelOutputs<3> out(1, this->n_compositional_fields());
              in.position[0] = position;
              in.temperature[0] = background_temperature;
              in.pressure[0] = this->get_adiabatic_conditions().pressure(position);
              in.velocity[0] = Tensor<1,3> ();
              for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
                in.composition[0][c] = this->get_initial_composition_manager().initial_composition(position, c);
              in.strain_rate.resize(0);

              this->get_material_model().evaluate(in, out);

              temperature_perturbation = -1./(out.thermal_expansion_coefficients[0]) * density_perturbation;
            }
          else
            temperature_perturbation = -1./thermal_alpha * density_perturbation;
        }
      else
        // set heterogeneity to zero down to a specified depth
        temperature_perturbation = 0.0;

      // add the temperature perturbation to the background temperature
      return background_temperature + temperature_perturbation;

    }


    template <int dim>
    void
    TX2008<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection ("Initial temperature model");
      {
        prm.declare_entry ("Use thermal expansion coefficient from material model", "false",
                           Patterns::Bool (),
                           "Option to take the thermal expansion coefficient from the "
                           "material model instead of from what is specified in this "
                           "section.");
        prm.declare_entry ("Thermal expansion coefficient in initial temperature scaling", "2e-5",
                           Patterns::Double (0),
                           "The value of the thermal expansion coefficient $\\beta$. "
                           "Units: $1/K$.");
        prm.declare_entry ("Reference temperature", "1600.0",
                           Patterns::Double (0),
                           "The reference temperature that is perturbed by the spherical "
                           "harmonic functions. Only used in incompressible models.");
        prm.declare_entry ("Remove temperature heterogeneity down to specified depth", boost::lexical_cast<std::string>(-std::numeric_limits<double>::max()),
                           Patterns::Double (),
                           "This will set the heterogeneity prescribed by S20RTS or S40RTS to zero "
                           "down to the specified depth (in meters). Note that your resolution has "
                           "to be adequate to capture this cutoff. For example if you specify a depth "
                           "of 660km, but your closest spherical depth layers are only at 500km and "
                           "750km (due to a coarse resolution) it will only zero out heterogeneities "
                           "down to 500km. Similar caution has to be taken when using adaptive meshing.");

        Utilities::AsciiDataBase<dim>::declare_parameters(prm,
                                                          "$ASPECT_SOURCE_DIR/data/initial-temperature/TX2008/",
                                                          "TX2008_dens_ascii.txt");
      }
      prm.leave_subsection();
    }


    template <int dim>
    void
    TX2008<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection ("Initial temperature model");
      {
        use_material_model_thermal_alpha = prm.get_bool ("Use thermal expansion coefficient from material model");
        thermal_alpha           = prm.get_double ("Thermal expansion coefficient in initial temperature scaling");
        reference_temperature   = prm.get_double ("Reference temperature");
        no_perturbation_depth   = prm.get_double ("Remove temperature heterogeneity down to specified depth");

        Utilities::AsciiDataBase<dim>::parse_parameters(prm);
      }
      prm.leave_subsection();
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace InitialTemperature
  {
    ASPECT_REGISTER_INITIAL_TEMPERATURE_MODEL(TX2008,
                                              "TX2008 perturbation",
                                              "Implementation of a model in which the initial "
                                              "temperature is derived from files containing data "
                                              "in ascii format. Note the required format of the "
                                              "input data: The first lines may contain any number of comments "
                                              "if they begin with '#', but one of these lines needs to "
                                              "contain the number of grid points in each dimension as "
                                              "for example '# POINTS: 3 3'. "
                                              "The order of the data columns "
                                              "has to be `x', `y', 'Temperature [K]' in a 2d model and "
                                              " `x', `y', `z', 'Temperature [K]' in a 3d model, which means that "
                                              "there has to be a single column "
                                              "containing the temperature. "
                                              "Note that the data in the input "
                                              "files need to be sorted in a specific order: "
                                              "the first coordinate needs to ascend first, "
                                              "followed by the second and the third at last in order to "
                                              "assign the correct data to the prescribed coordinates. "
                                              "If you use a spherical model, "
                                              "then the data will still be handled as Cartesian, "
                                              "however the assumed grid changes. `x' will be replaced by "
                                              "the radial distance of the point to the bottom of the model, "
                                              "`y' by the azimuth angle and `z' by the polar angle measured "
                                              "positive from the north pole. The grid will be assumed to be "
                                              "a latitude-longitude grid. Note that the order "
                                              "of spherical coordinates is `r', `phi', `theta' "
                                              "and not `r', `theta', `phi', since this allows "
                                              "for dimension independent expressions.")
  }
}
