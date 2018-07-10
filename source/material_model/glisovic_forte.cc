/*
  Copyright (C) 2011 - 2018 by the authors of the ASPECT code.

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


#include <aspect/material_model/glisovic_forte.h>
#include <aspect/utilities.h>
#include <aspect/geometry_model/interface.h>

namespace aspect
{
  namespace MaterialModel
  {
    template <int dim>
    void
    GlisovicForte<dim>::
    evaluate(const MaterialModel::MaterialModelInputs<dim> &in,
             MaterialModel::MaterialModelOutputs<dim> &out) const
    {
      for (unsigned int i=0; i < in.position.size(); ++i)
        {
          const double delta_temp = in.temperature[i]-reference_T;
          const double temperature_dependence
            = (reference_T > 0
               ?
               //    std::max(std::min(std::exp(-thermal_viscosity_exponent *
               //                               delta_temp/reference_T),
               std::max(std::min(std::exp(- thermal_viscosity_exponent * delta_temp),
                                 maximum_thermal_prefactor),
                        minimum_thermal_prefactor)
               :
               1.0);

          out.viscosities[i] = ((composition_viscosity_prefactor != 1.0) && (in.composition[i].size()>0))
                               ?
                               // Geometric interpolation
                               std::pow(10.0, ((1-in.composition[i][0]) * std::log10(eta *
                                                                                     temperature_dependence)
                                               + in.composition[i][0] * std::log10(eta *
                                                                                   composition_viscosity_prefactor *
                                                                                   temperature_dependence)))
                               :
                               temperature_dependence * eta;

          const double c = (in.composition[i].size()>0)
                           ?
                           std::max(0.0, in.composition[i][0])
                           :
                           0.0;

          out.specific_heat[i] = reference_specific_heat;
          //  out.thermal_conductivities[i] = k_value;
          out.compressibilities[i] = 0.0;
          // Pressure derivative of entropy at the given positions.
          out.entropy_derivative_pressure[i] = 0.0;
          // Temperature derivative of entropy at the given positions.
          out.entropy_derivative_temperature[i] = 0.0;
          // Change in composition due to chemical reactions at the
          // given positions. The term reaction_terms[i][c] is the
          // change in compositional field c at point i.
          for (unsigned int c=0; c<in.composition[i].size(); ++c)
            out.reaction_terms[i][c] = 0.0;

          // --------------- Thermal expansivisty
          // Set up three values of thermal expansion for different depths and
          // interpolate linearly inbetween

          const double depth = this->get_geometry_model().depth(in.position[i]);
          double thermal_alpha_val = 0.;
          double B_val, A_val;

          std::vector<double>  alpha_val (3,3.5e-5);
          std::vector<double>  depth_val_a (3,0);

          alpha_val[1] = 2.5e-5;
          alpha_val[2] = 1.0e-5;

          depth_val_a[1] =  670000;
          depth_val_a[2] = 2890000;

          if (depth < 670000)
            {
              B_val = (alpha_val[0] - alpha_val[1])/(depth_val_a[0] - depth_val_a[1]);
              A_val = alpha_val[0] - B_val * depth_val_a[0];
              thermal_alpha_val = A_val + B_val * depth;
            }

          if (depth >= 670000)
            {
              B_val = (alpha_val[1] - alpha_val[2])/(depth_val_a[1] - depth_val_a[2]);
              A_val = alpha_val[1] - B_val * depth_val_a[1];
              thermal_alpha_val = A_val + B_val * depth;
            }

          if (thermal_alpha_constant == true)
            thermal_alpha_val = thermal_alpha;

          out.thermal_expansion_coefficients[i] = thermal_alpha_val;


          // --------------- Thermal conductivity
          // Set up three values of thermal conducivity for different depths and
          // interpolate linearly inbetween
          double thermal_cond_val = 0.;

          std::vector<double>  cond_val (4,3.3);
          std::vector<double>  depth_val (4,0);

          cond_val[1] = 2.5;
          cond_val[2] = 6.25;
          cond_val[3] = 4.8;

          depth_val[1] =   80000;
          depth_val[2] = 2650000;
          depth_val[3] = 2890000;

          if (depth < depth_val[1])
            {
              B_val = (cond_val[0] - cond_val[1])/(depth_val[0] - depth_val[1]);
              A_val = cond_val[0] - B_val * depth_val[0];
              thermal_cond_val = A_val + B_val * depth;
            }

          if (depth >= depth_val[1])
            if (depth < depth_val[2])
              {
                B_val = (cond_val[1] - cond_val[2])/(depth_val[1] - depth_val[2]);
                A_val = cond_val[1] - B_val * depth_val[1];
                thermal_cond_val = A_val + B_val * depth;
              }

          if (depth >= depth_val[2])
            {
              B_val = (cond_val[2] - cond_val[3])/(depth_val[2] - depth_val[3]);
              A_val = cond_val[2] - B_val * depth_val[2];
              thermal_cond_val = A_val + B_val * depth;
            }

          if (thermal_cond_constant == true)
            thermal_cond_val = k_value;

          out.thermal_conductivities[i] = thermal_cond_val;

          // ------------ Density
          out.densities[i] = reference_rho * (1 - thermal_alpha_val * (in.temperature[i] - reference_T))
                             + compositional_delta_rho * c;

        }
    }


    template <int dim>
    double
    GlisovicForte<dim>::
    reference_viscosity () const
    {
      return eta;
    }



    template <int dim>
    bool
    GlisovicForte<dim>::
    is_compressible () const
    {
      return false;
    }



    template <int dim>
    void
    GlisovicForte<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Glisovic Forte model");
        {
          prm.declare_entry ("Reference density", "3300",
                             Patterns::Double (0),
                             "Reference density $\\rho_0$. Units: $kg/m^3$.");
          prm.declare_entry ("Reference temperature", "293",
                             Patterns::Double (0),
                             "The reference temperature $T_0$. The reference temperature is used "
                             "in both the density and viscosity formulas. Units: $K$.");
          prm.declare_entry ("Viscosity", "5e24",
                             Patterns::Double (0),
                             "The value of the constant viscosity $\\eta_0$. This viscosity may be "
                             "modified by both temperature and compositional dependencies. Units: $kg/m/s$.");
          prm.declare_entry ("Composition viscosity prefactor", "1.0",
                             Patterns::Double (0),
                             "A linear dependency of viscosity on the first compositional field. "
                             "Dimensionless prefactor. With a value of 1.0 (the default) the "
                             "viscosity does not depend on the composition. See the general documentation "
                             "of this model for a formula that states the dependence of the "
                             "viscosity on this factor, which is called $\\xi$ there.");
          prm.declare_entry ("Thermal viscosity exponent", "0.0",
                             Patterns::Double (0),
                             "The temperature dependence of viscosity. Dimensionless exponent. "
                             "See the general documentation "
                             "of this model for a formula that states the dependence of the "
                             "viscosity on this factor, which is called $\\beta$ there.");
          prm.declare_entry("Maximum thermal prefactor","1.0e2",
                            Patterns::Double (0),
                            "The maximum value of the viscosity prefactor associated with temperature "
                            "dependence.");
          prm.declare_entry("Minimum thermal prefactor","1.0e-2",
                            Patterns::Double (0),
                            "The minimum value of the viscosity prefactor associated with temperature "
                            "dependence.");
          prm.declare_entry ("Thermal conductivity", "4.7",
                             Patterns::Double (0),
                             "The value of the thermal conductivity $k$. "
                             "Units: $W/m/K$.");
          prm.declare_entry ("Reference specific heat", "1250",
                             Patterns::Double (0),
                             "The value of the specific heat $C_p$. "
                             "Units: $J/kg/K$.");
          prm.declare_entry ("Thermal expansion coefficient", "2e-5",
                             Patterns::Double (0),
                             "The value of the thermal expansion coefficient $\\alpha$. "
                             "Units: $1/K$.");
          prm.declare_entry ("Density differential for compositional field 1", "0",
                             Patterns::Double(),
                             "If compositional fields are used, then one would frequently want "
                             "to make the density depend on these fields. In this simple material "
                             "model, we make the following assumptions: if no compositional fields "
                             "are used in the current simulation, then the density is simply the usual "
                             "one with its linear dependence on the temperature. If there are compositional "
                             "fields, then the density only depends on the first one in such a way that "
                             "the density has an additional term of the kind $+\\Delta \\rho \\; c_1(\\mathbf x)$. "
                             "This parameter describes the value of $\\Delta \\rho$. Units: $kg/m^3/\\textrm{unit "
                             "change in composition}$.");
          prm.declare_entry ("Thermal conductivity constant", "false",
                             Patterns::Bool(),
                             "Switch to leave the thermal conductivity constant.");
          prm.declare_entry ("Thermal expansion constant", "false",
                             Patterns::Bool(),
                             "Switch to leave the thermal expansion constant.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }



    template <int dim>
    void
    GlisovicForte<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Glisovic Forte model");
        {
          reference_rho              = prm.get_double ("Reference density");
          reference_T                = prm.get_double ("Reference temperature");
          eta                        = prm.get_double ("Viscosity");
          composition_viscosity_prefactor = prm.get_double ("Composition viscosity prefactor");
          thermal_viscosity_exponent = prm.get_double ("Thermal viscosity exponent");
          maximum_thermal_prefactor       = prm.get_double ("Maximum thermal prefactor");
          minimum_thermal_prefactor       = prm.get_double ("Minimum thermal prefactor");
          if ( maximum_thermal_prefactor == 0.0 ) maximum_thermal_prefactor = std::numeric_limits<double>::max();
          if ( minimum_thermal_prefactor == 0.0 ) minimum_thermal_prefactor = std::numeric_limits<double>::min();

          k_value                    = prm.get_double ("Thermal conductivity");
          reference_specific_heat    = prm.get_double ("Reference specific heat");
          thermal_alpha              = prm.get_double ("Thermal expansion coefficient");
          compositional_delta_rho    = prm.get_double ("Density differential for compositional field 1");
          thermal_cond_constant        = prm.get_bool ("Thermal conductivity constant");
          thermal_alpha_constant       = prm.get_bool ("Thermal expansion constant");

          if (thermal_viscosity_exponent!=0.0 && reference_T == 0.0)
            AssertThrow(false, ExcMessage("Error: Material model simple with Thermal viscosity exponent can not have reference_T=0."));
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      // Declare dependencies on solution variables
      this->model_dependence.compressibility = NonlinearDependence::none;
      this->model_dependence.specific_heat = NonlinearDependence::none;
      this->model_dependence.thermal_conductivity = NonlinearDependence::none;
      this->model_dependence.viscosity = NonlinearDependence::none;
      this->model_dependence.density = NonlinearDependence::none;

      if (thermal_viscosity_exponent != 0)
        this->model_dependence.viscosity |= NonlinearDependence::temperature;
      if (composition_viscosity_prefactor != 1.0)
        this->model_dependence.viscosity |= NonlinearDependence::compositional_fields;

      if (thermal_alpha != 0)
        this->model_dependence.density |=NonlinearDependence::temperature;
      if (compositional_delta_rho != 0)
        this->model_dependence.density |=NonlinearDependence::compositional_fields;
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace MaterialModel
  {
    ASPECT_REGISTER_MATERIAL_MODEL(GlisovicForte,
                                   "Glisovic Forte",
                                   "")
  }
}
