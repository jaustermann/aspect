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



#include <aspect/material_model/glisovic_forte.h>
#include <deal.II/base/parameter_handler.h>
#include <aspect/utilities.h>
#include <aspect/lateral_averaging.h>

using namespace dealii;

namespace aspect
{
  namespace MaterialModel
  {  
    namespace internal
    {
      class RadialViscosityLookup
      {
        public:
          RadialViscosityLookup(const std::string &filename)
          {
            // read in depth dependent viscosity
            std::string temp;
            std::ifstream in(filename.c_str(), std::ios::in);
            AssertThrow (in,
                         ExcMessage (std::string("Couldn't open file <") + filename));

            min_depth=1e20;
            max_depth=-1;

            while (!in.eof())
              {
                double visc, depth;
                in >> visc;;
                if (in.eof())
                  break;
                in >> depth;
                depth *=1000.0;
                getline(in, temp);

                min_depth = std::min(depth, min_depth);
                max_depth = std::max(depth, max_depth);

                values.push_back(visc);
                depthvalues.push_back(depth);
              }
          }


       double radial_viscosity(double depth)
          {
            // Do nearest neighbour approach to find which viscosity should be assigned. 
            // Make sure the viscosity file is set up in a way that this makes sense.
            std::vector<double> depth_diff (values.size(), 0);

            Assert(depth>=min_depth, ExcMessage("not in range"));
            Assert(depth<=max_depth, ExcMessage("not in range"));
            
            for (int i = 0; i < values.size(); i++)
               depth_diff[i] = std::abs(depthvalues[i] - depth);
            
            double depth_val = 1e7;
            for (int i = 0; i < values.size(); i++)
               depth_val = std::min(depth_diff[i],depth_val);
	    
	    unsigned int idx = values.size();
            for (int i = 0; i < values.size(); i++)
               if (depth_val == std::abs(depthvalues[i] - depth))
                  idx = i;
	    
            Assert(idx<values.size(), ExcMessage("not in range"));
            return values[idx];
          }

        private:
	  std::vector<double> depthvalues;
          std::vector<double> values;
          double min_depth;
          double max_depth;

      };

    }

    template <int dim>
    void
    GlisovicForte<dim>::initialize()
    {
      radial_viscosity_lookup.reset(new internal::RadialViscosityLookup(datadirectory+radial_viscosity_file_name));
      avg_temp.resize(100);
    }


    template <int dim>
    void
    GlisovicForte<dim>::
    update()
    {
      this->get_lateral_averaging().get_temperature_averages(avg_temp);
    }


    template <int dim>
    double
    GlisovicForte<dim>::
    viscosity (const double temperature,
               const double /*pressure*/,
               const std::vector<double> &, /*compositional_fields, */
               const SymmetricTensor<2,dim> &,
               const Point<dim> &position) const
    {
      const double depth = this->get_geometry_model().depth(position);

      // Get temperature perturbation relative to average temperature
      // unsigned int idx = static_cast<unsigned int>((avg_temp.size()-1) * depth / this->get_geometry_model().maximal_depth());

      // const double delta_temperature = temperature-avg_temp[idx];

      // If you want the adiabatic temperature as your reference temp use the following
      const double background_temperature = this->get_adiabatic_conditions().temperature(position); 

      double delta_temperature = temperature - background_temperature;

      // Scaling from temperature to viscosity
      const double vis_lateral_exp = - temp_to_visc * delta_temperature;
     
      // Limit the lateral viscosity variation to a reasonable interval
      double vis_lateral = std::max(std::min(std::exp(vis_lateral_exp),max_lateral_eta_variation),1/max_lateral_eta_variation);

      // Get radial viscosity
      const double vis_radial = radial_viscosity_lookup->radial_viscosity(depth);

    

      //std_cxx1x::array<double,dim> scoord = aspect::Utilities::spherical_coordinates(position);

      //double theta = scoord[2] * 180/numbers::PI;
      //theta -= 90.;
      //theta *= -1.;

      //if (theta >= vis_lat_cutoff)
      //  vis_lateral = 1;


      // For now just 1D density profile (no lateral variations in viscosity)
      // const double eta = vis_radial;
      // For lateral variations use the following. Also then make sure that the depth average 
      // actually the depth dependent viscosity(?)     
      const double eta = std::max(std::min(vis_lateral * vis_radial,max_eta),min_eta);
      return eta;
    }

    
    // Reference viscosity is needed for some pressure scaling
    template <int dim>
    double
    GlisovicForte<dim>::
    reference_viscosity () const
    {
      return reference_eta;
    }

    template <int dim>
    double
    GlisovicForte<dim>::
    reference_density () const
    {
      return reference_rho;
    }

    template <int dim>
    double
    GlisovicForte<dim>::
    reference_thermal_expansion_coefficient () const
    {
      return thermal_alpha;
    }

    template <int dim>
    double
    GlisovicForte<dim>::
    specific_heat (const double,
                   const double,
                   const std::vector<double> &, /*composition*/
                   const Point<dim> &) const
    {
      return reference_specific_heat;
    }

    template <int dim>
    double
    GlisovicForte<dim>::
    reference_cp () const
    {
      return reference_specific_heat;
    }

    template <int dim>
    double
    GlisovicForte<dim>::
    thermal_conductivity (const double,
                          const double,
                          const std::vector<double> &, /*composition*/
                          const Point<dim> &position) const
    {
      // Set up three values of thermal conducivity for different depths and
      // interpolate linearly inbetween
      const double depth = this->get_geometry_model().depth(position);
      double thermal_cond_val;
      double B_val, A_val;

      std::vector<double>  cond_val (4,3.3);
      std::vector<double>  depth_val (4,0);

      cond_val[1] = 2.5;
      cond_val[2] = 6.25;
      cond_val[3] = 4.8;

      depth_val[1] =   80000;
      depth_val[2] = 2650000;
      depth_val[3] = 2890000;

      if (depth < depth_val[1])
       {B_val = (cond_val[0] - cond_val[1])/(depth_val[0] - depth_val[1]);
        A_val = cond_val[0] - B_val * depth_val[0];
        thermal_cond_val = A_val + B_val * depth;
        }

      if (depth >= depth_val[1])
       if (depth < depth_val[2])
       {B_val = (cond_val[1] - cond_val[2])/(depth_val[1] - depth_val[2]);
        A_val = cond_val[1] - B_val * depth_val[1];
        thermal_cond_val = A_val + B_val * depth;
        }   

      if (depth >= depth_val[2])
       {B_val = (cond_val[2] - cond_val[3])/(depth_val[2] - depth_val[3]);
        A_val = cond_val[2] - B_val * depth_val[2];
        thermal_cond_val = A_val + B_val * depth;
        }

      if(thermal_cond_constant == true)
       thermal_cond_val = k_value;

      return thermal_cond_val;
    }

    template <int dim>
    double
    GlisovicForte<dim>::
    reference_thermal_diffusivity () const
    {
      if(thermal_diff_off == true)
        return k_value/(reference_rho*reference_specific_heat)/1000.;
      else 
        return k_value/(reference_rho*reference_specific_heat);
    }

    template <int dim>
    double
    GlisovicForte<dim>::
    density (const double temperature,
             const double pressure,
             const std::vector<double> &,
             const Point<dim> &position) const
    { // Set up PREM as reference density

      double reference_rho_PREM = reference_rho;
      const double depth = this->get_geometry_model().depth(position);
      const double norm_rad = (6371000 - depth)/6371000;

      // Lower Mantle below 670km
      if(depth > (6371000 - 5701000))
        reference_rho_PREM = 7.9565 - 6.4761 *norm_rad + 5.5283 *norm_rad *norm_rad - 3.0807 *norm_rad *norm_rad *norm_rad;

      // Transition zone 670 - 600km
      if(depth > (6371000 - 5771000))
        if (depth <= (6371000 - 5701000))
        reference_rho_PREM = 5.3197 - 1.4836 *norm_rad;

      // Transition zone 600 - 400km
      if(depth > (6371000 - 5971000))
        if (depth <= (6371000 - 5771000))
        reference_rho_PREM = 11.2494 - 8.0298 *norm_rad;

      // Upper mantle 400 - 220km
      if(depth > (6371000 - 6151000))
        if (depth <= (6371000 - 5971000))
        reference_rho_PREM = 7.1089 - 3.8045 *norm_rad;
 
// model after Tromp
      // 220km to 24.4km
    //  if(depth > (6371000 - 6346600))
    //    if (depth <= (6371000 - 6151000))
    //    reference_rho_PREM = 7.285 - 4.0 *norm_rad;

      // LVZ & LID
//      if(depth > (6371000 - 6346600))
      if(depth > (6371000 - 6271000))
        if (depth <= (6371000 - 6151000))
        reference_rho_PREM = 2.6910 + 0.6924 *norm_rad;

     //this is not PREM since its transversely isotropic in that region ... just interpolated between 3.4 and 2.9
   //   if(depth <= (6371000 - 6151000))
   //     reference_rho_PREM = 17.18 - 14.29 *norm_rad;

      // Crust down to 24.4km
     // if(depth <= (6371000 - 6346600))
      if(depth <= (6371000 - 6271000))
        reference_rho_PREM = 3.2; //2.6;

      reference_rho_PREM *= 1000;

      // Set up three values of thermal expansion for different depths and
      // interpolate linearly inbetween
      double thermal_alpha_val;
      double B_val, A_val;

      std::vector<double>  alpha_val (3,3.5e-5);
      std::vector<double>  depth_val (3,0);

      alpha_val[1] = 2.5e-5;
      alpha_val[2] = 1.0e-5;

      depth_val[1] =  670000;
      depth_val[2] = 2890000;

      if (depth < 670000)
       {B_val = (alpha_val[0] - alpha_val[1])/(depth_val[0] - depth_val[1]);
        A_val = alpha_val[0] - B_val * depth_val[0];
        thermal_alpha_val = A_val + B_val * depth;
        }

      if (depth >= 670000)
       thermal_alpha_val = 3.48e-05 + 2.72e-18 * depth*depth - 1.644e-11*depth;

      if(thermal_alpha_constant == true)
        thermal_alpha_val = thermal_alpha;

      // don't need to account for compressibility, that's already in PREM
      //double rho = reference_rho_PREM * std::exp(reference_compressibility * (pressure - this->get_surface_pressure()));
      double rho = 
      (reference_rho_constant
       ?
       reference_rho * std::exp(reference_compressibility * (pressure - this->get_surface_pressure()))
       :
       reference_rho_PREM); 
 

      // if (adiabat_temp == true)
       rho *= (1 - thermal_alpha_val * (temperature - this->get_adiabatic_conditions().temperature(position)));
      // else
      //   {
      //    unsigned int idx = static_cast<unsigned int>((avg_temp.size()-1) * depth / this->get_geometry_model().maximal_depth());
      //    rho *= (1 - thermal_alpha_val * (temperature - avg_temp[idx]));
      //   }

       return rho;
    }


    template <int dim>
    double
    GlisovicForte<dim>::
    thermal_expansion_coefficient (const double,
                                   const double,
                                   const std::vector<double> &,
                                   const Point<dim> &position) const
    {
      // Set up three values of thermal expansion for different depths and
      // interpolate linearly inbetween

      const double depth = this->get_geometry_model().depth(position);
      double thermal_alpha_val;
      double B_val, A_val;

      std::vector<double>  alpha_val (3,3.5e-5);
      std::vector<double>  depth_val (3,0);

      alpha_val[1] = 2.5e-5;
      alpha_val[2] = 1.0e-5;

      depth_val[1] =  670000;
      depth_val[2] = 2890000;

      if (depth < 670000)
       {B_val = (alpha_val[0] - alpha_val[1])/(depth_val[0] - depth_val[1]);
        A_val = alpha_val[0] - B_val * depth_val[0];
        thermal_alpha_val = A_val + B_val * depth; 
        }

      if (depth >= 670000)
        thermal_alpha_val = 3.48e-05 + 2.72e-18 * depth*depth - 1.644e-11*depth;
     //  {B_val = (alpha_val[1] - alpha_val[2])/(depth_val[1] - depth_val[2]);
     //   A_val = alpha_val[1] - B_val * depth_val[1];
     //   thermal_alpha_val = A_val + B_val * depth;
     //  }        

      if(thermal_alpha_constant == true)
        thermal_alpha_val = thermal_alpha;

      return thermal_alpha_val;
    }


    template <int dim>
    double
    GlisovicForte<dim>::
    compressibility (const double,
                     const double,
                     const std::vector<double> &,
                     const Point<dim> &) const
    {
      // compressibility = 1/rho drho/dp
      return reference_compressibility;
    }

    template <int dim>
    bool
    GlisovicForte<dim>::
    viscosity_depends_on (const NonlinearDependence::Dependence dependence) const
    {
      return true;
    }


    template <int dim>
    bool
    GlisovicForte<dim>::
    density_depends_on (const NonlinearDependence::Dependence dependence) const
    {
      // compare this with the implementation of the density() function
      // to see the dependencies
      if ((dependence & NonlinearDependence::temperature) != NonlinearDependence::none)
        return (thermal_alpha != 0);
      else if ((dependence & NonlinearDependence::pressure) != NonlinearDependence::none)
        return (reference_compressibility != 0);
      else
        return false;
    }

    template <int dim>
    bool
    GlisovicForte<dim>::
    compressibility_depends_on (const NonlinearDependence::Dependence) const
    {
      return false;
    }

    template <int dim>
    bool
    GlisovicForte<dim>::
    specific_heat_depends_on (const NonlinearDependence::Dependence) const
    {
      return false;
    }

    template <int dim>
    bool
    GlisovicForte<dim>::
    thermal_conductivity_depends_on (const NonlinearDependence::Dependence dependence) const
    {
      return false;
    }


    template <int dim>
    bool
    GlisovicForte<dim>::
    is_compressible () const
    {
      return (reference_compressibility != 0);
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
          prm.declare_entry ("Reference viscosity", "1e21",
                             Patterns::Double (0),
                            "The value of the constant viscosity $\\eta_0$. Units: $kg/m/s$.");
          prm.declare_entry ("Thermal conductivity", "4.7",
                             Patterns::Double (0),
                             "The value of the thermal conductivity $k$. "
                             "Units: $W/m/K$.");
          prm.declare_entry ("Reference specific heat", "1250",
                             Patterns::Double (0),
                             "The value of the specific heat $cp$. "
                             "Units: $J/kg/K$.");
          prm.declare_entry ("Thermal expansion coefficient", "2e-5",
                             Patterns::Double (0),
                             "The value of the thermal expansion coefficient $\\beta$. "
                             "Units: $1/K$.");
          prm.declare_entry ("Reference compressibility", "4e-12",
                             Patterns::Double (0),
                             "The value of the reference compressibility. "
                             "Units: $1/Pa$.");
          prm.declare_entry ("Data directory", "$ASPECT_SOURCE_DIR/data/material-model/gfm/",
                             Patterns::DirectoryName (),
                             "The path to the model data. The path may also include the special "
                             "text '$ASPECT_SOURCE_DIR' which will be interpreted as the path "
                             "in which the ASPECT source files were located when ASPECT was "
                             "compiled. This interpretation allows, for example, to reference "
                             "files located in the 'data/' subdirectory of ASPECT. ");
          prm.declare_entry("Radial viscosity file name", "rad_viscosity_MF.txt",
                             Patterns::Anything (),
                             "The file name of the radial viscosity data. ");  
          prm.declare_entry ("Minimum viscosity", "1e19",
                             Patterns::Double(0),
                             "The minimum viscosity that is allowed in the viscosity "
                             "calculation. Smaller values will be cut off.");
          prm.declare_entry ("Maximum viscosity", "5e23",
                             Patterns::Double(0),
                             "The maximum viscosity that is allowed in the viscosity "
                             "calculation. Larger values will be cut off.");
          prm.declare_entry ("Maximum lateral viscosity variation", "1e2",
                             Patterns::Double(0),
                             "The relative cutoff value for lateral viscosity variations "
                             "caused by temperature deviations. The viscosity may vary "
                             "laterally by this factor squared.");
          prm.declare_entry ("Temperature to viscosity scaling", "0",
                             Patterns::Double(0),
                             "Scales the lateral variations in temperature into lateral "
                             "variations in viscosity.");
          prm.declare_entry ("Thermal conductivity constant", "false",
                             Patterns::Bool(),
                             "Switch to leave the thermal conductivity constant.");
          prm.declare_entry ("Reference density constant", "false",
                             Patterns::Bool(),
                             "Switch to leave the reference density constant and not PREM.");
          prm.declare_entry ("Thermal expansion constant", "false",
                             Patterns::Bool(),
                             "Switch to leave the thermal expansion constant.");
          prm.declare_entry ("Thermal diffusivity zero", "false",
                             Patterns::Bool(),
                             "Switch to set the thermal diffusivity to zero for the "
                             "purpose of simulating backward advection.");
          prm.declare_entry ("Latitude cutoff for lateral variations in viscosity","-45.0",
                             Patterns::Double(),
                             "Latitude after which the lateral variations in viscosity are assumerd.");
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
          reference_eta              = prm.get_double ("Reference viscosity");
          k_value                    = prm.get_double ("Thermal conductivity");
          reference_specific_heat    = prm.get_double ("Reference specific heat");
          thermal_alpha              = prm.get_double ("Thermal expansion coefficient");
          reference_compressibility  = prm.get_double ("Reference compressibility");
          datadirectory        = prm.get ("Data directory");
          {
            const std::string      subst_text = "$ASPECT_SOURCE_DIR";
            std::string::size_type position;
            while (position = datadirectory.find (subst_text),  position!=std::string::npos)
              datadirectory.replace (datadirectory.begin()+position,
                                     datadirectory.begin()+position+subst_text.size(),
                                     ASPECT_SOURCE_DIR);
          }
          radial_viscosity_file_name   = prm.get ("Radial viscosity file name");
          min_eta                      = prm.get_double ("Minimum viscosity");
          max_eta                      = prm.get_double ("Maximum viscosity");
          max_lateral_eta_variation    = prm.get_double ("Maximum lateral viscosity variation");
          temp_to_visc                 = prm.get_double ("Temperature to viscosity scaling");
          thermal_cond_constant        = prm.get_bool ("Thermal conductivity constant");
          reference_rho_constant       = prm.get_bool ("Reference density constant");
          thermal_alpha_constant       = prm.get_bool ("Thermal expansion constant");
          thermal_diff_off             = prm.get_bool ("Thermal diffusivity zero");
          vis_lat_cutoff               = prm.get_double ("Latitude cutoff for lateral variations in viscosity");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
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
                                   "A compressible material model that incorporates a depth dependent "
                                   "viscosity, density, thermal expansivity and thermal conductivity. "
                                   "It also includes the option to introduce lateral variations in "
                                   "viscosity. The model parameters follow the paper 'Time-dependent "
                                   "convection models of mantle thermal structure contrained by seismic "
                                   "tomography and geodynamics: implications for mantle plume dynamics "
                                   "and CMB heat flux.' by Glisociv, Forte and Moucha, 2012. ")
  }
}
