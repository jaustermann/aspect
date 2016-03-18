/*
  Copyright (C) 2011 - 2015 by the authors of the ASPECT code.

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


#include <aspect/initial_conditions/S40RTS_perturbation_TXlith.h>
#include <aspect/geometry_model/spherical_shell.h>
#include <aspect/utilities.h>
#include <fstream>
#include <iostream>
#include <deal.II/base/std_cxx11/array.h>

#include <boost/math/special_functions/spherical_harmonic.hpp>

namespace aspect
{
  namespace InitialConditions
  {
    namespace internal
    {

      namespace S40RTS
      {

       class TX2008Lookup
       {
         public:
         TX2008Lookup(const std::string &filename)
         {
           std::string temp;
           std::ifstream in(filename.c_str(), std::ios::in);
           AssertThrow (in,
                        ExcMessage (std::string("Couldn't open file <") + filename));

          // in >> order;
          // getline(in,temp);  // throw away the rest of the line    

          // const int maxnumber = num_splines * (order+1)*(order+1);
           const int maxnumber = 65341 * 22;
           // read in all coefficients as a single data vector
           for (int i=0; i<maxnumber; i++)
           {
              double new_val;
              in >> new_val;
              coeffs.push_back(new_val);
           }

         }

         // Declare a function that returns the cosine coefficients
         const std::vector<double> & grid_perturbations() const
         {
           return coeffs;
         }

         private:
           std::vector<double> coeffs;
       };

        // Read in the spherical harmonics that are located in data/initial-conditions/S40RTS
        // and were downloaded from http://www.earth.lsa.umich.edu/~jritsema/research.html
        // Ritsema et al. choose real sine and cosine coefficients that follow the normalization
        // by Dahlen & Tromp, Theoretical Global Seismology (equations B.58 and B.99).

        class SphericalHarmonicsLookup
        {
          public:
            SphericalHarmonicsLookup(const std::string &filename)
            {
              std::string temp;
              std::ifstream in(filename.c_str(), std::ios::in);
              AssertThrow (in,
                           ExcMessage (std::string("Could not open file <") + filename + ">."));

              in >> order;
              getline(in,temp);  // throw away the rest of the line

              const int num_splines = 21;
              const int maxnumber = num_splines * (order+1)*(order+1);

              // read in all coefficients as a single data vector
              for (int i=0; i<maxnumber; i++)
                {
                  double new_val;
                  in >> new_val;
                  coeffs.push_back(new_val);
                }

              // reorder the coefficients into sin and cos coefficients. a_lm will be the cos coefficients
              // and b_lm the sin coefficients.
              int ind = 0;
              int ind_degree;

              for (int j=0; j<num_splines; j++)

                for (int i=0; i<order+1; i++)
                  {
                    a_lm.push_back(coeffs[ind]);
                    b_lm.push_back(0.0);
                    ind += 1;

                    ind_degree = 0;
                    while (ind_degree < i)
                      {
                        a_lm.push_back(coeffs[ind]);
                        ind += 1;
                        b_lm.push_back(coeffs[ind]);
                        ind += 1;
                        ind_degree +=1;
                      }
                  }
            }

            // Declare a function that returns the cosine coefficients
            const std::vector<double> &cos_coeffs() const
            {
              return a_lm;
            }

            // Declare a function that returns the sine coefficients
            const std::vector<double> &sin_coeffs() const
            {
              return b_lm;
            }

            int maxdegree()
            {
              return order;
            }

          private:
            int order;
            std::vector<double> coeffs;
            std::vector<double> a_lm;
            std::vector<double> b_lm;

        };

        // Read in the knot points for the spline interpolation. They are located in data/
        // initial-conditions/S40RTS and were taken from the plotting script
        // lib/libS20/splhsetup.f which is part of the plotting package downloadable at
        // http://www.earth.lsa.umich.edu/~jritsema/research.html
        class SplineDepthsLookup
        {
          public:
            SplineDepthsLookup(const std::string &filename)
            {
              std::string temp;
              std::ifstream in(filename.c_str(), std::ios::in);
              AssertThrow (in,
                           ExcMessage (std::string("Could not open file <") + filename + ">."));

              getline(in,temp);  // throw away the rest of the line
              getline(in,temp);  // throw away the rest of the line

              int num_splines = 21;

              for (int i=0; i<num_splines; i++)
                {
                  double new_val;
                  in >> new_val;

                  depths.push_back(new_val);
                }
            }

            const std::vector<double> &spline_depths() const
            {
              return depths;
            }

         private:
         std::vector<double> depths;
       };
    
      class VsToDensityLookup
      {
        public:
          VsToDensityLookup(const std::string &filename)
          {
            std::string temp;
            std::ifstream in(filename.c_str(), std::ios::in);
            AssertThrow (in,
                         ExcMessage (std::string("Couldn't open file <") + filename));

            min_depth=1e20;
            max_depth=-1;

            getline(in,temp);  //eat first line

            while (!in.eof())
              {
                double scaling, depth;
                in >> scaling;
                if (in.eof())
                  break;
                in >> depth;
                depth *=1000.0;
                getline(in, temp);

                min_depth = std::min(depth, min_depth);
                max_depth = std::max(depth, max_depth);

                values.push_back(scaling);
                depthvalues.push_back(depth);
              }
          }

       double vstodensity_scaling(double depth)
          {

            std::vector<double> depth_diff (values.size(), 0);

            Assert(depth>=min_depth, ExcMessage("not in range"));
            Assert(depth<=max_depth, ExcMessage("not in range"));

            for (int i = 0; i < values.size(); i++)
               depth_diff[i] = std::abs(depthvalues[i] - depth);

            double depth_val = 1e6;
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


   class GeothermLookup
      {
        public:
          GeothermLookup(const std::string &filename)
          {
            std::string temp;
            std::ifstream in(filename.c_str(), std::ios::in);
            AssertThrow (in,
                         ExcMessage (std::string("Couldn't open file <") + filename));

            min_depth=1e20;
            max_depth=-1;

            getline(in,temp);  //eat first line

            while (!in.eof())
              {
                double val, depth;
                in >> depth;
                if (in.eof())
                  break;
                in >> val;
                depth *=1000.0;
                getline(in, temp);

                min_depth = std::min(depth, min_depth);
                max_depth = std::max(depth, max_depth);

                values.push_back(val);
                depthvalues.push_back(depth);
              }
          }

         double geotherm(double depth)
          {

            std::vector<double> depth_diff (values.size(), 0);

            Assert(depth>=min_depth, ExcMessage("not in range"));
            Assert(depth<=max_depth, ExcMessage("not in range"));

            for (int i = 0; i < values.size(); i++)
               depth_diff[i] = std::abs(depthvalues[i] - depth);

            double depth_val = 1e6;
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
    }


    template <int dim>
    void
    S40RTSPerturbation_TXlith<dim>::initialize()
    {

      TX2008_lookup.reset(new internal::S40RTS::TX2008Lookup(datadirectory+ "TX2008_dens.txt"));

      spherical_harmonics_lookup.reset(new internal::S40RTS::SphericalHarmonicsLookup(datadirectory+harmonics_coeffs_file_name));
      spline_depths_lookup.reset(new internal::S40RTS::SplineDepthsLookup(datadirectory+spline_depth_file_name));

      if (vs_to_depth_constant == false)
        vs_to_density_lookup.reset(new internal::S40RTS::VsToDensityLookup(datadirectory+vs_to_density_file_name));
      if (read_geotherm_in == true)
        geotherm_lookup.reset(new internal::S40RTS::GeothermLookup(datadirectory+geotherm_file_name));
     }

    // NOTE: this module uses the Boost spherical harmonics package which is not designed
    // for very high order (> 100) spherical harmonics computation. If you use harmonic
    // perturbations of a high order be sure to confirm the accuracy first.
    // For more information, see:
    // http://www.boost.org/doc/libs/1_49_0/libs/math/doc/sf_and_dist/html/math_toolkit/special/sf_poly/sph_harm.html

    template <>
    double
    S40RTSPerturbation_TXlith<2>::
    initial_temperature (const Point<2> &) const
    {
      // we shouldn't get here but instead should already have been
      // kicked out by the assertion in the parse_parameters()
      // function
      Assert (false, ExcNotImplemented());
      return 0;
    }


    template <>
    double
    S40RTSPerturbation_TXlith<3>::
    initial_temperature (const Point<3> &position) const
    {
     const unsigned int dim = 3;

     // use either the user-input reference temperature as background temperature
     // (incompressible model) or the adiabatic temperature profile (compressible model)
      const double background_temperature = this->get_material_model().is_compressible() ?
                                            this->get_adiabatic_conditions().temperature(position) :
                                            reference_temperature;        

     // convert coordinates from [x,y,z] to [r, phi, theta] 
     std_cxx1x::array<double,dim> scoord = aspect::Utilities::spherical_coordinates(position);;

     const double depth = this->get_geometry_model().depth(position);
     
     double density_perturbation;



     if (depth < 250000) {   // If in the upper 250km use the TX model

       // depth values over which gypsum is defined
       double depth_values[] = {0, 100, 175, 250, 325, 400, 525, 650, 750, 850, 1000, 1150, 1300, 1450, 1600, 1750, 1900, 2050, 2200, 2350, 2500, 2650, 2891};

       for (int i=0; i<23; i++)
         depth_values[i] *= 1000.;

       const double depth = this->get_geometry_model().depth(position);
       int depth_index;
         for (int i=0; i<22; i++)
           if ( depth > depth_values[i] && depth < depth_values[i+1])
              depth_index = i;

       double phi = scoord[1] * 180/numbers::PI;
       if (phi > 180)
          phi -= 360;

       double theta = scoord[2] * 180/numbers::PI;
       theta -= 90.;
       theta *= -1.;

       // Make sure floor and ceil produces two different coordinates
       phi += 0.000001;
       theta += 0.000001;

       int x1 = floor(phi);
       int x2 = ceil(phi);
       int y1 = floor(theta);
       int y2 = ceil(theta);

       std::vector<int> index_lonlat (4,0);
       index_lonlat[0] = (x1+180) + 361*(y1 + 90);
       index_lonlat[1] = (x2+180) + 361*(y1 + 90);
       index_lonlat[2] = (x1+180) + 361*(y2 + 90);
       index_lonlat[3] = (x2+180) + 361*(y2 + 90);

       const std::vector<double>  coeffs = TX2008_lookup->grid_perturbations();

       double density_pert[22][65341];

       int next_ind = 0;
       for (int i=0; i<65341; i++)
         for (int j=0; j<22; j++)
           {
           density_pert[j][i] = coeffs[next_ind];
           next_ind += 1;
           }


       // bilinear interpolation from http://en.wikipedia.org/wiki/Bilinear_interpolation
       const double Q11 = density_pert[depth_index][index_lonlat[0]];
       const double Q21 = density_pert[depth_index][index_lonlat[1]];
       const double Q12 = density_pert[depth_index][index_lonlat[2]];
       const double Q22 = density_pert[depth_index][index_lonlat[3]];

       // The last term (0.01) is necessary to convert percent perturbations to absolute values
       const double perturbation = 1/((x2-x1) * (y2-y1)) *
                                 (Q11 *(x2 - phi)*(y2 - theta) +
                                  Q21 *(phi - x1)*(y2 - theta) +
                                  Q12 *(x2 - phi)*(theta - y1) +
                                  Q22 *(phi - x1)*(theta - y1)) * 0.01;



        density_perturbation = perturbation;
     }
  
     else {

       // getthe degree from the input file (20 or 40)
       const int maxdegree = spherical_harmonics_lookup->maxdegree();

       const int num_spline_knots = 21; // The tomography models are parameterized by 21 layers

       const int num_coeffs = (maxdegree+1) * (maxdegree+2) / 2 * num_spline_knots;

       // get the spherical harmonics coefficients
       const std::vector<double> a_lm = spherical_harmonics_lookup->cos_coeffs();
       const std::vector<double> b_lm = spherical_harmonics_lookup->sin_coeffs();

       // get spline knots and rescale them from [-1 1] to [CMB moho]
       const std::vector<double> r = spline_depths_lookup->spline_depths();
       const double rmoho = 6346e3;
       const double rcmb = 3480e3;
       std::vector<double> depth_values(num_spline_knots,0);

       for (int i = 0; i<num_spline_knots; i++)
         depth_values[i] = rcmb+(rmoho-rcmb)*0.5*(r[i]+1);

       // iterate over all degrees and orders at each depth and sum them all up.
       std::vector<double> spline_values(num_spline_knots,0);
       double prefact;
       int ind = 0;

       for (int depth_interp = 0; depth_interp < num_spline_knots; depth_interp++)
       {
       for (int degree_l = 0; degree_l < maxdegree+1; degree_l++)
         {
         for (int order_m = 0; order_m < degree_l+1; order_m++)
           {
           const double cos_component = boost::math::spherical_harmonic_r(degree_l,order_m,scoord[2],scoord[1]); //real / cos part
           const double sin_component = boost::math::spherical_harmonic_i(degree_l,order_m,scoord[2],scoord[1]); //imaginary / sine part
           if (degree_l == 0) 
             {
             // option to zero out degree 0, i.e. make sure that the average of the perturbation
             // is 0 and the average of the temperature is the background temperature 
                  if (degree_l == 0)
                    prefact = (zero_out_degree_0
                               ?
                               0.
                               :
                               1.);
                  else if (order_m == 0)
                    prefact = 1.;
                  else
                    prefact = sqrt(2.);

                  spline_values[depth_interp] += prefact * (a_lm[ind]*cos_component + b_lm[ind]*sin_component);

                  ind += 1;
              }
           }
         }
       }
       

       // We need to reorder the spline_values because the coefficients are given from 
       // the surface down to the CMB and the interpolation knots range from the CMB up to
       // the surface.
       std::vector<double> spline_values_inv(num_spline_knots,0);
       for (int i=0; i<num_spline_knots; i++)
         spline_values_inv[i] = spline_values[num_spline_knots-1 - i];

       // The boundary condition for the cubic spline interpolation is that the function is linear 
       // at the boundary (i.e. moho and CMB). Values outside the range are linearly 
       // extrapolated.
       aspect::Utilities::tk::spline s;     
       s.set_points(depth_values,spline_values_inv);

       // Get value at specific depth
       const double perturbation = s(scoord[0]);

       // scale the perturbation in seismic velocity into a density perturbation
       // vs_to_density is an input parameter
       const double depth = this->get_geometry_model().depth(position);

       double dens_scaling;
       if (vs_to_depth_constant == true)
         dens_scaling = vs_to_density;
       else
         dens_scaling = vs_to_density_lookup -> vstodensity_scaling(depth);

       density_perturbation = dens_scaling * perturbation;

     }


     //get thermal alpha
     double thermal_alpha_val;
     double B_val, A_val;

     std::vector<double>  alpha_val (3,3.5e-5);
     std::vector<double>  depth_val (3,0);

     alpha_val[1] = 2.5e-5;
     alpha_val[2] = 1.5e-5;

     depth_val[1] =  670000;
     depth_val[2] = 2890000;

     if (depth < 670000)
       {B_val = (alpha_val[0] - alpha_val[1])/(depth_val[0] - depth_val[1]);
        A_val = alpha_val[0] - B_val * depth_val[0];
        thermal_alpha_val = A_val + B_val * depth;
        }

     if (depth >= 670000)
       {B_val = (alpha_val[1] - alpha_val[2])/(depth_val[1] - depth_val[2]);
        A_val = alpha_val[1] - B_val * depth_val[1];
        thermal_alpha_val = A_val + B_val * depth;
        }

     if (thermal_alpha_constant == true)
        thermal_alpha_val = thermal_alpha;

     // scale the density perturbation into a temperature perturbation
     // THIS ISNT COMPRESSIBLE - GLISOVIC ET AL 2012 THAT ITS THIRD ORDER EFFECT
      double temperature_perturbation;
      if (depth > no_perturbation_depth)
        // scale the density perturbation into a temperature perturbation
        temperature_perturbation =  -1./thermal_alpha_val * density_perturbation;
      else
        // set heterogeneity to zero down to a specified depth
        temperature_perturbation = 0.0;


     // set up background temperature as a geotherm
     /*       Note that the values we read in here have reasonable default values equation to
       the following:*/
 
     double temperature;

     // option to either take simplified geotherm or read one in from file
     if(read_geotherm_in == true)
        temperature = geotherm_lookup->geotherm(depth) + temperature_perturbation;
     else
        temperature = background_temperature + temperature_perturbation;

     return temperature;
     }


    template <int dim>
    void
    S40RTSPerturbation_TXlith<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Initial conditions");
      {
        prm.enter_subsection("S40RTS perturbation TX lithosphere");
        {
          prm.declare_entry("Data directory", "$ASPECT_SOURCE_DIR/data/initial-conditions/S40RTS/",
                            Patterns::DirectoryName (),
                            "The path to the model data. ");
          prm.declare_entry ("Initial condition file name", "S40RTS.sph",
                             Patterns::Anything(),
                             "The file name of the spherical harmonics coefficients "
                             "from Ritsema et al.");
          prm.declare_entry ("Spline knots depth file name", "Spline_knots.txt",
                             Patterns::Anything(),
                             "The file name of the spline knot locations from "
                             "Ritsema et al.");
          prm.declare_entry ("Vs to density scaling file", "R_scaling.txt",
                            Patterns::Anything(),
                            "The file name of the scaling between vs and density. "
                            "Default values are from Simmons et al., 2009."); 
          prm.declare_entry("Geotherm file name","Geotherm-red.txt",
                             Patterns::Anything (),
                             "The file name for the geotherm / background temp from Glisovic "
                             "et al., 2014.");
          prm.declare_entry ("vs to density scaling", "0.25",
                             Patterns::Double (0),
                             "This parameter specifies how the perturbation in shear wave velocity "
                             "as prescribed by S20RTS or S40RTS is scaled into a density perturbation. "
                             "See the general description of this model for more detailed information.");
          prm.declare_entry ("Thermal expansion coefficient in initial temperature scaling", "2e-5",
                             Patterns::Double (0),
                             "The value of the thermal expansion coefficient $\\beta$. "
                             "Units: $1/K$.");
          prm.declare_entry ("Remove degree 0 from perturbation","true",
                             Patterns::Bool (),
                             "Option to remove the degree zero component from the perturbation, "
                             "which will ensure that the laterally averaged temperature for a fixed "
                             "depth is equal to the background temperature.");
          prm.declare_entry ("Read geotherm in from file","false",
                             Patterns::Bool (),
                             "Option to read in a geotherm / background temperature for the "
                             "initial temperature field from a file.");
          prm.declare_entry ("Reference temperature", "1600.0",
                             Patterns::Double (0),
                             "The reference temperature that is perturbed by the spherical "
                             "harmonic functions. Only used in incompressible models.");
          prm.declare_entry ("Thermal expansion constant","false",
                             Patterns::Bool(),
                             "Switch to set the thermal expansion to a constant value.");
          prm.declare_entry ("Vs to density scaling constant","false",
                             Patterns::Bool(),
                             "Switch to set the vs to density scalind to a constant value.");
          prm.declare_entry ("Remove temperature heterogeneity down to specified depth", boost::lexical_cast<std::string>(-std::numeric_limits<double>::max()),
                             Patterns::Double (),
                             "This will set the heterogeneity prescribed by S20RTS or S40RTS to zero "
                             "down to the specified depth (in meters).Note that your resolution has "
                             "to be adquate to capture this cutoff. For example if you specify a depth "
                             "of 660km, but your closest spherical depth layers are only at 500km and "
                             "750km (due to a coarse resolution) it will only zero out heterogeneities "
                             "down to 500km. Similar caution has to be taken when using adaptive meshing.");
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();
    }


    template <int dim>
    void
    S40RTSPerturbation_TXlith<dim>::parse_parameters (ParameterHandler &prm)
    {
      AssertThrow (dim == 3,
                   ExcMessage ("The 'S40RTS perturbation' model for the initial "
                               "temperature is only available for 3d computations."));

      prm.enter_subsection("Initial conditions");
      {
        prm.enter_subsection("S40RTS perturbation TX lithosphere");
        {
          datadirectory           = prm.get ("Data directory");
          {
            const std::string      subst_text = "$ASPECT_SOURCE_DIR";
            std::string::size_type position;
            while (position = datadirectory.find (subst_text),  position!=std::string::npos)
              datadirectory.replace (datadirectory.begin()+position,
                                     datadirectory.begin()+position+subst_text.size(),
                                     ASPECT_SOURCE_DIR);
          }
          harmonics_coeffs_file_name = prm.get ("Initial condition file name");
          spline_depth_file_name  = prm.get ("Spline knots depth file name");
          vs_to_density_file_name = prm.get("Vs to density scaling file");
          geotherm_file_name      = prm.get ("Geotherm file name");
          vs_to_density           = prm.get_double ("vs to density scaling");
          thermal_alpha           = prm.get_double ("Thermal expansion coefficient in initial temperature scaling");
          zero_out_degree_0       = prm.get_bool ("Remove degree 0 from perturbation");
          read_geotherm_in        = prm.get_bool ("Read geotherm in from file");
          reference_temperature   = prm.get_double ("Reference temperature");
          thermal_alpha_constant  = prm.get_bool ("Thermal expansion constant");
          vs_to_depth_constant    = prm.get_bool ("Vs to density scaling constant");
          constant_temp           = prm.get_bool ("Constant background temperature");
          no_perturbation_depth   = prm.get_double ("Remove temperature heterogeneity down to specified depth");
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();

      initialize ();
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace InitialConditions
  {
    ASPECT_REGISTER_INITIAL_CONDITIONS(S40RTSPerturbation_TXlith,
                                       "S40RTS perturbation TX lithosphere",
                                       "An initial temperature field in which the temperature "
                                       "is perturbed following the S20RTS or S40RTS shear wave "
                                       "velocity model by Ritsema and others, which can be downloaded "
                                       "here \\url{http://www.earth.lsa.umich.edu/~jritsema/research.html}. "
                                       "Information on the vs model can be found in Ritsema, J., Deuss, "
                                       "A., van Heijst, H.J. \\& Woodhouse, J.H., 2011. S40RTS: a "
                                       "degree-40 shear-velocity model for the mantle from new Rayleigh "
                                       "wave dispersion, teleseismic traveltime and normal-mode "
                                       "splitting function measurements, Geophys. J. Int. 184, 1223-1236. "
                                       "The scaling between the shear wave perturbation and the "
                                       "temperature perturbation can be set by the user with the "
                                       "'vs to density scaling' parameter and the 'Thermal "
                                       "expansion coefficient in initial temperature scaling' "
                                       "parameter. The scaling is as follows: $\\delta ln \\rho "
                                       "(r,\\theta,\\phi) = \\xi \\cdot \\delta ln v_s(r,\\theta, "
                                       "\\phi)$ and $\\delta T(r,\\theta,\\phi) = - \\frac{1}{\\alpha} "
                                       "\\delta ln \\rho(r,\\theta,\\phi)$. $\\xi$ is the 'vs to "
                                       "density scaling' parameter and $\\alpha$ is the 'Thermal "
                                       "expansion coefficient in initial temperature scaling' "
                                       "parameter. The temperature perturbation is added to an "
                                       "otherwise constant temperature (incompressible model) or "
                                       "adiabatic reference profile (compressible model). If a depth "
                                       "is specified in 'Remove temperature heterogeneity down to "
                                       "specified depth', there is no temperature perturbation "
                                       "prescribed down to that depth.")
  }
}
