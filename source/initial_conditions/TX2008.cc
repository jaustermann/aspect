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


#include <aspect/initial_conditions/TX2008.h>
#include <aspect/geometry_model/spherical_shell.h>
#include <fstream>
#include <iostream>
#include <deal.II/base/std_cxx1x/array.h>
#include <aspect/utilities.h>

namespace aspect
{
  namespace InitialConditions
  {
    
    namespace internal
    {
       // Read in the spherical harmonics that are located in data/initial-conditions/S40RTS
       // and were downloaded from http://www.earth.lsa.umich.edu/~jritsema/research.html
       // Ritsema et al. choose real sine and cosine coefficients that follow the normalization
       // by Dahlen & Tromp, Theoretical Global Seismology (equations B.58 and B.99). 

       class TX2008Lookup
       {
         public:
         TX2008Lookup(const std::string &filename,
                                     const MPI_Comm &comm)
         {
           std::string temp;
           std::istringstream in(Utilities::read_and_distribute_file_content(filename, comm));
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


    template <int dim>
    void
    TX2008Perturbation<dim>::initialize()
    {
      TX2008_lookup.reset(new internal::TX2008Lookup(datadirectory+TX2008_file_name,this->get_mpi_communicator()));

      if (read_geotherm_in == true)
        geotherm_lookup.reset(new internal::GeothermLookup(datadirectory+geotherm_file_name));
     }

    // NOTE: this module uses the Boost spherical harmonics package which is not designed
    // for very high order (> 100) spherical harmonics computation. If you use harmonic
    // perturbations of a high order be sure to confirm the accuracy first.
    // For more information, see:
    // http://www.boost.org/doc/libs/1_49_0/libs/math/doc/sf_and_dist/html/math_toolkit/special/sf_poly/sph_harm.html
    
    template <int dim>
    double
    TX2008Perturbation<dim>::
    initial_temperature (const Point<dim> &position) const
    {

     // this initial condition only makes sense if the geometry is a
     // spherical shell. verify that it is indeed
     AssertThrow (dynamic_cast<const GeometryModel::SphericalShell<dim>*>(&this->get_geometry_model())
                   != 0,
                   ExcMessage ("This initial condition can only be used if the geometry "
                               "is a spherical shell."));

        
     // depth values over which gypsum is defined
     double depth_values[] = {0, 100, 175, 250, 325, 400, 525, 650, 750, 850, 1000, 1150, 1300, 1450, 1600, 1750, 1900, 2050, 2200, 2350, 2500, 2650, 2891};
     
     for (int i=0; i<23; i++)
       depth_values[i] *= 1000.;

     // convert coordinates from [x,y,z] to [r, phi, theta] 
     std_cxx1x::array<double,dim> scoord = spherical_surface_coordinates(position);

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
           


     double density_perturbation = perturbation;

     if (take_upper_200km_out == true)
       if (depth <= 200000)
          density_perturbation = 0;

     //get thermal alpha
     double thermal_alpha_val;
     double B_val, A_val;

     std::vector<double>  alpha_val (3,3.5e-5);
     std::vector<double>  depth_val (3,0);

     alpha_val[1] = 2.5e-5;
     alpha_val[2] = 1.0e-5;

     depth_val[1] =  670000;
     depth_val[2] = 2890000;

     if (depth < 670000)
       {
        B_val = (alpha_val[0] - alpha_val[1])/(depth_val[0] - depth_val[1]);
        A_val = alpha_val[0] - B_val * depth_val[0];
        thermal_alpha_val = A_val + B_val * depth;
        }

     if (depth >= 670000)
       thermal_alpha_val = 3.48e-05 + 2.72e-18 * depth*depth - 1.644e-11*depth;
//       {
//        B_val = (alpha_val[1] - alpha_val[2])/(depth_val[1] - depth_val[2]);
//        A_val = alpha_val[1] - B_val * depth_val[1];
 //       thermal_alpha_val = A_val + B_val * depth;
 //       }

      if (thermal_alpha_constant == true)
        thermal_alpha_val = thermal_alpha;

     // scale the density perturbation into a temperature perturbation
     // THIS ISNT COMPRESSIBLE - GLISOVIC ET AL 2012 THAT ITS THIRD ORDER EFFECT
     const double temperature_perturbation =  -1./thermal_alpha_val * density_perturbation;

     double temperature;

     // set up background temperature as a geotherm
      /*       Note that the values we read in here have reasonable default values equation to
       the following:*/
/* 
      if(read_geotherm_in == false)
      { 
// start geotherm stuff
      std::vector<double> geotherm (4,0);
      std::vector<double> radial_position (4,0);
      geotherm[0] = 1e0;
      geotherm[1] = 0.75057142857142856;
      geotherm[2] = 0.32199999999999995;
      geotherm[3] = 0.0;
      radial_position[0] =  0e0-1e-3;
      radial_position[1] =  0.16666666666666666;
      radial_position[2] =  0.83333333333333337;
      radial_position[3] =  1e0+1e-3;

      const double
      R0 = dynamic_cast<const GeometryModel::SphericalShell<dim>&> (this->get_geometry_model()).inner_radius(),
      R1 = dynamic_cast<const GeometryModel::SphericalShell<dim>&> (this->get_geometry_model()).outer_radius();
      const double dT = this->get_boundary_temperature().maximal_temperature()
                        - this->get_boundary_temperature().minimal_temperature();
      const double T0 = this->get_boundary_temperature().maximal_temperature()/dT;
      const double T1 = this->get_boundary_temperature().minimal_temperature()/dT;
      const double h = R1-R0;

      // s = fraction of the way from
      // the inner to the outer
      // boundary; 0<=s<=1
      const double r_geotherm = position.norm();
      const double s_geotherm  = (r_geotherm-R0)/h;

      const double scale=R1/(R1 - R0);
      const float eps = 1e-4;

      int indx = -1;
      for (unsigned int i=0; i<3; ++i)
        {
          if ((radial_position[i] - s_geotherm) < eps && (radial_position[i+1] - s_geotherm ) > eps)
            {
              indx = i;
              break;
            }
        }
      Assert (indx >= 0, ExcInternalError());
      Assert (indx < 3,  ExcInternalError());
      int indx1 = indx + 1;
      const float dx = radial_position[indx1] - radial_position[indx];
      const float dy = geotherm[indx1] - geotherm[indx];

      const double InterpolVal    = (( dx > 0.5*eps)
                                  ?
                                  // linear interpolation
                                  std::max(geotherm[3],geotherm[indx] + (s_geotherm -radial_position[indx]) * (dy/dx))
                                  :
                                  // evaluate the point in the discontinuity
                                  0.5*( geotherm[indx] + geotherm[indx1] ));

       temperature = InterpolVal * dT + temperature_perturbation;
      }
 
 */    // option to either take simplified geotherm or read one in from file
     if(read_geotherm_in == true)
         temperature = geotherm_lookup->geotherm(depth) + temperature_perturbation;
 
     
     if(adiabat_temp == true)
         temperature = this->get_adiabatic_conditions().temperature(position) + temperature_perturbation;                                                
 
     if(constant_temp == true)        
         temperature = reference_temperature + temperature_perturbation;
 
     return temperature;

    }

    template <int dim>
    std_cxx1x::array<double,dim>
    TX2008Perturbation<dim>::
    spherical_surface_coordinates(const dealii::Point<dim,double> &position) 
    {
      std_cxx1x::array<double,dim> scoord;

      scoord[0] = std::sqrt(position.norm_square()); // R
      scoord[1] = std::atan2(position[1],position[0]); // Phi
      if (scoord[1] < 0.0) 
        scoord[1] = 2*numbers::PI + scoord[1]; // correct phi to [0,2*pi]
      
      if (dim==3)
        scoord[2] = std::acos(position[2]/std::sqrt(position.norm_square())); // Theta

      return scoord;
    } 



    template <int dim>
    void
    TX2008Perturbation<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Initial conditions");
      { 
          prm.enter_subsection("TX2008 perturbation");
          {
          prm.declare_entry("Data directory", "$ASPECT_SOURCE_DIR/data/initial-conditions/TX2008/",
                            Patterns::DirectoryName (),
                             "The path to the model data. ");
          prm.declare_entry ("Initial condition file name", "TX2008_dens.txt",
                            Patterns::Anything(),
                             "GyPSuM Model from Simmons et al..");
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
          prm.declare_entry ("Constant background temperature","false",
                             Patterns::Bool(),
                             "Switch to make the background temp. constant. Good to check "
                             "initial perturbation.");
          prm.declare_entry ("Zero out heterogeneity within 200km of surface", "false",
                             Patterns::Bool(),
                             "Switch to zero out density heterogeneities in upper "
                             "200km of Earth's mantle.");
          prm.declare_entry ("Use adiabat as background temperature", "false",
                             Patterns::Bool(),
                             "Switch to use the adiabat as background temp. Only possible "
                             "when const temp is set to false.");
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();
    }


    template <int dim>
    void
    TX2008Perturbation<dim>::parse_parameters (ParameterHandler &prm)
    {

      prm.enter_subsection("Initial conditions");
      {
        prm.enter_subsection("TX2008 perturbation");
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
          TX2008_file_name = prm.get ("Initial condition file name");
          geotherm_file_name      = prm.get ("Geotherm file name");
          vs_to_density           = prm.get_double ("vs to density scaling");
          thermal_alpha           = prm.get_double ("Thermal expansion coefficient in initial temperature scaling");
          read_geotherm_in        = prm.get_bool ("Read geotherm in from file");
          reference_temperature   = prm.get_double ("Reference temperature");
          thermal_alpha_constant  = prm.get_bool ("Thermal expansion constant");
          vs_to_depth_constant    = prm.get_bool ("Vs to density scaling constant");
          constant_temp           = prm.get_bool ("Constant background temperature");
          take_upper_200km_out    = prm.get_bool ("Zero out heterogeneity within 200km of surface");
          adiabat_temp            = prm.get_bool ("Use adiabat as background temperature");
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
    ASPECT_REGISTER_INITIAL_CONDITIONS(TX2008Perturbation,
                                       "TX2008 perturbation",
                                       "An initial temperature field in which the temperature "
                                       "is perturbed following the S20RTS or S40RTS shear wave "
                                       "velocity model by Ritsema and others, which can be downloaded " 
                                       "here \\url{http://www.earth.lsa.umich.edu/~jritsema/research.html}. "
                                       "Information on the vs model can be found in Ritsema, J., Deuss, " 
                                       "A., van Heijst, H.J. & Woodhouse, J.H., 2011. S40RTS: a " 
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
                                       "adiabatic reference profile (compressible model).")
  }
}
