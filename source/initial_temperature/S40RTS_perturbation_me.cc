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


#include <aspect/initial_temperature/S40RTS_perturbation_me.h>
#include <aspect/utilities.h>
#include <fstream>
#include <iostream>
#include <deal.II/base/std_cxx11/array.h>

#include <boost/lexical_cast.hpp>

namespace aspect
{
  namespace InitialTemperature
  {
    namespace internal
    {
      namespace S40RTS
      {
        // Read in the spherical harmonics that are located in data/initial-conditions/S40RTS
        // and were downloaded from http://www.earth.lsa.umich.edu/~jritsema/research.html
        // Ritsema et al. choose real sine and cosine coefficients that follow the normalization
        // by Dahlen & Tromp, Theoretical Global Seismology (equations B.58 and B.99).

        class SphericalHarmonicsLookup
        {
          public:
            SphericalHarmonicsLookup(const std::string &filename,
                                     const MPI_Comm &comm)
            {
              std::string temp;
              // Read data from disk and distribute among processes
              std::istringstream in(Utilities::read_and_distribute_file_content(filename, comm));

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

            unsigned int maxdegree()
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
            SplineDepthsLookup(const std::string &filename,
                               const MPI_Comm &comm)
            {
              std::string temp;
              // Read data from disk and distribute among processes
              std::istringstream in(Utilities::read_and_distribute_file_content(filename, comm));

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
            VsToDensityLookup(const std::string &filename,
                              const MPI_Comm &comm)
            {
              std::string temp;
              std::istringstream in(Utilities::read_and_distribute_file_content(filename, comm));
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

              for (unsigned int i = 0; i < values.size(); i++)
                depth_diff[i] = std::abs(depthvalues[i] - depth);

              double depth_val = 1e6;
              for (unsigned int i = 0; i < values.size(); i++)
                depth_val = std::min(depth_diff[i],depth_val);

              unsigned int idx = values.size();
              for (unsigned int i = 0; i < values.size(); i++)
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

        class ContinentLookup
        {
          public:
            ContinentLookup(const std::string &filename,
                            const MPI_Comm &comm)
            {
              std::string temp;
              std::istringstream in(Utilities::read_and_distribute_file_content(filename, comm));
              AssertThrow (in,
                           ExcMessage (std::string("Couldn't open file <") + filename));

              // in >> order;
              // getline(in,temp);  // throw away the rest of the line

              // const int maxnumber = num_splines * (order+1)*(order+1);
              const int maxnumber = 65341;
              // read in all coefficients as a single data vector
              for (int i=0; i<maxnumber; i++)
                {
                  double new_val;
                  in >> new_val;
                  on_continent.push_back(new_val);
                }

            }

            // Declare a function that returns the cosine coefficients
            const std::vector<double> &continent_function() const
            {
              return on_continent;
            }

          private:
            std::vector<double> on_continent;
        };
      }
    }


    template <int dim>
    void
    S40RTSPerturbation_me<dim>::initialize()
    {
      spherical_harmonics_lookup.reset(new internal::S40RTS::SphericalHarmonicsLookup(datadirectory+harmonics_coeffs_file_name,this->get_mpi_communicator()));
      spline_depths_lookup.reset(new internal::S40RTS::SplineDepthsLookup(datadirectory+spline_depth_file_name,this->get_mpi_communicator()));

      if (vs_to_depth_constant == false)
        vs_to_density_lookup.reset(new internal::S40RTS::VsToDensityLookup(datadirectory+vs_to_density_file_name,this->get_mpi_communicator()));

      Continent_lookup.reset(new internal::S40RTS::ContinentLookup(datadirectory+ "Cont_func.txt",this->get_mpi_communicator()));
    }

    // NOTE: this module uses the Boost spherical harmonics package which is not designed
    // for very high order (> 100) spherical harmonics computation. If you use harmonic
    // perturbations of a high order be sure to confirm the accuracy first.
    // For more information, see:
    // http://www.boost.org/doc/libs/1_49_0/libs/math/doc/sf_and_dist/html/math_toolkit/special/sf_poly/sph_harm.html

    template <>
    double
    S40RTSPerturbation_me<2>::
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
    S40RTSPerturbation_me<3>::
    initial_temperature (const Point<3> &position) const
    {
      const unsigned int dim = 3;

      // use either the user-input reference temperature as background temperature
      // (incompressible model) or the adiabatic temperature profile (compressible model)
      const double background_temperature = this->get_material_model().is_compressible() ?
                                            this->get_adiabatic_conditions().temperature(position) :
                                            reference_temperature;

      //get the degree from the input file (20 or 40)
      const unsigned int maxdegree = spherical_harmonics_lookup->maxdegree();

      const int num_spline_knots = 21; // The tomography models are parameterized by 21 layers

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

      // convert coordinates from [x,y,z] to [r, phi, theta]
      std_cxx11::array<double,dim> scoord = aspect::Utilities::Coordinates::cartesian_to_spherical_coordinates(position);

      // Evaluate the spherical harmonics at this position. Since they are the
      // same for all depth splines, do it once to avoid multiple evaluations.
      // NOTE: there is apparently a factor of sqrt(2) difference
      // between the standard orthonormalized spherical harmonics
      // and those used for S40RTS (see PR # 966)
      std::vector<std::vector<double> > cosine_components(maxdegree+1,std::vector<double>(maxdegree+1,0.0));
      std::vector<std::vector<double> > sine_components(maxdegree+1,std::vector<double>(maxdegree+1,0.0));

      for (unsigned int degree_l = 0; degree_l < maxdegree+1; ++degree_l)
        {
          for (unsigned int order_m = 0; order_m < degree_l+1; ++order_m)
            {
              const std::pair<double,double> sph_harm_vals = Utilities::real_spherical_harmonic(degree_l, order_m, scoord[2], scoord[1]);
              cosine_components[degree_l][order_m] = sph_harm_vals.first;
              sine_components[degree_l][order_m] = sph_harm_vals.second;
            }
        }

      // iterate over all degrees and orders at each depth and sum them all up.
      std::vector<double> spline_values(num_spline_knots,0);
      double prefact;
      unsigned int ind = 0;

      for (unsigned int depth_interp = 0; depth_interp < num_spline_knots; depth_interp++)
        {
          for (unsigned int degree_l = 0; degree_l < maxdegree+1; degree_l++)
            {
              for (unsigned int order_m = 0; order_m < degree_l+1; order_m++)
                {
                  if (degree_l == 0)
                    prefact = (zero_out_degree_0
                               ?
                               0.
                               :
                               1.);
                  else if (order_m != 0)
                    prefact = 1./sqrt(2.);
                  else prefact = 1.0;

                  spline_values[depth_interp] += prefact * (a_lm[ind] * cosine_components[degree_l][order_m]
                                                            + b_lm[ind] * sine_components[degree_l][order_m]);

                  ++ind;
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

      // Get depth
      const double depth = this->get_geometry_model().depth(position);

      double dens_scaling;
      if (vs_to_depth_constant == true)
        dens_scaling = vs_to_density;
      else
        dens_scaling = vs_to_density_lookup -> vstodensity_scaling(depth);


      // scale the perturbation in seismic velocity into a density perturbation
      // vs_to_density is an input parameter
      double density_perturbation = dens_scaling * perturbation;


      // check whether continental lithosphere should be scaled differently
      if (include_continents == true)
        {
          // only scale if within 400km of the surface
          if (depth < 400000)
            {
              //calculate whether in continent
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

              const std::vector<double>  coeffs = Continent_lookup->continent_function();

              // bilinear interpolation from http://en.wikipedia.org/wiki/Bilinear_interpolation
              const double Q11 = coeffs[index_lonlat[0]];
              const double Q21 = coeffs[index_lonlat[1]];
              const double Q12 = coeffs[index_lonlat[2]];
              const double Q22 = coeffs[index_lonlat[3]];

              // The last term (0.01) is necessary to convert percent perturbations to absolute values
              const double cont_averaging = 1/((x2-x1) * (y2-y1)) *
                                            (Q11 *(x2 - phi)*(y2 - theta) +
                                             Q21 *(phi - x1)*(y2 - theta) +
                                             Q12 *(x2 - phi)*(theta - y1) +
                                             Q22 *(phi - x1)*(theta - y1));

              if (cont_averaging > 0.5) // you're in the ocean, don't do anything to the vs scaling
                // you're on the continent, check whether you'r in or below the craton
                {
                  const double F_tot = 0.08;
                  const double z_0 = 140000;
                  const double cont_dens_pert = 0.001;

                  const double RHS = 0.031 - F_tot * (1- erf(depth/z_0));

                  if (perturbation > RHS)
                    density_perturbation = cont_dens_pert;
                }
            }
        }

      //get thermal alpha
      double B_val, A_val;
      double thermal_alpha_gliso;
      std::vector<double>  alpha_val (3,3.5e-5);
      std::vector<double>  depth_val (3,0);

      alpha_val[1] = 2.5e-5;
      alpha_val[2] = 1.5e-5;

      depth_val[1] =  670000;
      depth_val[2] = 2890000;

      if (depth < 670000)
        {
          B_val = (alpha_val[0] - alpha_val[1])/(depth_val[0] - depth_val[1]);
          A_val = alpha_val[0] - B_val * depth_val[0];
          thermal_alpha_gliso = A_val + B_val * depth;
        }

      if (depth >= 670000)
        {
          B_val = (alpha_val[1] - alpha_val[2])/(depth_val[1] - depth_val[2]);
          A_val = alpha_val[1] - B_val * depth_val[1];
          thermal_alpha_gliso = A_val + B_val * depth;
        }

      double thermal_alpha_used;
      if (thermal_alpha_constant == true)
        thermal_alpha_used = thermal_alpha;
      else
        thermal_alpha_used = thermal_alpha_gliso;


      double temperature_perturbation;
      if (depth > no_perturbation_depth)
        // scale the density perturbation into a temperature perturbation
        temperature_perturbation =  -1./thermal_alpha_used * density_perturbation;
      else
        // set heterogeneity to zero down to a specified depth
        temperature_perturbation = 0.0;

      // add the temperature perturbation to the background temperature
      return background_temperature + temperature_perturbation;
    }


    template <int dim>
    void
    S40RTSPerturbation_me<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Initial temperature model");
      {
        prm.enter_subsection("S40RTS perturbation");
        {
          prm.declare_entry("Data directory", "$ASPECT_SOURCE_DIR/data/initial-temperature/S40RTS/",
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
          prm.declare_entry ("Reference temperature", "1600.0",
                             Patterns::Double (0),
                             "The reference temperature that is perturbed by the spherical "
                             "harmonic functions. Only used in incompressible models.");
          prm.declare_entry ("Remove temperature heterogeneity down to specified depth", boost::lexical_cast<std::string>(-std::numeric_limits<double>::max()),
                             Patterns::Double (),
                             "This will set the heterogeneity prescribed by S20RTS or S40RTS to zero "
                             "down to the specified depth (in meters).Note that your resolution has "
                             "to be adquate to capture this cutoff. For example if you specify a depth "
                             "of 660km, but your closest spherical depth layers are only at 500km and "
                             "750km (due to a coarse resolution) it will only zero out heterogeneities "
                             "down to 500km. Similar caution has to be taken when using adaptive meshing.");
          prm.declare_entry ("Vs to density scaling constant","false",
                             Patterns::Bool(),
                             "Switch to set the vs to density scalind to a constant value.");
          prm.declare_entry ("Vs to density scaling file", "R_scaling.txt",
                             Patterns::Anything(),
                             "The file name of the scaling between vs and density. "
                             "Default values are from Simmons et al., 2009.");
          prm.declare_entry ("Thermal expansion constant","false",
                             Patterns::Bool(),
                             "");
          prm.declare_entry ("Scale continental lithosphere differently", "false",
                             Patterns::Bool(),
                             "");
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();
    }


    template <int dim>
    void
    S40RTSPerturbation_me<dim>::parse_parameters (ParameterHandler &prm)
    {
      AssertThrow (dim == 3,
                   ExcMessage ("The 'S40RTS perturbation' model for the initial "
                               "temperature is only available for 3d computations."));

      prm.enter_subsection("Initial temperature model");
      {
        prm.enter_subsection("S40RTS perturbation");
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
          vs_to_density           = prm.get_double ("vs to density scaling");
          thermal_alpha           = prm.get_double ("Thermal expansion coefficient in initial temperature scaling");
          zero_out_degree_0       = prm.get_bool ("Remove degree 0 from perturbation");
          reference_temperature   = prm.get_double ("Reference temperature");
          no_perturbation_depth   = prm.get_double ("Remove temperature heterogeneity down to specified depth");
          vs_to_depth_constant    = prm.get_bool ("Vs to density scaling constant");
          vs_to_density_file_name = prm.get("Vs to density scaling file");
          thermal_alpha_constant  = prm.get_bool ("Thermal expansion constant");
          include_continents      = prm.get_bool ("Scale continental lithosphere differently");
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
  namespace InitialTemperature
  {
    ASPECT_REGISTER_INITIAL_TEMPERATURE_MODEL(S40RTSPerturbation_me,
                                              "S40RTS perturbation 2.0",
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
