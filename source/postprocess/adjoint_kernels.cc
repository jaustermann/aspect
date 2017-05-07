/*
  Copyright (C) 2011 - 2016 by the authors of the ASPECT code.

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


#include <aspect/postprocess/adjoint_kernels.h>
#include <aspect/utilities.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

namespace aspect
{
  namespace Postprocess
  {

//    template <int dim>
//    struct DynTopoData
//    {
//      DynTopoData(MPI_Comm mpi_communicator)
//      {

//        std::string temp;
        // Read data from disk and distribute among processes
//        std::string filename = "/Users/jackyaustermann/Desktop/Aspect_code/aspect/data/adjoint-observations/dynamic_topography_observations.txt";
//        std::istringstream in(Utilities::read_and_distribute_file_content(filename, mpi_communicator));

//        getline(in,temp);  // throw away the rest of the line
//        getline(in,temp);  // throw away the rest of the line

//        int number_of_observations;
//        in >> number_of_observations;

//        for (int i=0; i<number_of_observations; i++)
//          {
//            double x_val, y_val, tempval;
//            in >> x_val;
//            in >> y_val;
//            Point<dim> point(x_val,y_val);
//            measurement_locations.push_back(point);

//            in >> tempval;
//            dynamic_topographies.push_back(tempval);

//            in >> tempval;
//            dynamic_topographies_sigma.push_back(tempval);
//          }
//      };

//      std::vector<double> dynamic_topographies;
//      std::vector<double>  dynamic_topographies_sigma;
//      std::vector <Point<dim>> measurement_locations;
//    };



    template <int dim>
    std::pair<std::string,std::string>
    AdjointKernels<dim>::execute (TableHandler &)
    {
      const unsigned int quadrature_degree = this->get_fe().base_element(this->introspection().base_elements.velocities).degree;
      const QGauss<dim> quadrature_formula(quadrature_degree);
//      const QGauss<dim-1> quadrature_formula_face(quadrature_degree);

      FEValues<dim> fe_values (this->get_mapping(),
                               this->get_fe(),
                               quadrature_formula,
                               update_values |
                               update_gradients |
                               update_q_points |
                               update_JxW_values);

      // compute DT at surface as that is how it's done in kernel calculation
//      FEFaceValues<dim> fe_face_values (this->get_mapping(),
//                                        this->get_fe(),
//                                        quadrature_formula_face,
//                                        update_values |
//                                        update_gradients |
//                                        update_q_points |
//                                        update_JxW_values);

      FEValues<dim> fe_values_adjoint (this->get_mapping(),
                                       this->get_fe(),
                                       quadrature_formula,
                                       update_values |
                                       update_gradients |
                                       update_q_points |
                                       update_JxW_values);

      MaterialModel::MaterialModelInputs<dim> in(fe_values.n_quadrature_points, this->n_compositional_fields());
      MaterialModel::MaterialModelOutputs<dim> out(fe_values.n_quadrature_points, this->n_compositional_fields());
      MaterialModel::MaterialModelInputs<dim> in_adjoint(fe_values_adjoint.n_quadrature_points, this->n_compositional_fields());

//      MaterialModel::MaterialModelInputs<dim> in_face(fe_face_values.n_quadrature_points, this->n_compositional_fields());
//      MaterialModel::MaterialModelOutputs<dim> out_face(fe_face_values.n_quadrature_points, this->n_compositional_fields());

      // have a stream into which we write the data. the text stream is then
      // later sent to processor 0
      std::ostringstream output;

      typename DoFHandler<dim>::active_cell_iterator
      cell = this->get_dof_handler().begin_active(),
      endc = this->get_dof_handler().end();

//      static DynTopoData<dim> observations(this->get_mpi_communicator());
//      const double density_above = 0;

      std::vector <double> kernel_density;
      std::vector <double> kernel_viscosity;
      std::vector<Point<dim> > location;

//      double viscosity_kernel_term = 0;
//      double density_kernel_term = 0;

//      for (; cell!=endc; ++cell)
//        if (cell->is_locally_owned())
//          {
            // calculate DT contribution
//            if (cell->at_boundary())
//              {
                // see if the cell is at the *top* boundary, not just any boundary
//                unsigned int top_face_idx = numbers::invalid_unsigned_int;
//                {
//                  for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
//                    if (cell->at_boundary(f) && this->get_geometry_model().depth (cell->face(f)->center()) < cell->face(f)->minimum_vertex_distance()/3)
//                      {
//                        top_face_idx = f;
//                        break;
//                      }

//                  if (top_face_idx == numbers::invalid_unsigned_int)
//                    continue;
//                }

 //               fe_face_values.reinit (cell,top_face_idx);

//                const QMidpoint<dim> quadrature_formula;
//                const Point<dim> midpoint_at_surface = cell->face(top_face_idx)->center();

 //               for (unsigned int j=0; j<observations.measurement_locations.size(); ++j)
//                  {

//                    const Point<dim> next_measurement_location = observations.measurement_locations[j];
//                    if (next_measurement_location.distance(midpoint_at_surface) < cell->face(top_face_idx)->minimum_vertex_distance()/2) //* sqrt(2))
//                      {

//                        double density_kernel_factor_x_surface = 0;
//                        double viscosity_kernel_factor_x_surface = 0;
//                        double dynamic_topography_x_surface = 0;
//                        double surface = 0;

//                        fe_face_values[this->introspection().extractors.temperature]
//                        .get_function_values (this->get_solution(), in_face.temperature);
//                        fe_face_values[this->introspection().extractors.pressure]
//                        .get_function_values (this->get_solution(), in_face.pressure);
 //                       fe_face_values[this->introspection().extractors.velocities]
//                        .get_function_values (this->get_solution(), in_face.velocity);
//                        fe_face_values[this->introspection().extractors.velocities]
//                        .get_function_symmetric_gradients (this->get_solution(), in_face.strain_rate);
//                        fe_face_values[this->introspection().extractors.pressure]
//                        .get_function_gradients (this->get_solution(), in_face.pressure_gradient);

//                        in_face.position = fe_face_values.get_quadrature_points();
//                        in_face.cell = &cell;

//                        this->get_material_model().evaluate(in_face, out_face);

                        // Compute the integral of the dynamic topography function
                        // over the entire cell, by looping over all quadrature points
//                        for (unsigned int q=0; q<quadrature_formula_face.size(); ++q)
//                          {
//                            Point<dim> location = fe_face_values.quadrature_point(q);
//                            const double viscosity = out_face.viscosities[q];
//                            const double density   = out_face.densities[q];

//                            const SymmetricTensor<2,dim> strain_rate = in_face.strain_rate[q] - 1./3 * trace(in_face.strain_rate[q]) * unit_symmetric_tensor<dim>();
//                            const SymmetricTensor<2,dim> shear_stress = 2 * viscosity * strain_rate;

//                            const Tensor<1,dim> gravity = this->get_gravity_model().gravity_vector(location);
//                            const Tensor<1,dim> gravity_direction = gravity/gravity.norm();

                            // Subtract the dynamic pressure
//                            const double dynamic_pressure   = in_face.pressure[q] - this->get_adiabatic_conditions().pressure(location);
//                            const double sigma_rr           = gravity_direction * (shear_stress * gravity_direction) - dynamic_pressure;
//                            const double dynamic_topography = - sigma_rr / gravity.norm() / (density - density_above);

//                            const double viscosity_kernel_factor = - sigma_rr / gravity.norm() / (density - density_above);
//                            const double density_kernel_factor = sigma_rr / gravity.norm() / ((density - density_above)*(density - density_above));

                            // JxW provides the volume quadrature weights. This is a general formulation
                            // necessary for when a quadrature formula is used that has more than one point.

 //                           viscosity_kernel_factor_x_surface += viscosity_kernel_factor * fe_face_values.JxW(q);
 //                           density_kernel_factor_x_surface += density_kernel_factor * fe_face_values.JxW(q);
 //                           dynamic_topography_x_surface += dynamic_topography * fe_face_values.JxW(q);
 //                           surface += fe_face_values.JxW(q);

//                          }

//                        const double misfit = (dynamic_topography_x_surface / surface - observations.dynamic_topographies[j])/observations.dynamic_topographies_sigma[j];
//                        viscosity_kernel_term += misfit * viscosity_kernel_factor_x_surface / surface;
//                        density_kernel_term += misfit * density_kernel_factor_x_surface / surface;
//                      }
//                  }
//              }
//          }

//      const double viscosity_kernel_term_allMPIs = Utilities::MPI::sum (viscosity_kernel_term,this->get_mpi_communicator());
//      const double density_kernel_term_allMPIs = Utilities::MPI::sum (density_kernel_term,this->get_mpi_communicator());

//      std::cout << "visc kernel " << viscosity_kernel_term_allMPIs << std::endl;
//      std::cout << "density kernel " << density_kernel_term_allMPIs << std::endl;

//      cell = this->get_dof_handler().begin_active();

      for (; cell!=endc; ++cell)
        if (cell->is_locally_owned())
          {
            fe_values.reinit (cell);
            fe_values_adjoint.reinit (cell);

            // get the various components of the solution, then
            // evaluate the material properties there
            //fe_values[this->introspection().extractors.velocities]
            //.get_function_values (this->get_solution(), in.velocity);

            const Point<dim> midpoint_of_cell = cell->center();
            in.position = fe_values.get_quadrature_points();

            fe_values[this->introspection().extractors.velocities]
            .get_function_symmetric_gradients (this->get_solution(), in.strain_rate);
            fe_values[this->introspection().extractors.temperature]
            .get_function_values (this->get_solution(), in.temperature);
            fe_values[this->introspection().extractors.pressure]
            .get_function_values (this->get_solution(), in.pressure);
            fe_values[this->introspection().extractors.velocities]
            .get_function_values (this->get_solution(), in.velocity);
            fe_values[this->introspection().extractors.pressure]
            .get_function_gradients (this->get_solution(), in.pressure_gradient);

            this->get_material_model().evaluate(in, out);

            fe_values_adjoint[this->introspection().extractors.velocities]
            .get_function_values (this->get_current_adjoint_solution(), in_adjoint.velocity);
            fe_values_adjoint[this->introspection().extractors.velocities]
            .get_function_symmetric_gradients (this->get_current_adjoint_solution(), in_adjoint.strain_rate);

            double kernel_density_temp = 0;
            double kernel_viscosity_temp = 0;

            // over the entire cell, by looping over all quadrature points
            for (unsigned int q=0; q<quadrature_formula.size(); ++q)
              {
                Point<dim> location = fe_values.quadrature_point(q);
                // this has to be viscosity from base model, not sure this is guaranteed
        //        const double reference_viscosity = out.viscosities[q];
                const SymmetricTensor<2,dim> strain_rate_forward = in.strain_rate[q] - 1./3 * trace(in.strain_rate[q]) * unit_symmetric_tensor<dim>();
                const SymmetricTensor<2,dim> strain_rate_adjoint = in_adjoint.strain_rate[q] - 1./3 * trace(in_adjoint.strain_rate[q]) * unit_symmetric_tensor<dim>();
                const Tensor<1,dim> velocity_adjoint = in_adjoint.velocity[q];
                const Tensor<1,dim> gravity = this->get_gravity_model().gravity_vector(location);

                kernel_density_temp += (gravity*velocity_adjoint) * fe_values.JxW(q);
                kernel_viscosity_temp += (-2.0 * (strain_rate_adjoint*strain_rate_forward))* fe_values.JxW(q);
              }

// this should be commented in but it shifits kernels by too much, not sure if it makes sense in the strong form
//            kernel_density_temp += density_kernel_term_allMPIs;
//            kernel_viscosity_temp += viscosity_kernel_term_allMPIs;

            kernel_density.push_back(kernel_density_temp);
            kernel_viscosity.push_back(kernel_viscosity_temp);
            location.push_back(midpoint_of_cell);
          }


      // Write the solution to an output file
      // if (DT_mean_switch == true) subtract the average dynamic topography,
      // otherwise leave as is
      for (unsigned int i=0; i<kernel_viscosity.size(); ++i)
        {
          output << location[i]
                 << ' '
                 << kernel_density[i]
                 << ' '
                 << kernel_viscosity[i]
                 << ' '
                 << std::endl;
        }


      const std::string filename = this->get_output_directory() +
                                   "adjoint_kernel." +
                                   Utilities::int_to_string(this->get_timestep_number(), 5);

      const unsigned int max_data_length = Utilities::MPI::max (output.str().size()+1,
                                                                this->get_mpi_communicator());
      const unsigned int mpi_tag = 123;

      // on processor 0, collect all of the data the individual processors send
      // and concatenate them into one file
      if (Utilities::MPI::this_mpi_process(this->get_mpi_communicator()) == 0)
        {
          std::ofstream file (filename.c_str());

          file << "# "
               << ((dim==2)? "x y" : "x y z")
               << " density  viscosity" << std::endl;

          // first write out the data we have created locally
          file << output.str();

          std::string tmp;
          tmp.resize (max_data_length, '\0');

          // then loop through all of the other processors and collect
          // data, then write it to the file
          for (unsigned int p=1; p<Utilities::MPI::n_mpi_processes(this->get_mpi_communicator()); ++p)
            {
              MPI_Status status;
              // get the data. note that MPI says that an MPI_Recv may receive
              // less data than the length specified here. since we have already
              // determined the maximal message length, we use this feature here
              // rather than trying to find out the exact message length with
              // a call to MPI_Probe.
              MPI_Recv (&tmp[0], max_data_length, MPI_CHAR, p, mpi_tag,
                        this->get_mpi_communicator(), &status);

              // output the string. note that 'tmp' has length max_data_length,
              // but we only wrote a certain piece of it in the MPI_Recv, ended
              // by a \0 character. write only this part by outputting it as a
              // C string object, rather than as a std::string
              file << tmp.c_str();
            }
        }
      else
        // on other processors, send the data to processor zero. include the \0
        // character at the end of the string
        {
          MPI_Send (&output.str()[0], output.str().size()+1, MPI_CHAR, 0, mpi_tag,
                    this->get_mpi_communicator());
        }

      return std::pair<std::string,std::string>("Writing adjoint kernels:",
                                                filename);
    }

  }
}


// explicit instantiations
namespace aspect
{
  namespace Postprocess
  {
    ASPECT_REGISTER_POSTPROCESSOR(AdjointKernels,
                                  "adjoint kernels",
                                  "A postprocessor that computes a measure of dynamic topography "
                                  "based on the stress at the surface. The data is written into text "
                                  "files named 'dynamic\\_topography.NNNNN' in the output directory, "
                                  "where NNNNN is the number of the time step."
                                  "\n\n"
                                  "The exact approach works as follows: At the centers of all cells "
                                  "that sit along the top surface, we evaluate the stress and "
                                  "evaluate the component of it in the direction in which "
                                  "gravity acts. In other words, we compute "
                                  "$\\sigma_{rr}={\\hat g}^T(2 \\eta \\varepsilon(\\mathbf u)- \\frac 13 (\\textrm{div}\\;\\mathbf u)I)\\hat g - p_d$ "
                                  "where $\\hat g = \\mathbf g/\\|\\mathbf g\\|$ is the direction of "
                                  "the gravity vector $\\mathbf g$ and $p_d=p-p_a$ is the dynamic "
                                  "pressure computed by subtracting the adiabatic pressure $p_a$ "
                                  "from the total pressure $p$ computed as part of the Stokes "
                                  "solve. From this, the dynamic "
                                  "topography is computed using the formula "
                                  "$h=\\frac{\\sigma_{rr}}{\\|\\mathbf g\\| \\rho}$ where $\\rho$ "
                                  "is the density at the cell center."
                                  "\n"
                                  "The file format then consists of lines with Euclidiean coordinates "
                                  "followed by the corresponding topography value."
                                  "\n\n"
                                  "(As a side note, the postprocessor chooses the cell center "
                                  "instead of the center of the cell face at the surface, where we "
                                  "really are interested in the quantity, since "
                                  "this often gives better accuracy. The results should in essence "
                                  "be the same, though.)")
  }
}
