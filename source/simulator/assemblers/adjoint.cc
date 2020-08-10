/*
  Copyright (C) 2016 - 2020 by the authors of the ASPECT code.

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

#include <aspect/simulator.h>
#include <aspect/simulator_access.h>
#include <aspect/utilities.h>

#include <aspect/simulator/assemblers/adjoint.h>
#include <aspect/postprocess/dynamic_topography.h>

namespace aspect
{
  namespace Assemblers
  {

    template <int dim>
    struct DynTopoData
    {
      DynTopoData(MPI_Comm mpi_communicator, std::string filename)
      {
        // Read data from disk and distribute among processes
        std::string temp;
        std::istringstream in(Utilities::read_and_distribute_file_content(filename, mpi_communicator));

        getline(in,temp);  // throw away the rest of the line
        getline(in,temp);  // throw away the rest of the line

        int number_of_observations;
        in >> number_of_observations;

        for (int i=0; i<number_of_observations; i++)
          {
            Tensor<1,dim> temp_tensor;
            double tempval;

            for (int j=0; j< dim; j++)
              in >> temp_tensor[j];

            // read in location
            Point<dim> point(temp_tensor);

            measurement_locations.push_back(point);

            // read in dynamic topography value and uncertainty thereof
            in >> tempval;
            dynamic_topographies.push_back(tempval);

            in >> tempval;
            dynamic_topographies_sigma.push_back(tempval);
          }

      };

      std::vector<double> dynamic_topographies;
      std::vector<double>  dynamic_topographies_sigma;
      std::vector<Point<dim> > measurement_locations;
    };



    template <int dim>
    void
    StokesAdjointRHS<dim>::
    execute (internal::Assembly::Scratch::ScratchBase<dim>   &scratch_base,
             internal::Assembly::CopyData::CopyDataBase<dim> &data_base) const
    {
      internal::Assembly::Scratch::StokesSystem<dim> &scratch = dynamic_cast<internal::Assembly::Scratch::StokesSystem<dim>& > (scratch_base);
      internal::Assembly::CopyData::StokesSystem<dim> &data = dynamic_cast<internal::Assembly::CopyData::StokesSystem<dim>& > (data_base);

      const Introspection<dim> &introspection = this->introspection();
      const FiniteElement<dim> &fe = scratch.finite_element_values.get_fe();

      const unsigned int stokes_dofs_per_cell = data.local_dof_indices.size();
      const unsigned int n_face_q_points      = scratch.face_finite_element_values.n_quadrature_points;

      const double pressure_scaling = this->get_pressure_scaling();

      const double density_above = 0.;

      // Get a pointer to the dynamic topography postprocessor.
      const Postprocess::DynamicTopography<dim> &dynamic_topography =
        this->get_postprocess_manager().template get_matching_postprocessor<Postprocess::DynamicTopography<dim> >();

      // Get the already-computed dynamic topography solution.
      const LinearAlgebra::BlockVector &topography_vector = dynamic_topography.topography_vector();

      std::vector<double> topo_values( n_face_q_points );
      // check that the cell is at the top
      if (scratch.cell->face(scratch.face_number)->at_boundary()
          &&
          this->get_geometry_model().depth (scratch.cell->face(scratch.face_number)->center()) < scratch.cell->face(scratch.face_number)->minimum_vertex_distance()/3)
        {
          scratch.face_finite_element_values.reinit (scratch.cell, scratch.face_number);
          scratch.face_finite_element_values[introspection.extractors.temperature].get_function_values(topography_vector, topo_values);

          // check whether observation is within this cell
          // TODO current scheme of finding whether there is an observation in the current cell isn't great /
          // doesn't extend to 3D. Also doesn't check for multiple obesrvations per cell. Maybe better:
          // calculate DT everywhere and then interpolate to locations
          // TODO: currenct implementation doesn't check if there are more than two points at one location
          const Point<dim> midpoint_at_surface = scratch.cell->face(scratch.face_number)->center();

          // initiate in a way that they work if no points are read in
          bool calc_RHS = false;
          double DT_obs = 0;
          double DT_sigma = 1;


          if (this->get_parameters().read_in_points == true)
            {
              static DynTopoData<dim> observations(this->get_mpi_communicator(), this->get_parameters().adjoint_input_file);
              for (unsigned int j=0; j<observations.measurement_locations.size(); ++j)
                {
                  const Point<dim> next_measurement_location = observations.measurement_locations[j];
                  if (next_measurement_location.distance(midpoint_at_surface) < scratch.cell->face(scratch.face_number)->minimum_vertex_distance()/2)
                    {
                      calc_RHS = true;
                      DT_obs = observations.dynamic_topographies[j];
                      DT_sigma = observations.dynamic_topographies_sigma[j];
                    }
                }
            }

          // assembler a right hand side either if we are interested in the spectral kernels or if we are interested
          // in the spatial kernels and have a point in that location
          if (this->get_parameters().read_in_points == false | calc_RHS)
            {
              // TODO: interpolate to measurement location?

              double cell_surface_area = 0.0;
              double dynamic_topography_surface_average = 0.0;

              for (unsigned int q=0; q<n_face_q_points; ++q)
                {
                  dynamic_topography_surface_average += topo_values[q] * scratch.face_finite_element_values.JxW(q);
                  cell_surface_area += scratch.face_finite_element_values.JxW(q);
                }
              dynamic_topography_surface_average /= cell_surface_area;


              // ----------  Assemble RHS  ---------------

              for (unsigned int q=0; q<n_face_q_points; ++q)
                {
                  for (unsigned int i=0, i_stokes=0; i_stokes<stokes_dofs_per_cell; /*increment at end of loop*/)
                    {
                      if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
                        {
                          scratch.div_phi_u[i_stokes]   = scratch.finite_element_values[introspection.extractors.velocities].divergence (i, q);
                          scratch.phi_p[i_stokes] = scratch.finite_element_values[introspection.extractors.pressure].value (i, q);
                          scratch.grads_phi_u[i_stokes] = scratch.finite_element_values[introspection.extractors.velocities].symmetric_gradient(i,q);
                          ++i_stokes;
                        }
                      ++i;
                    }


                  const Tensor<1,dim> n_hat = scratch.face_finite_element_values.normal_vector(q);
                  const Tensor<1,dim> gravity = this->get_gravity_model().gravity_vector (scratch.finite_element_values.quadrature_point(q));
                  const double density = scratch.material_model_outputs.densities[q];
                  const double eta = scratch.material_model_outputs.viscosities[q];
                  const double JxW = scratch.face_finite_element_values.JxW(q);


                  // -------- to calculate sensitivity to specific degree
                  //const Point<dim> position = scratch.face_finite_element_values.quadrature_point(q);
                  //const std_cxx11::array<double, dim> spherical_point = Utilities::Coordinates::cartesian_to_spherical_coordinates(position);
                  //const double surface_difference = std::sin(4*spherical_point[1]);

                  for (unsigned int i=0; i<stokes_dofs_per_cell; ++i)
                    {
                      const double surface_difference = this->get_parameters().use_fixed_surface_value
                                                        ?
                                                        1
                                                        :
                                                        (dynamic_topography_surface_average - DT_obs)/DT_sigma;
                      data.local_rhs(i) += surface_difference * (2.0*eta *(n_hat * (scratch.grads_phi_u[i] * n_hat))
                                                                 - pressure_scaling *scratch.phi_p[i]) / ((density-density_above)* gravity*n_hat)
                                           * JxW; // don't divide by cell_surface_area because we're interested in cell integral
                    }
                }
            }
        }
    }

  }
} // namespace aspect

// explicit instantiation of the functions we implement in this file
namespace aspect
{
  namespace Assemblers
  {
#define INSTANTIATE(dim) \
  template class StokesAdjointRHS<dim>;

    ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
  }
}
