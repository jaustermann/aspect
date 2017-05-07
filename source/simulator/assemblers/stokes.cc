/*
  Copyright (C) 2016 by the authors of the ASPECT code.

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

#include <aspect/simulator.h>
#include <aspect/utilities.h>
#include <aspect/assembly.h>
#include <aspect/simulator_access.h>

namespace aspect
{
  namespace Assemblers
  {
    template <int dim>
    void
    StokesAssembler<dim>::
    preconditioner (const double                                             pressure_scaling,
                    internal::Assembly::Scratch::StokesPreconditioner<dim>  &scratch,
                    internal::Assembly::CopyData::StokesPreconditioner<dim> &data) const
    {
      const Introspection<dim> &introspection = this->introspection();
      const FiniteElement<dim> &fe = this->get_fe();
      const unsigned int stokes_dofs_per_cell = data.local_dof_indices.size();
      const unsigned int n_q_points           = scratch.finite_element_values.n_quadrature_points;

      // First loop over all dofs and find those that are in the Stokes system
      // save the component (pressure and dim velocities) each belongs to.
      for (unsigned int i = 0, i_stokes = 0; i_stokes < stokes_dofs_per_cell; /*increment at end of loop*/)
        {
          if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
            {
              scratch.dof_component_indices[i_stokes] = fe.system_to_component_index(i).first;
              ++i_stokes;
            }
          ++i;
        }

      // Loop over all quadrature points and assemble their contributions to
      // the preconditioner matrix
      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          for (unsigned int i = 0, i_stokes = 0; i_stokes < stokes_dofs_per_cell; /*increment at end of loop*/)
            {
              if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
                {
                  scratch.grads_phi_u[i_stokes] =
                    scratch.finite_element_values[introspection.extractors
                                                  .velocities].symmetric_gradient(i, q);
                  scratch.phi_p[i_stokes] = scratch.finite_element_values[introspection
                                                                          .extractors.pressure].value(i, q);
                  ++i_stokes;
                }
              ++i;
            }

          const double eta = scratch.material_model_outputs.viscosities[q];
          const double one_over_eta = 1. / eta;

          const SymmetricTensor<4, dim> &stress_strain_director = scratch
                                                                  .material_model_outputs.stress_strain_directors[q];
          const bool use_tensor = (stress_strain_director
                                   != dealii::identity_tensor<dim>());

          const double JxW = scratch.finite_element_values.JxW(q);

          for (unsigned int i = 0; i < stokes_dofs_per_cell; ++i)
            for (unsigned int j = 0; j < stokes_dofs_per_cell; ++j)
              if (scratch.dof_component_indices[i] ==
                  scratch.dof_component_indices[j])
                data.local_matrix(i, j) += ((
                                              use_tensor ?
                                              eta * (scratch.grads_phi_u[i]
                                                     * stress_strain_director
                                                     * scratch.grads_phi_u[j]) :
                                              eta * (scratch.grads_phi_u[i]
                                                     * scratch.grads_phi_u[j]))
                                            + one_over_eta * pressure_scaling
                                            * pressure_scaling
                                            * (scratch.phi_p[i] * scratch
                                               .phi_p[j]))
                                           * JxW;
        }
    }



    template <int dim>
    struct DynTopoData
    {
      DynTopoData(MPI_Comm mpi_communicator)
      {

        std::string temp;
        // Read data from disk and distribute among processes
        std::string filename = "/Users/jackyaustermann/Desktop/Aspect_code/aspect/data/adjoint-observations/dynamic_topography_observations.txt";
        std::istringstream in(Utilities::read_and_distribute_file_content(filename, mpi_communicator));

        getline(in,temp);  // throw away the rest of the line
        getline(in,temp);  // throw away the rest of the line

        int number_of_observations;
        in >> number_of_observations;

        for (int i=0; i<number_of_observations; i++)
          {
            double x_val, y_val, tempval;
            in >> x_val;
            in >> y_val;
            Point<dim> point(x_val,y_val);
            measurement_locations.push_back(point);

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
    StokesAssembler<dim>::
    adjoint_rhs (const typename DoFHandler<dim>::active_cell_iterator &cell,
                 const unsigned int                                    face_no,
                 const double                                     pressure_scaling,
                 internal::Assembly::Scratch::StokesSystem<dim>       &scratch,
                 internal::Assembly::CopyData::StokesSystem<dim>      &data) const
    {
      //only do this if we want the adjoint RHS

      if (this->get_adjoint_problem() == true)
        {

          const Introspection<dim> &introspection = this->introspection();
          const FiniteElement<dim> &fe = this->get_fe();
          const unsigned int stokes_dofs_per_cell = data.local_dof_indices.size();

          // read in data
          static DynTopoData<dim> observations(this->get_mpi_communicator());

          // find the top face
          if (cell->face(face_no)->at_boundary()
              &&
              this->get_geometry_model().depth (cell->face(face_no)->center()) < cell->face(face_no)->minimum_vertex_distance()/3)
            {
              scratch.face_finite_element_values.reinit (cell, face_no);

              this->compute_material_model_input_values (this->get_current_linearization_point(),
                                                         scratch.face_finite_element_values,
                                                         cell,
                                                         true,
                                                         scratch.face_material_model_inputs);

              this->get_material_model().evaluate(scratch.face_material_model_inputs,
                                                  scratch.face_material_model_outputs);


              // ----------  Check whether there's an observation in the current cell  ---------------

              // get the midpoint at the surface face
              const QMidpoint<dim> quadrature_formula;
              const Point<dim> midpoint_at_surface = cell->face(face_no)->center();

              // loop over all DT observations to find whether the current cell has any observations in them
              // if so add contribution to adjoint RHS
              // TODO: current scheme doesn't fully extend to 3D

 
              //COMMENT BACK IN
            //  for (unsigned int j=0; j<observations.measurement_locations.size(); ++j)
            //    {
            //      const Point<dim> next_measurement_location = observations.measurement_locations[j];
            //      if (next_measurement_location.distance(midpoint_at_surface) < cell->face(face_no)->minimum_vertex_distance()/2) //* sqrt(2))
            //        {


                      // ----------  Calculate DT from forward solution ---------------

                      // TODO - test changing to volume integral

                      double dynamic_topography_x_surface = 0;
                      double surface = 0;
                      const double density_above = 0;

                      // Compute the integral of the dynamic topography function
                      // over the entire cell, by looping over all quadrature points
                      for (unsigned int q=0; q<scratch.face_finite_element_values.n_quadrature_points; ++q)
                        {
                          const Point<dim> location = scratch.face_finite_element_values.quadrature_point(q);
                          const double viscosity = scratch.face_material_model_outputs.viscosities[q];
                          const double density   = scratch.face_material_model_outputs.densities[q];

                          const SymmetricTensor<2,dim> strain_rate_in = scratch.face_material_model_inputs.strain_rate[q];
                          const SymmetricTensor<2,dim> strain_rate = strain_rate_in - 1./3 * trace(strain_rate_in) * unit_symmetric_tensor<dim>();
                          const SymmetricTensor<2,dim> shear_stress = 2 * viscosity * strain_rate;

                          const Tensor<1,dim> gravity = this->get_gravity_model().gravity_vector (scratch.face_finite_element_values.quadrature_point(q));;
                          const Tensor<1,dim> gravity_direction = gravity/gravity.norm();

                          // Subtract the dynamic pressure
                          const double dynamic_pressure   = scratch.face_material_model_inputs.pressure[q] - this->get_adiabatic_conditions().pressure(location);
                          const double sigma_rr           = gravity_direction * (shear_stress * gravity_direction) - dynamic_pressure;
                          const double dynamic_topography = - sigma_rr / gravity.norm() / (density - density_above);


                          // JxW provides the volume quadrature weights. This is a general formulation
                          // necessary for when a quadrature formula is used that has more than one point.
                          dynamic_topography_x_surface += dynamic_topography * scratch.face_finite_element_values.JxW(q);
                          surface += scratch.face_finite_element_values.JxW(q);
                        }

                      // NEED TO ACTUALLY CALCULATE THE AVERAGE
                      const double average_dynamic_topography = 0;
                      const double dynamic_topography_cell_average = dynamic_topography_x_surface / surface - average_dynamic_topography;

//                      std::cout << "*** DT at point 4450000, 4510254 is " << dynamic_topography_cell_average << std::endl;
                      //std::cout << "*** Found face " << cell->face(face_no)->center() << std::endl;


                      // ----------  Assemble RHS  ---------------

                      for (unsigned int q=0; q<scratch.face_finite_element_values.n_quadrature_points; ++q)
                        {


                          for (unsigned int i=0, i_stokes=0; i_stokes<stokes_dofs_per_cell; /*increment at end of loop*/)
                            {
                              if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
                                {
                                  scratch.phi_p[i_stokes] = scratch.finite_element_values[introspection.extractors.pressure].value (i, q);
                                  scratch.grads_phi_u[i_stokes] = scratch.finite_element_values[introspection.extractors.velocities].symmetric_gradient(i,q);
                                  ++i_stokes;
                                }
                              ++i;
                            }


                          const Tensor<1,dim> n_hat = scratch.face_finite_element_values.normal_vector(q);
                          const Tensor<1,dim> gravity = this->get_gravity_model().gravity_vector (scratch.finite_element_values.quadrature_point(q));
                          const double dens_surf = 3300;
                          const double eta = scratch.material_model_outputs.viscosities[q];
                          const double JxW = scratch.face_finite_element_values.JxW(q);

                          for (unsigned int i=0; i<stokes_dofs_per_cell; ++i)
                            {
                              data.local_rhs(i) += dynamic_topography_cell_average  //100.0 //dynamic_topography_cell_average 
//COMMENT BACK IN
//(dynamic_topography_cell_average - observations.dynamic_topographies[j])/observations.dynamic_topographies_sigma[j]
                                                   * (2.0*eta *(n_hat * (scratch.grads_phi_u[i] * n_hat))
                                                      -  pressure_scaling *scratch.phi_p[i]) / (dens_surf* gravity*n_hat)
                                                   *JxW / surface;
                            }
                        }
                      //std::cout << "*** " << data.local_rhs.l2_norm() << std::endl;
//COMMENT BACK IN
              //      }
              //  }

            }

        }
    }


    template <int dim>
    void
    StokesAssembler<dim>::
    incompressible_terms (const double                                     pressure_scaling,
                          const bool                                       rebuild_stokes_matrix,
                          internal::Assembly::Scratch::StokesSystem<dim>  &scratch,
                          internal::Assembly::CopyData::StokesSystem<dim> &data) const
    {
      const Introspection<dim> &introspection = this->introspection();
      const FiniteElement<dim> &fe = this->get_fe();
      const unsigned int stokes_dofs_per_cell = data.local_dof_indices.size();
      const unsigned int n_q_points    = scratch.finite_element_values.n_quadrature_points;

      for (unsigned int q=0; q<n_q_points; ++q)
        {
          for (unsigned int i=0, i_stokes=0; i_stokes<stokes_dofs_per_cell; /*increment at end of loop*/)
            {
              if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
                {
                  scratch.phi_u[i_stokes] = scratch.finite_element_values[introspection.extractors.velocities].value (i,q);
                  scratch.phi_p[i_stokes] = scratch.finite_element_values[introspection.extractors.pressure].value (i, q);
                  if (rebuild_stokes_matrix)
                    {
                      scratch.grads_phi_u[i_stokes] = scratch.finite_element_values[introspection.extractors.velocities].symmetric_gradient(i,q);
                      scratch.div_phi_u[i_stokes]   = scratch.finite_element_values[introspection.extractors.velocities].divergence (i, q);
                    }
                  ++i_stokes;
                }
              ++i;
            }


          // Viscosity scalar
          const double eta = (rebuild_stokes_matrix
                              ?
                              scratch.material_model_outputs.viscosities[q]
                              :
                              std::numeric_limits<double>::quiet_NaN());

          const SymmetricTensor<4,dim> &stress_strain_director =
            scratch.material_model_outputs.stress_strain_directors[q];
          const bool use_tensor = (stress_strain_director !=  dealii::identity_tensor<dim> ());

          const Tensor<1,dim>
          gravity = this->get_gravity_model().gravity_vector (scratch.finite_element_values.quadrature_point(q));

          const double density = scratch.material_model_outputs.densities[q];

          const double JxW = scratch.finite_element_values.JxW(q);


          for (unsigned int i=0; i<stokes_dofs_per_cell; ++i)
            {
              data.local_rhs(i) += (this->get_adjoint_problem() ? 0 :
                                    (density * gravity * scratch.phi_u[i])
                                    * JxW);

              if (rebuild_stokes_matrix)
                for (unsigned int j=0; j<stokes_dofs_per_cell; ++j)
                  {
                    data.local_matrix(i,j) += ( (use_tensor ?
                                                 eta * 2.0 * (scratch.grads_phi_u[i] * stress_strain_director * scratch.grads_phi_u[j])
                                                 :
                                                 eta * 2.0 * (scratch.grads_phi_u[i] * scratch.grads_phi_u[j]))
                                                // assemble \nabla p as -(p, div v):
                                                - (pressure_scaling *
                                                   scratch.div_phi_u[i] * scratch.phi_p[j])
                                                // assemble the term -div(u) as -(div u, q).
                                                // Note the negative sign to make this
                                                // operator adjoint to the grad p term:
                                                - (pressure_scaling *
                                                   scratch.phi_p[i] * scratch.div_phi_u[j]))
                                              * JxW;
                  }
            }
        }
    }





    template <int dim>
    void
    StokesAssembler<dim>::
    compressible_strain_rate_viscosity_term (const double                                     /*pressure_scaling*/,
                                             const bool                                       rebuild_stokes_matrix,
                                             internal::Assembly::Scratch::StokesSystem<dim>  &scratch,
                                             internal::Assembly::CopyData::StokesSystem<dim> &data) const
    {
      if (!rebuild_stokes_matrix)
        return;

      const Introspection<dim> &introspection = this->introspection();
      const FiniteElement<dim> &fe = this->get_fe();
      const unsigned int stokes_dofs_per_cell = data.local_dof_indices.size();
      const unsigned int n_q_points    = scratch.finite_element_values.n_quadrature_points;

      for (unsigned int q=0; q<n_q_points; ++q)
        {
          for (unsigned int i=0, i_stokes=0; i_stokes<stokes_dofs_per_cell; /*increment at end of loop*/)
            {
              if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
                {
                  scratch.grads_phi_u[i_stokes] = scratch.finite_element_values[introspection.extractors.velocities].symmetric_gradient(i,q);
                  scratch.div_phi_u[i_stokes]   = scratch.finite_element_values[introspection.extractors.velocities].divergence (i, q);

                  ++i_stokes;
                }
              ++i;
            }

          // Viscosity scalar
          const double eta_two_thirds = scratch.material_model_outputs.viscosities[q] * 2.0 / 3.0;

          const SymmetricTensor<4,dim> &stress_strain_director =
            scratch.material_model_outputs.stress_strain_directors[q];
          const bool use_tensor = (stress_strain_director !=  dealii::identity_tensor<dim> ());

          const double JxW = scratch.finite_element_values.JxW(q);

          for (unsigned int i=0; i<stokes_dofs_per_cell; ++i)
            for (unsigned int j=0; j<stokes_dofs_per_cell; ++j)
              {
                data.local_matrix(i,j) += (- (use_tensor ?
                                              eta_two_thirds * (scratch.div_phi_u[i] * trace(stress_strain_director * scratch.grads_phi_u[j]))
                                              :
                                              eta_two_thirds * (scratch.div_phi_u[i] * scratch.div_phi_u[j])
                                             ))
                                          * JxW;
              }
        }
    }



    template <int dim>
    void
    StokesAssembler<dim>::
    reference_density_compressibility_term (const double                                     pressure_scaling,
                                            const bool                                       /*rebuild_stokes_matrix*/,
                                            internal::Assembly::Scratch::StokesSystem<dim>  &scratch,
                                            internal::Assembly::CopyData::StokesSystem<dim> &data,
                                            const Parameters<dim> &parameters) const
    {
      // assemble RHS of:
      //  - div u = 1/rho * drho/dz g/||g||* u
      Assert(parameters.formulation_mass_conservation ==
             Parameters<dim>::Formulation::MassConservation::reference_density_profile,
             ExcInternalError());

      const Introspection<dim> &introspection = this->introspection();
      const FiniteElement<dim> &fe = this->get_fe();
      const unsigned int stokes_dofs_per_cell = data.local_dof_indices.size();
      const unsigned int n_q_points    = scratch.finite_element_values.n_quadrature_points;

      for (unsigned int q=0; q<n_q_points; ++q)
        {
          for (unsigned int i=0, i_stokes=0; i_stokes<stokes_dofs_per_cell; /*increment at end of loop*/)
            {
              if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
                {
                  scratch.phi_p[i_stokes] = scratch.finite_element_values[introspection.extractors.pressure].value (i, q);
                  ++i_stokes;
                }
              ++i;
            }

          const Tensor<1,dim>
          gravity = this->get_gravity_model().gravity_vector (scratch.finite_element_values.quadrature_point(q));
          const double drho_dz_u = scratch.reference_densities_depth_derivative[q]
                                   * (gravity * scratch.velocity_values[q]) / gravity.norm();
          const double one_over_rho = 1.0/scratch.reference_densities[q];
          const double JxW = scratch.finite_element_values.JxW(q);

          for (unsigned int i=0; i<stokes_dofs_per_cell; ++i)
            data.local_rhs(i) += (pressure_scaling *
                                  one_over_rho * drho_dz_u * scratch.phi_p[i])
                                 * JxW;
        }
    }



    template <int dim>
    void
    StokesAssembler<dim>::
    implicit_reference_density_compressibility_term (const double                                     pressure_scaling,
                                                     const bool                                       rebuild_stokes_matrix,
                                                     internal::Assembly::Scratch::StokesSystem<dim>  &scratch,
                                                     internal::Assembly::CopyData::StokesSystem<dim> &data,
                                                     const Parameters<dim> &parameters) const
    {
      // assemble compressibility term of:
      //  - div u - 1/rho * drho/dz g/||g||* u = 0
      Assert(parameters.formulation_mass_conservation ==
             Parameters<dim>::Formulation::MassConservation::implicit_reference_density_profile,
             ExcInternalError());

      if (!rebuild_stokes_matrix)
        return;

      const Introspection<dim> &introspection = this->introspection();
      const FiniteElement<dim> &fe = this->get_fe();
      const unsigned int stokes_dofs_per_cell = data.local_dof_indices.size();
      const unsigned int n_q_points    = scratch.finite_element_values.n_quadrature_points;

      for (unsigned int q=0; q<n_q_points; ++q)
        {
          for (unsigned int i=0, i_stokes=0; i_stokes<stokes_dofs_per_cell; /*increment at end of loop*/)
            {
              if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
                {
                  scratch.phi_u[i_stokes] = scratch.finite_element_values[introspection.extractors.velocities].value (i,q);
                  scratch.phi_p[i_stokes] = scratch.finite_element_values[introspection.extractors.pressure].value (i,q);
                  ++i_stokes;
                }
              ++i;
            }

          const Tensor<1,dim>
          gravity = this->get_gravity_model().gravity_vector (scratch.finite_element_values.quadrature_point(q));
          const Tensor<1,dim> drho_dz = scratch.reference_densities_depth_derivative[q]
                                        * gravity / gravity.norm();
          const double one_over_rho = 1.0/scratch.reference_densities[q];
          const double JxW = scratch.finite_element_values.JxW(q);

          for (unsigned int i=0; i<stokes_dofs_per_cell; ++i)
            for (unsigned int j=0; j<stokes_dofs_per_cell; ++j)
              data.local_matrix(i,j) += -(pressure_scaling *
                                          one_over_rho * drho_dz * scratch.phi_u[j] * scratch.phi_p[i])
                                        * JxW;
        }
    }



    template <int dim>
    void
    StokesAssembler<dim>::
    isothermal_compression_term (const double                                     pressure_scaling,
                                 const bool                                       /*rebuild_stokes_matrix*/,
                                 internal::Assembly::Scratch::StokesSystem<dim>  &scratch,
                                 internal::Assembly::CopyData::StokesSystem<dim> &data,
                                 const Parameters<dim> &parameters) const
    {
      // assemble RHS of:
      //  - div u = 1/rho * drho/dp rho * g * u
      Assert(parameters.formulation_mass_conservation ==
             Parameters<dim>::Formulation::MassConservation::isothermal_compression,
             ExcInternalError());

      const Introspection<dim> &introspection = this->introspection();
      const FiniteElement<dim> &fe = this->get_fe();
      const unsigned int stokes_dofs_per_cell = data.local_dof_indices.size();
      const unsigned int n_q_points    = scratch.finite_element_values.n_quadrature_points;

      for (unsigned int q=0; q<n_q_points; ++q)
        {
          for (unsigned int i=0, i_stokes=0; i_stokes<stokes_dofs_per_cell; /*increment at end of loop*/)
            {
              if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
                {
                  scratch.phi_p[i_stokes] = scratch.finite_element_values[introspection.extractors.pressure].value (i, q);
                  ++i_stokes;
                }
              ++i;
            }

          const Tensor<1,dim>
          gravity = this->get_gravity_model().gravity_vector (scratch.finite_element_values.quadrature_point(q));

          const double compressibility
            = scratch.material_model_outputs.compressibilities[q];

          const double density = scratch.material_model_outputs.densities[q];
          const double JxW = scratch.finite_element_values.JxW(q);

          for (unsigned int i=0; i<stokes_dofs_per_cell; ++i)
            data.local_rhs(i) += (
                                   // add the term that results from the compressibility. compared
                                   // to the manual, this term seems to have the wrong sign, but this
                                   // is because we negate the entire equation to make sure we get
                                   // -div(u) as the adjoint operator of grad(p)
                                   (pressure_scaling *
                                    compressibility * density *
                                    (scratch.velocity_values[q] * gravity) *
                                    scratch.phi_p[i])
                                 )
                                 * JxW;
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
  template class \
  StokesAssembler<dim>;

    ASPECT_INSTANTIATE(INSTANTIATE)
  }
}
