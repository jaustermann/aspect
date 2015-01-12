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


#include <aspect/postprocess/visualization/temperature_perturbation.h>
#include <aspect/simulator_access.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

namespace aspect
{
  namespace Postprocess
  {
    namespace VisualizationPostprocessors
    {
/*      template <int dim>
      Temperature_perturbation<dim>::
      Temperature_perturbation ()
        :
        DataPostprocessorScalar<dim> ("temperature perturbation",
                                      update_values | update_q_points)
      {}

  */    

      template <int dim>
      std::pair<std::string, Vector<float> *>
      Temperature_perturbation<dim>::execute() const
      {
        std::pair<std::string, Vector<float> *>
        return_value ("temperature_perturbation",
                      new Vector<float>(this->get_triangulation().n_active_cells()));

        std::vector<double> avg_temp(100);
        this->get_depth_average_temperature(avg_temp);
        const unsigned int num_slices = avg_temp.size();
        const double max_depth = this->get_geometry_model().maximal_depth();

        // evaluate a single point per cell
        const QMidpoint<dim> quadrature_formula;
        const unsigned int n_q_points = quadrature_formula.size();

        FEValues<dim> fe_values (this->get_mapping(),
                                 this->get_fe(),
                                 quadrature_formula,
                                 update_values   |
                                 update_quadrature_points );

        std::vector<double> temperature_values(n_q_points);

        typename DoFHandler<dim>::active_cell_iterator
        cell = this->get_dof_handler().begin_active(),
        endc = this->get_dof_handler().end();

        unsigned int cell_index = 0;
        for (; cell!=endc; ++cell,++cell_index)
          if (cell->is_locally_owned())
            {
              fe_values.reinit (cell);
              fe_values[this->introspection().extractors.temperature].get_function_values (this->get_solution(),temperature_values);

              const double depth = this->get_geometry_model().depth(fe_values.quadrature_point(0));
              unsigned int idx = static_cast<unsigned int>((depth*num_slices-1)/max_depth);

              (*return_value.second)(cell_index) =  temperature_values[0] - avg_temp[idx];
//std::cout << " temperature at point  " << temperature_values[0] << std::flush;
//std::cout << " average temperature at that depth  " << avg_temp[idx] << std::flush; 
             }
           return return_value;
         }
     }
  }
}


// explicit instantiations
namespace aspect
{
  namespace Postprocess
  {
    namespace VisualizationPostprocessors
    {
      ASPECT_REGISTER_VISUALIZATION_POSTPROCESSOR(Temperature_perturbation,
                                                  "temperature perturbation",
                                                  "A visualization output object that generates output "
                                                  "for the temperature perturbation around the geotherm.")
    }
  }
}
