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


#ifndef _aspect_postprocess_surface_topography_h
#define _aspect_postprocess_surface_topography_h

#include <aspect/postprocess/interface.h>
#include <aspect/simulator_access.h>


namespace aspect
{
  namespace Postprocess
  {

    /**
     * A postprocessor that computes dynamic topography at the top and bottom of the domain.
     *
     * @ingroup Postprocessing
     */
    template <int dim>
    class DynamicTopography : public Interface<dim>, public ::aspect::SimulatorAccess<dim>
    {
      public:
        /**
         * Evaluate the solution for the dynamic topography.
         */
        virtual
        std::pair<std::string,std::string>
        execute (TableHandler &statistics);

        /**
         * Return the topography vector as calculated by CBF formulation
         */
        const LinearAlgebra::BlockVector &
        topography_vector() const;

        /**
         * Register the other postprocessor that we need: BoundaryPressures
         */
        virtual
        std::list<std::string>
        required_other_postprocessors() const;

        /**
         * Parse the parameters for the postprocessor.
         */
        void
        parse_parameters (ParameterHandler &prm);

        /**
         * Declare the parameters for the postprocessor.
         */
        static
        void
        declare_parameters (ParameterHandler &prm);

      private:
        /**
         * Output the dynamic topography solution to
         * a file.
         */
        void output_to_file(bool upper, std::vector<std::pair<Point<dim>, double> > &values);

        /**
         * A vector which stores the surface stress values calculated
         * by the postprocessor.
         */
        LinearAlgebra::BlockVector topo_vector;

        /**
         * A parameter allows users to set the density value
         * above the top surface.
         */
        double density_above;

        /**
         * A parameter allows users to set the density value
         * below the bottom surface.
         */
        double density_below;

        /**
         * Whether to output the surface topography.
         */
        bool output_surface;

        /**
         * Whether to output the bottom topography.
         */
        bool output_bottom;
    };
  }
}


#endif
