/*
  Copyright (C) 2011, 2012 by the authors of the ASPECT code.

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


#ifndef __aspect__postprocess_visualization_density_h
#define __aspect__postprocess_visualization_density_h

#include <aspect/postprocess/visualization.h>
#include <aspect/simulator_access.h>

#include <deal.II/numerics/data_out.h>


namespace aspect
{
  namespace Postprocess
  {
    namespace VisualizationPostprocessors
    {

      /**
       * A class derived from DataPostprocessor that takes an output vector
       * and computes a variable that represents the density at every point.
       *
       * The member functions are all implementations of those declared in the
       * base class. See there for their meaning.
       */
      template <int dim>
      class Temperature_perturbation
       :
       public CellDataVectorCreator<dim>,
       public SimulatorAccess<dim>
      {
        public:

          virtual
          std::pair<std::string, Vector<float> *>
          execute() const;

      };
    }
  }
}

#endif
