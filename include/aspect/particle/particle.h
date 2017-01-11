/*
 Copyright (C) 2015 by the authors of the ASPECT code.

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

#ifndef _aspect_particle_particle_h
#define _aspect_particle_particle_h

#include <aspect/global.h>
#include <aspect/particle/property_pool.h>

#include <deal.II/base/point.h>
#include <deal.II/base/types.h>
#include <deal.II/base/array_view.h>

#include <boost/serialization/vector.hpp>

namespace aspect
{
  namespace Particle
  {
    using namespace dealii;

    /**
     * A namespace for all type definitions related to particles.
     */
    namespace types
    {
      using namespace dealii::types;

      /**
       * Typedef of cell level/index pair. TODO: replace this by the
       * active_cell_index from deal.II 8.3 onwards.
       */
      typedef std::pair<int, int> LevelInd;

      /* Type definitions */

#ifdef DEAL_II_WITH_64BIT_INDICES
      /**
       * The type used for indices of tracers. While in
       * sequential computations the 4 billion indices of 32-bit unsigned integers
       * is plenty, parallel computations using hundreds of processes can overflow
       * this number and we need a bigger index space. We here utilize the same
       * build variable that controls the dof indices of deal.II because the number
       * of degrees of freedom and the number of tracers are typically on the same
       * order of magnitude.
       *
       * The data type always indicates an unsigned integer type.
       */
      typedef unsigned long long int particle_index;

      /**
       * An identifier that denotes the MPI type associated with
       * types::global_dof_index.
       */
#  define ASPECT_TRACER_INDEX_MPI_TYPE MPI_UNSIGNED_LONG_LONG
#else
      /**
       * The type used for indices of tracers. While in
       * sequential computations the 4 billion indices of 32-bit unsigned integers
       * is plenty, parallel computations using hundreds of processes can overflow
       * this number and we need a bigger index space. We here utilize the same
       * build variable that controls the dof indices of deal.II because the number
       * of degrees of freedom and the number of tracers are typically on the same
       * order of magnitude.
       *
       * The data type always indicates an unsigned integer type.
       */
      typedef unsigned int particle_index;

      /**
       * An identifier that denotes the MPI type associated with
       * types::global_dof_index.
       */
#  define ASPECT_TRACER_INDEX_MPI_TYPE MPI_UNSIGNED
#endif
    }

    /**
     * Base class of particles - represents a particle with position,
     * an ID number and a variable number of properties. This class
     * can be extended to include data related to a particle by the property
     * manager.
     *
     * @ingroup Particle
     *
     */
    template <int dim>
    class Particle
    {
      public:
        /**
         * Empty constructor for Particle, creates a particle at the
         * origin.
         */
        Particle ();

        /**
         * Constructor for Particle, creates a particle with the specified
         * ID at the specified location. Note that Aspect
         * does not check for duplicate particle IDs so the generator must
         * make sure the IDs are unique over all processes.
         *
         * @param[in] new_location Initial location of particle.
         * @param[in] new_reference_location Initial location of the particle
         * in the coordinate system of the reference cell.
         * @param[in] new_id Globally unique ID number of particle.
         */
        Particle (const Point<dim> &new_location,
                  const Point<dim> &new_reference_location,
                  const types::particle_index new_id);

        /**
         * Copy-Constructor for Particle, creates a particle with exactly the
         * state of the input argument. Note that since each particle has a
         * handle for a certain piece of the property memory, and is responsible
         * for registering and freeing this memory in the property pool this
         * constructor registers a new chunk, and copies the properties.
         */
        Particle (const Particle<dim> &particle);

        /**
         * Constructor for Particle, creates a particle from a data vector.
         * This constructor is usually called after sending a particle to a
         * different process.
         *
         * @param[in,out] begin_data A pointer to a memory location from which
         * to read the information that completely describes a particle. This
         * class then de-serializes its data from this memory location and
         * advance the pointer accordingly.
         *
         * @param[in,out] new_property_pool A property pool that is used to
         * allocate the property data used by this particle.
         */
        Particle (const void *&begin_data,
                  PropertyPool &new_property_pool);

#ifdef DEAL_II_WITH_CXX11
        /**
         * Move constructor for Particle, creates a particle from an existing
         * one by stealing its state.
         */
        Particle (Particle<dim> &&particle);

        /**
         * Copy assignment operator.
         */
        Particle<dim> &operator=(const Particle<dim> &particle);

        /**
         * Move assignment operator.
         */
        Particle<dim> &operator=(Particle<dim> &&particle);
#endif

        /**
         * Destructor. Releases the property handle if it is valid, and
         * therefore frees that memory space for other particles. (Note:
         * the memory is managed by the property_pool, and the memory is not
         * deallocated by this function, it is kept in reserve for other
         * particles).
         */
        ~Particle ();

        /**
         * Write particle data into a data array. The array is expected
         * to be large enough to take the data, and the void pointer should
         * point to the first element in which the data should be written. This
         * function is meant for serializing all particle properties and
         * afterwards de-serializing the properties by calling the appropriate
         * constructor Particle(void *&data, const unsigned int data_size);
         *
         * @param [in,out] data The memory location to write particle data
         * into. This pointer points to the begin of the memory, in which the
         * data will be written and it will be advanced by the serialized size
         * of this particle.
         */
        void
        write_data(void *&data) const;

        /**
         * Set the location of this particle. Note that this does not check
         * whether this is a valid location in the simulation domain.
         *
         * @param [in] new_loc The new location for this particle.
         */
        void
        set_location (const Point<dim> &new_loc);

        /**
         * Get the location of this particle.
         *
         * @return The location of this particle.
         */
        const Point<dim> &
        get_location () const;

        /**
         * Set the reference location of this particle. Note that this does not
         * check whether this is a valid location in the simulation domain.
         *
         * @param [in] new_loc The new reference location for this particle.
         */
        void
        set_reference_location (const Point<dim> &new_loc);

        /**
         * Get the reference location of this particle in its current cell.
         *
         * @return The reference location of this particle.
         */
        const Point<dim> &
        get_reference_location () const;

        /**
         * Get the ID number of this particle.
         *
         * @return The id of this particle.
         */
        types::particle_index
        get_id () const;

        /**
         * Tell the particle where to store its properties (even if it does not
         * own properties). Usually this is only done once per particle, but
         * since the particle generator does not know about the properties
         * we want to do it not at construction time. Another use for this
         * function is after particle transfer to a new process.
         */
        void
        set_property_pool(PropertyPool &property_pool);

        /**
         * Set the properties of this particle.
         *
         * @param [in] new_properties A vector containing the
         * new properties for this particle.
         */
        void
        set_properties (const std::vector<double> &new_properties);

        /**
         * Get write-access to properties of this particle.
         *
         * @return An ArrayView of the properties of this particle.
         */
        const ArrayView<double>
        get_properties ();

        /**
         * Get read-access to properties of this particle.
         *
         * @return An ArrayView of the properties of this particle.
         */
        const ArrayView<const double>
        get_properties () const;

        /**
         * Serialize the contents of this class.
         */
        template <class Archive>
        void serialize (Archive &ar, const unsigned int version);

      private:
        /**
         * Current particle location
         */
        Point<dim>             location;

        /**
         * Current particle location in the reference cell.
         * Storing this reduces the number of times we need to compute this
         * location, which takes a significant amount of computing time.
         */
        Point<dim>             reference_location;

        /**
         * Globally unique ID of particle
         */
        types::particle_index  id;

        /**
         * A pointer to the property pool. Necessary to translate from the
         * handle to the actual memory locations.
         */
        PropertyPool *property_pool;

        /**
         * A handle to all tracer properties
         */
        PropertyPool::Handle properties;
    };

    /* -------------------------- inline and template functions ---------------------- */

    template <int dim>
    template <class Archive>
    void Particle<dim>::serialize (Archive &ar, const unsigned int)
    {
      ar &location
      & id
      & properties
      ;
    }
  }
}

#endif

