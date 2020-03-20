# -*- coding: utf-8 -*-
"""
CMod Project B: 3D velocity Verlet time integration of N particles interacting via
the Lennard-Jones (LJ) Potential.
Produces a trajectory file, with which the particles’trajectories can be visualised
using a molecular visualisation program, VMD (Visual Molecular Dynamics).
Writes the following to a file at regular time intervals:
    1) the system’s total, kinetic and potential energies.
       The file will be called energy.dat, and will have the format: <time> <KE> <PE> <Total Energy>
       The energies will also be plotted on a graph.
    2) the particles’ mean square displacement (MSD).
    3) the particles’ radial distribution function (RDF). This will also be visualised as a histogram.
Note that the units used will be reduced units. These are explained below.

The potential energy of the two particles in the LJ potential is:
    U(r1, r2) = 4[1/(r12^12) - 1/(r12^6)]
where:
    * r12 = abs(r2 - r1) (scalar)
    * r1 and r2 are the positions of the two particles (vectors)

The force on particle 1 (vector) is:
    F1(r1, r2) = 48[1/(r12^14)-1/(2*r12^8)](r1-r2)
The force on particle 2 (vector) is:
    F2(r1, r2) = -F1(r1, r2)

The program will require an input file. This will contain the following, where relevant in
reduced units:
    * number of particles
    * number of simulation steps
    * timestep
    * temperature
    * particle density
    * LJ cut-off distance
    * interval ( in terms of timesteps) at which the positions of the particles will be printed to the trajectory file
The format will be:
    <no. particles> <no. steps> <timestep> <temp> <density> <LJ cut-off> <print interval>

The output trajectory file will have the format:

<number of points to plot>
<title line (the timestep number)>
<particle label> <x coordinate> <y-coordinate> <z coordinate>
The final line is repeated N times for N particles.

The ouput files containing the equilibrium properties of the simulated material will have the format:
    *** complete this later***

Reduced units for length, energy, mass, temperature and time will be used.
These (labelled with an asterisk) are given by:
    r* = r/sigma    E* = E/epsilon    m* = 1    T* = epsilon/k_B    t* = sigma * sqrt(m/epsilon)
where sigma is the hard sphere diameter, epsilon is the depth of the potential well,
and k_B is the Boltzmann constant.

Author: Damaris Tan and Rachel Honeysett
Student number: s1645055 and s1711116
Version: 20/02/20
"""

import sys
import math
import numpy as np
import matplotlib.pyplot as pyplot
from Particle3D import Particle3D
from pbc import image_in_cube
from pbc import image_closest_to_particle
from MDUtilities import set_initial_positions
from MDUtilities import set_initial_velocities

def force_lj(particle1, particle2, box_size, lj_cutoff):
    """
    Method to return the force on particle 1 due to the interaction with particle 2 in a LJ potential.
    Force is given by:
    F1(r1, r2) = 48[1/(r12^14)-1/(2*r12^8)](r1-r2)
    :param particle1: Particle3D instance
    :param particle2: Particle3D instance
    :return: force acting on particle 1 as NumPy array, taking into account minimum image convention.
    """
    # compute particle separation, using minimum image convention.
    direction = -image_closest_to_particle(particle1, particle2, box_size)
    # compute the magnitude of this
    r12 = np.linalg.norm(direction)

    # set force to zero
    force = np.zeros(3)
    # if the distance between particle1 and particle2 is less than the LJ cutoff distance,
    # compute the force and add it to the force vector.
    if 0 < r12 < lj_cutoff:
        force = 48 * (1/(r12**14)-1/(2*r12**8)) * direction

    return force


def pot_energy_lj(particle1, particle2, box_size):
    """
    Method to return potential energy of two particles interacting via the LJ potential.
    U(r1, r2) = 4[1/(r12^12) - 1/(r12^6)]
    :param particle1: Particle3D instance
    :param particle2: Particle3D instance
    :param box_size: length of the simulation box (a cube)
    :return: potential energy of the particles as float, taking into account minimum image convention
    """
    # compute the particle separation, using minimum image convention
    r12 = np.linalg.norm( image_closest_to_particle(particle1, particle2, box_size) )

    # Use this in formula for potential
    potential = 4 * (1/(r12**12) - 1/(r12**6))

    return potential


def total_force(particle, particles, num_particles, lj_cutoff, box_size):
    """
    Method to compute the total force on one particle, due to all the other particles.

    :param particle: Particle3D instance - the particle in question
    :param particles: list of all the Particle3D instances in simulation box
    :param num_particles: the number of particles in the simulation box
    :param lj_cutoff: the LJ cut-off distance - the force of particles beyond this distance
                from the particle in question will be considered negligible
    :param box_size: length of the simulation box (a cube)
    :return: the total force on the particle in question
    """

    # start with zero force on the particle
    my_force = np.zeros(3)

    # add on the force between the particle in question, and all the other particles in the box
    for i in range ( num_particles):
        my_force += force_lj(particle, particles[i], box_size, lj_cutoff)

    return my_force

#Mean Square Displacement function
def MSD(particle1, initial_pos, box_size):
    """
    Method to find the average MSD over all particles.
    Uses the equation:
    MSD(t) = (1/N)*sigma(|ri(t) - ri0|^2)

    :param particle1: a Particle3D instance.
    :param initial_pos: the initial position of particle1
    :param box_size: length of the simulation box (a cube)

    :return: the sum of the MSDs for all particles, with each other.
    """
    SD = (np.linalg.norm( np.mod((initial_pos - particle1.position)+ box_size/2, box_size) - box_size/2))**2  #finds SD

    return SD


#collects data for radial distribution function

def RDF_collection(particle1, particle2, box_size):
    """
    Method to calculate the radial distribution at a single time period
    Uses the equation:
    RDF_collection = sigma (delta |rij -r|)

    :param particle1: a Particle3D instance.
    :param particle2: another Particle3D instance.

    :return: the distance between two Particles
    """
    r12 = np.linalg.norm( image_closest_to_particle(particle1, particle2, box_size) )

    return r12

def RDF_normalisation(RDF_collection, rho, num_step, bin_mids):
    """
    Method to normalise the radial distribution function
    Uses the Normalisation factor:
    RDF_normal = (1/4pi*rho*dr*num_step*(r^2))*RDF_collection

    :param RDF_collection: an array of the number of items in each bin, unnormalised
    :param rho: the number density
    :param num_step: the number of timesteps in the simulation
    :param bin_mids: a list of the middle of each bin

    :return: a list of the normalisation factor (to divide by) for
    each bin in the simulation
    """
    norm_factor = []
    for i in range (0, len(bin_mids)):
        norm_factor.append(4*math.pi*rho*0.01*num_step*(bin_mids[i])**2)
    return RDF_collection/np.asarray(norm_factor)

# Begin main code
def main():
    # Read name of input and output files from command line
    # input file is for input parameters specific to the system being described
    # output file is the trajectory file, to be visualised using VMD
    # If the wrong number of arguments are given, tell user the format which should be used in command line
    if len(sys.argv)!=3:
        print("Wrong number of arguments.")
        print("Usage: " + sys.argv[0] + " <input file>" + "<output file>")
        quit()
    else:
        input_file_name = sys.argv[1]
        output_file_name = sys.argv[2]

    # Open input and output files
    input_file = open(input_file_name, "r")
    output_file = open(output_file_name, "w")

    # Read in particle properties, initial conditions from file. File should have format:
    # <no. particles> <no. steps> <timestep> <temp> <density> <LJ cut-off> <print interval>
    line = input_file.readline()
    tokens = line.split(" ")
    # assign variables to the properties/initial conditions
    num_particles = int(tokens[0])
    numstep = int(tokens[1])
    dt = float(tokens[2])          # timestep
    temp = float(tokens[3])
    rho = float(tokens[4])         # density
    lj_cutoff = float(tokens[5])
    print_int = int(tokens[6])     # interval (in terms of timesteps) over which the positions of the particles will be printed to the trajectory file
    input_file.close()             # close the input file

    # create variable for the box size, using the same box size as in MDUtilities.py module
    box_size = (num_particles/rho)**(1./3.)


    # Create a list of Particle3D objects, with arbitrary positions and velocities.
    # Will use the arbitrary position (1, 0, 0) and velocity (1, 0, 0)
    # The list should have length equal to the user-defined number of particles, num_particles.
    particles = []
    arbitrary_position = np.array([1,0,0],float)
    arbitrary_velocity = np.array([1,0,0],float)
    for i in range (0, num_particles):
        name = "particle" + str(i+1)       # create systematic name for each particle
        particles.append(Particle3D(name, arbitrary_position, arbitrary_velocity))

    # Apply the set_initial_positions()and set_initial_velocities() functions to the 'particles' array
    set_initial_positions(rho, particles)
    set_initial_velocities(temp, particles)

    # create list of initial positions for MSD function
    initial_pos_list = []
    for j in range (num_particles):
        initial_pos_list.append(particles[j].position)


    # Print the number of particles and a header line to the trajectory file traj.xyz.
    # Print the initial positions of all the particles in the list 'particles' to the trajectory file.
    output_file.write(str(num_particles) + "\n")
    output_file.write("timestep = 0" +"\n")
    for i in range (0, num_particles):
        output_file.write(str(particles[i]) + "\n")

    # create list holding the total force on each particle
    forces1 = []
    for i in range (0, num_particles):
        force = total_force(particles[i], particles, num_particles, lj_cutoff, box_size)
        forces1.append(force)

    # open a file to which the energy values of the system will be written to. This will be of the format:
    # <time> <KE> <PE> <Total Energy>
    energy_file = open("energy.dat", "w")
    # create lists for energyies
    kinetic_list = []
    potential_list = []
    total_list = []
    time_list = []

    #initialise list of MSD over time
    MSD_list = []

    #initialise list of RDF over time
    RDF_list = []

    # Set time and energies to zero
    time = 0.

    # Start the time integration loop
    for i in range(numstep):

        # Increase time by one timestep
        time += dt

        # Set energies to zero
        kin_energy = 0
        pot_energy = 0

        # For each particle in list 'particles', use the corresponding force in list 'forces1' to update the particle's position
        for j in range (0, num_particles):
            particles[j].leap_pos2nd(dt, forces1[j])

            # make sure all particles remain in the cube
            image_in_cube(particles[j], box_size)

        # update the list of forces, using the new positions
        forces1_new = []
        for j in range (0, num_particles):
            force_new = total_force(particles[j], particles, num_particles, lj_cutoff, box_size)
            forces1_new.append(force_new)

        # Update particle velocity by averaging current and new forces
        for j in range (0, num_particles):
            particles[j].leap_velocity(dt, 0.5*(forces1[j]+forces1_new[j]))

        # Re-define force values
        forces1 = forces1_new

        # add up kinetic and potential energy of the system:
        for j in range (0, num_particles):
            kin_energy += particles[j].kinetic_energy()
            for l in range (0, num_particles):
                if j==l:
                    continue      # Avoid calculating potential between identical particles
                else:
                    individual_potential = 0.5 * pot_energy_lj(particles[j], particles[l], box_size)  # calculate individual potential energy between each particle pair.
                                                                                                      # Then divide by 2 to avoid double counting.
                    pot_energy += individual_potential
        # add KE and PE for total energy
        total_energy = kin_energy + pot_energy

        # Write the KE, PE and total energy to the file energy.dat, as well as the time ellapsed.
        # Everything will be in reduced units.
        energy_file.write("{0:f} {1:12.8f} {2:12.8f} {3:12.8f}\n".format(time, kin_energy, pot_energy, total_energy))

        #append energies to lists
        kinetic_list.append(kin_energy)
        potential_list.append(pot_energy)
        total_list.append(total_energy)
        time_list.append(time)

        #time integration element for MSD

        totalMSD = 0                  #initialise the total MSD
        for i in range (0, num_particles):
            for j in range (0, num_particles):
                if i != j:
                    continue          #ensures the MSD between a particle and itself is not calculated
                else:
                    totalMSD += MSD(particles[i], initial_pos_list[j], box_size)/num_particles  #adds the Squared Difference to sum and divides by N to find mean

        MSD_list.append(totalMSD)


        #time integration element for RDF
        """
        so, the plan here is to collect the radial distance of each of the particles from each other, and then
        round them to the nearest 0.01 (essentially making bins of 0.01 reduced units in length).  Then I made a list of all of the bin
        boundaries (0.01, 0.02, 0.03... etc.) and counted how many items in the rounded list that there were that matched the
        values of the bins specified in the "bins" list.  The number of items in each list was then recorded in the list count_for_bins
        and this could be divided by the normalisation factor to give the y values for historgram.
        """
        for k in range (0, num_particles):
            if k == 1:
                continue          #ensures distance not calculated for a distance and itself
            else:
                RDF_list.append(RDF_collection(particles[k], particles[1], box_size))
        RDF_rounded = np.around(RDF_list, 2)
        count_for_bins = []
        bins = []       #makes list of all bins used for RDF
        for j in range (0, int(box_size*100)):
            bins.append(0.01*j)
        for i in range (0, len(bins)):
            count_for_bins.append(np.count_nonzero(RDF_rounded == bins[i]))
        count_array = np.asarray(count_for_bins)



        # write new positions to trajectory file, for every nth timestep (named print_int) using the modulo function
        # At each nth timestep, write the energies to the file energy.dat, in format given above.
        if (i+1)%print_int == 0:                                 # print to trajectory file if timestep is a multiple of print_int
            output_file.write(str(num_particles) + "\n")
            output_file.write("timestep = " + str(i+1) + "\n")
            for j in range (0, num_particles):
                output_file.write(str(particles[j]) + "\n")

        else:
            continue  # don't print to traj file if timestep is a multiple of print_int
    #testing making a histogram for the RDF_list
    norm_bins = RDF_normalisation(count_array, rho, numstep, bins)
    #print(len(count_for_bins))


    pyplot.plot(bins, norm_bins)
    pyplot.title("Radial Distribution Function")
    pyplot.ylabel("No. Particles")
    pyplot.xlabel("r*")
    pyplot.show()
    # Plot system energy to screen
    pyplot.title('Energy vs time')
    pyplot.xlabel('Time in reduced units (t*)')
    pyplot.ylabel('Energy in reduced units (E*)')
    line1, = pyplot.plot(time_list, kinetic_list, label = 'Kinetic energy', color = 'red')
    line2, = pyplot.plot(time_list, potential_list, label = 'Potential energy', color = 'blue')
    line3, = pyplot.plot(time_list, total_list, label = 'Total energy', color = 'black')
    #create legends to identify lines
    first_legend=pyplot.legend(handles=[line1], loc='lower left')
    second_legend=pyplot.legend(handles=[line2], loc='lower center')
    third_legend=pyplot.legend(handles=[line3], loc='lower right')
    ax = pyplot.gca().add_artist(first_legend)
    ax = pyplot.gca().add_artist(second_legend)
    ax = pyplot.gca().add_artist(third_legend)
    pyplot.show()

    # Post-simulation:
    # Close output file and energy.dat file
    energy_file.close()
    output_file.close()

    # Plot system MSD over time
    pyplot.title('MSD vs time')
    pyplot.xlabel('Time in reduced units (t*)')
    pyplot.ylabel('MSD in reduced units (r*^2)')
    line, = pyplot.plot(time_list, MSD_list)
    pyplot.show()

# Execute main method, but only when directly invoked
if __name__ == "__main__":
    main()
