# -*- coding: utf-8 -*-
"""
CMod Project B: 3D velocity Verlet time integration of N particles interacting via 
the Lennard-Jones (LJ) Potential.

Produces a trajectory file, with which the particles’trajectories can be visualised 
using a molecular visualisation program, VMD (Visual Molecular Dynamics).
Writes the following to a file at regular time intervals:
    1) the system’s total, kinetic and potential energies. 
    2) the particles’ mean square displacement (MSD).
    3) the particles’ radial distribution function (RDF). This will also be visualised as a histogram.
    
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
The format will be:
    <no. particles> <no. steps> <timestep> <temp> <density> <LJ cut-off> 
    
The output trajectory file will have the format:
    
<number of points to plot>
<title line (in the example below it will give the timestep number)>
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
import numpy as np
import matplotlib.pyplot as pyplot
from Particle3D import Particle3D
from pbc import image_in_cube
from pbc import image_closest_to_particle
from MDUtilities import set_initial_positions
from MDUtilities import set_initial_velocities

def force_lj(particle1, particle2):
    """
    Method to return the force on particle 1 due to the interaction with particle 2 in a LJ potential.
    Force is given by:
    F1(r1, r2) = 48[1/(r12^14)-1/(2*r12^8)](r1-r2)

    :param particle1: Particle3D instance
    :param particle2: Particle3D instance
    :return: force acting on particle 1 as NumPy array
    """
    # compute particle separation
    r12 = Particle3D.particle_separation(particle1, particle2)
    # compute normalised direction vector pointing along axis of molecule - this is the direction in which the force acts
    direction = (1/r12)*(particle2.position - particle1.position)
    
    # compute the magnitude of the force and multiply by the normalised direction vector
    force = 48 * (1/(r12**14)-1/(2*r12**8)) * direction
    
    return force


def pot_energy_lj(particle1, particle2):
    """
    Method to return potential energy of two particles interacting via the LJ potential.
    U(r1, r2) = 4[1/(r12^12) - 1/(r12^6)]

    :param particle1: Particle3D instance
    :param particle2: Particle3D instance
    :return: potential energy of the particles as float
    """
    # compute the particle separation
    r12 = Particle3D.particle_separation(particle1, particle2)
    # Use this in formula for potential
    potential = 4 * (1/(r12**12) - 1/(r12**6))
    
    return potential
    
    
def total_force(particle, particles, num_particles, lj_cutoff, rho):
    """
    Method to compute the total force on one particle, due to all the other particles.
    Remembering to use the image closest to the particle in question! 
    ***Not sure if this will work!***
    
    :param particle: Particle3D instance - the particle in question
    :param particles: list of all the Particle3D instances in simulation box
    :num_particles: the number of particles in the simulation box
    :lj_cutoff: the LJ cut-off distance - the force of particles beyond this distance 
                from the particle in question will be considered negligible. 
    :return: the total force on the particle in question. 
    """ 
    total_force = np.array([0,0,0])
    
    for i in range (0, num_particles):
        # Find image closest to the particle in question
        box_size = (num_particles/rho)**(1./3.)
        closest_image = image_closest_to_particle(particle, particles[i], box_size)
        
        # Calculate distance between particle in question and the next particle in the simulation box
        r12 = Particle3D.particle_separation(particle, closest_image)
        
        # If the distance between the particles is zero (ie you have two of the same particle)
        # move on to the next particle
        if r12 == 0:
            continue
            
        # If distance between the particles is greater than the LJ-cutoff distance
        # move on to the next particle.
        elif r12 > lj_cutoff:
            continue
            
        # If r12 is less than the LJ-cutoff distance, add the force due to this particle
        # to the total force on the particle in question.
        else:
            total_force = total_force + force_lj(particle, closest_image)
    
    return total_force

# Begin main code
def main():
    # Read name of input and output files from command line 
    # input file is for input parameters specific to the system being described
    # output file is the trajectory file, to be visualised using VMD
    # If the wrong number of arguments are given, tell user the format which should be used in command line
    if len(sys.argv)!=3:
        print("Wrong number of arguments.")
        print("Usage: " + sys.argv[0] + " <output file 1>" + "<output file 2>")
        quit()
    else:
        input_file_name = sys.argv[1]
        output_file_name = sys.argv[2]

    # Open input and output files
    input_file = open(input_file_name, "r")
    output_file = open(output_file_name, "w")

    # Read in particle properties, initial conditions from file. File should have format:
    # <no. particles> <no. steps> <timestep> <temp> <density> <LJ cut-off> 
    line = input_file.readline()
    tokens = line.split(" ")
    # assign variables to the properties/initial conditions
    num_particles = int(tokens[0])
    numstep = int(tokens[1])
    dt = float(tokens[2])          # timestep
    temp = float(tokens[3])
    rho = float(tokens[4])         # density
    lj_cutoff = float(tokens[5])
    input_file.close()             # close the input file
    
    # create variable for the box size, using the same box size as in MDUtilities.py module
    box_size = (num_particles/rho)**(1./3.)
    
    
    # Create a list of Particle3D objects, with arbitrary positions and velocities. 
    # Will use the arbitrary position (1, 0, 0) and velocity (1, 0, 0)
    # The list should have length equal to the user-defined number of particles, num_particles.
    particles = []
    arbitrary_position = np.array([1,0,0])
    arbitrary_velocity = np.array([1,0,0])
    for i in range (0, num_particles):
        name = "particle" + str(i+1)       # create systematic name for each particle
        particles.append(Particle3D(name, arbitrary_position, arbitrary_velocity))
    
    # Apply the set_initial_positions()and set_initial_velocities() functions to the 'particles' array
    set_initial_positions(rho, particles)
    set_initial_velocities(temp, particles)
    
    # Print the number of particles and a header line to the trajectory file traj.xyz. 
    # Print the initial positions of all the particles in the list 'particles' to the trajectory file.
    output_file.write(str(num_particles) + "\n")
    output_file.write("timestep = 0" +"\n")
    for i in range (0, num_particles):
        output_file.write(str(particles[i]) + "\n")
    
    
    # create list holding the total force on each particle 
    forces1 = []
    for i in range (0, num_particles):
        force = total_force(particles[i], particles, num_particles, lj_cutoff, rho)
        forces1.append(force)    
    
    # Start the time integration loop
    for i in range(numstep):
        # For each particle in list 'particles', use the corresponding froce in list 'forces1' to update the particle's position
        for j in range (0, num_particles):
            particles[j].leap_pos2nd(dt, forces1[j])
            
            # make sure all particles remain in the cube
            image_in_cube(particles[j], box_size)
        
        # update the list of forces, using the new positions
        forces1_new = []
        for j in range (0, num_particles):
            force_new = total_force(particles[j], particles, num_particles, lj_cutoff, rho)
            forces1_new.append(force_new)
            
        # Update particle velocity by averaging current and new forces
        for j in range (0, num_particles):
            particles[j].leap_velocity(dt, 0.5*(forces1[j]+forces1_new[j]))
            
        # Re-define force values
        forces1 = forces1_new
        
        # write new positions to trajectory file - once this is working, reduce the number of times we print this to the file - maybe for every kth step
        output_file.write(str(num_particles) + "\n")
        output_file.write("timestep = " + str(i+1) + "\n")
        for j in range (0, num_particles):
            output_file.write(str(particles[j]) + "\n")
    
    # Post-simulation:
    # Close output file
    output_file.close()

# Execute main method, but only when directly invoked
if __name__ == "__main__":
    main()

