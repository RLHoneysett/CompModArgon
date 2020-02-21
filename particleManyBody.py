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
from MDutilities import set_initial_velocities

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
    output_file.write("timestep = 1" +"\n")
    for i in range (0, num_particles):
        output_file.write(str(particles[i]) + "\n")
    
    
    
   # *** Everything below this is old code, still needs to be changed ***
    
    # Set up simulation parameters
    dt = 0.01
    numstep = int(2000) # must be an integer as this is used as the range in a for loop
    time = 0.0
    
    
    #initial relative separation:
    r12 = Particle3D.particle_separation(p1, p2)
    
    # The units of energy, length and mass are in eV, angstroms and a.m.u respectively. 
    # The units of time are [length]*([mass]/[energy])^(1/2)
    # So to convert to femto-seconds, need to multiply all time values by (10**-10 m)*[(1 a.m.u. in kg)/(1 eV in joules)]^(1/2) / 10^-15
    # CODATA conversions are used
    time_unit = 10**-10 * (1.66053906660*10**-27)**0.5 * (1.602176634*10**-19)**-0.5 * 10**15

    # Write out initial conditions
    # Write out energy and separation to 8 decimal places. Add space before energy/separation for clarity
    energy = p1.kinetic_energy()+ p2.kinetic_energy() + pot_energy_morse(p1, p2, r_e, D_e, alpha)   
    energy_file.write("{0:f} {1:12.8f}\n".format(time*time_unit, energy))
    separation_file.write("{0:f} {1:12.8f}\n".format(time*time_unit, r12))

    # Get initial force
    force1 = force_morse(p1, p2, r_e, D_e, alpha)

    # Initialise data lists for plotting later
    time_list = [time*time_unit]
    r12_list = [r12]  #relative separation list
    energy_list = [energy]

    # Start the time integration loop
    for i in range(numstep):
        # Update particle position and separation
        p1.leap_pos2nd(dt, force1)
        p2.leap_pos2nd(dt, -force1)
        #update relative position
        r12 = Particle3D.particle_separation(p1, p2)
        
        # Update force
        force1_new = force_morse(p1, p2, r_e, D_e, alpha)
        # Update particle velocity by averaging
        # current and new forces
        p1.leap_velocity(dt, 0.5*(force1+force1_new))
        p2.leap_velocity(dt, 0.5*(-force1-force1_new))
        
        
        # Re-define force value
        force1 = force1_new

        # Increase time
        time += dt
        
        # Output particle information
        energy = p1.kinetic_energy() + p2.kinetic_energy() + pot_energy_morse(p1, p2, r_e, D_e, alpha)
        energy_file.write("{0:f} {1:12.8f}\n".format(time*time_unit, energy))
        separation_file.write("{0:f} {1:12.8f}\n".format(time*time_unit, r12))

        # Append information to data lists
        time_list.append(time*time_unit)
        r12_list.append(r12)
        energy_list.append(energy)


    # Post-simulation:
    # Close output file
    energy_file.close()
    separation_file.close()

    # Plot particle trajectory to screen
    pyplot.title('Velocity Verlet: Relative Position vs time')
    pyplot.xlabel('Time (fs)')     
    pyplot.ylabel('Relative position (angstroms)')
    pyplot.plot(time_list, r12_list)
    pyplot.show()

    # Plot particle energy to screen
    pyplot.title('Velocity Verlet: total energy vs time')
    pyplot.xlabel('Time (fs)')     
    pyplot.ylabel('Energy (eV)')
    pyplot.plot(time_list, energy_list)
    pyplot.show()


# Execute main method, but only when directly invoked
if __name__ == "__main__":
    main()

