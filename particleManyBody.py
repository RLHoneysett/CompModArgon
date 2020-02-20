"""
CMod Ex2: 3D velocity Verlet time integration of a two particles interacting via 
the Morse Potential.

Produces plots of the relative positions of the particles and the total energy, 
both as function of time. Also saves these to two separate files: 
    1) time and total energy
    2) time and particle separation
    
The potential energy of the two particles in the potential is:
    U(r1, r2) = D_e {(1 - exp[-alpha(r12 - r_e)])^2 - 1}
where:
    * r1 and r2 are the positions of the two particles (vectors)
    * r12 = abs(r2 - r1) (scalar)
    * r_e is the equilibrium bond distance (scalar)
    * D_e is the well depth (scalar)
    * alpha is a parameter related to the force constant and D_e (scalar). 
      Describes the curvature of the potential minimum.
      
The force on particle 1 (vector) is:
    F1(r1, r2) = 2 * alpha * D_e {1 - exp [-apha(r12 - r_e)]} * exp[-alpha(r12 - r_e)] * (r1-r2)/r12
The force on particle 2 (vector) is:
    F2(r1, r2) = -F1(r1, r2)
    
The initial conditions for the particles are read from a user-defined input file of the format: 
    <label> <x pos> <y pos> <z pos> <x vel> <y vel> <z vel> <mass> 
    
The parameters of the Morse Potential specific to a particlular element are read from a user-defined
file of the format:    
    <D_e> <r_e> <alpha>
    
The units used will be eV, angstroms and a.m.u for energy, length and mass respectively.
The methods for calculating the force and potential take the the parameters 
r_e, D_e and alpha as arguments.

Author: Damaris Tan
Student number: s1645055
Version: 16/11/19
"""

import sys
import numpy as np
import matplotlib.pyplot as pyplot
from Particle3D import Particle3D

def force_lj(particle1, particle2):
    """
    Method to return the force on particle 1 due to the interaction with particle 2 in a LJ potential.
    Force is given by:
    F1(r1, r2) = 48((1/r12^14)-(1/(2*r12^8)))(r1-r2)

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
    U(r1, r2) = 4 * (1/(r12^12) - 1/(r^6))

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
    # Read name of output files from command line 
    # output files are for: 1) time and relative separation  2) time and total energy 
    # If the wrong number of arguments are given, tell user the format which should be used in command line
    if len(sys.argv)!=3:
        print("Wrong number of arguments.")
        print("Usage: " + sys.argv[0] + " <output file 1>" + "<output file 2>")
        quit()
    else:
        energy_file_name = sys.argv[1]
        separation_file_name = sys.argv[2]

    # Open output file
    energy_file = open(energy_file_name, "w")
    separation_file = open(separation_file_name, "w")

    # Set up simulation parameters
    dt = 0.01
    numstep = int(2000) # must be an integer as this is used as the range in a for loop
    time = 0.0
    
    # Read in particle properties, initial conditions from file. File should have format:
    # <label> <x pos> <y pos> <z pos> <x vel> <y vel> <z vel> <mass> 
    file_handle = open("oxygenInitialConditions.dat", "r")
    # file_handle.readline()
    p1 = Particle3D.from_file(file_handle)   
    p2 = Particle3D.from_file(file_handle)
    
    # Read in parameters for oxygen/nitrogen. File should have format:
    # <D_e> <r_e> <alpha>
    filein = open("oxygenParameters.dat", "r")  
    line = filein.readline()
    tokens = line.split(",")
    D_e = float(tokens[0])
    r_e = float(tokens[1])
    alpha = float(tokens[2])
    filein.close()
    
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
