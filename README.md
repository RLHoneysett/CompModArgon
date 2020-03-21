# CompModArgon
Simulation of Argon Using Python
To run this simulation, open a terminal in the file where all of the CompModArgon are
stored, and type the "python3 particleManyBody.py" command, followed by the input file
for the relevant state of matter, and then the name that is to be given to the output 
trajectory file produced by the simulation.  The trajectory file produced can be 
visualised using VMD.  Running the code will produce three graphs, the first showing 
the Radial Distribution Function of the particles, the second showing the Mean Squared 
Distribution, and the third showing a graph of Kinetic, potential and total energy of the 
summed particles in the simulation.

The simulation simulates a number of argon atoms (default: 32 atoms) moving under the 
interation of the Lennard-Jones force.  The movement of the particles in the simulation
uses periodic boundary conditions and follows the minimum image convention.  Time 
iterations are done using the velocity Verlet method.  

Input parameters have been set so that temperature and number density values are
appropriate to each state of matter.  In order to change the number of particles in 
the simulation, change the first number in the input file.  To change the number of time
steps, change the second number in the file.
