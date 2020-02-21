"""
 CMod Ex2: Particle3D, a class to describe 3D particles.
 
 Class to describe 3D particles.

    Properties:
    - name(string) - label of particle
    - position(NumPy array) - position as 3D coord (x1, x2, x3) 
    - velocity(NumPy array) - velocity vector (v1, v2, v3) 
    - mass(float) - particle mass. In reduced units this is always 1 and is therefore
                    not required as an argument in the __init__() method.

    Methods:
    - formatted output
    - kinetic energy
    - first-order velocity update
    - second order position update
    - return relative particle separation of two particles (static method)
 
 Author: Damaris Tan, Rachel Honeysett
 Student number: s1645055
 Version: 4/2/20
"""
import numpy as np

class Particle3D(object):

    def __init__(self, name, position, velocity):
        """
        Initialise a Particle3D instance

        :param name: name of the particle as a string
        :param position: position vector r=(x1, x2, x3), stored  as a NumPy array 
        :param v: velocity vector v=(v1, v2, v3), stored as a NumPy array 
        :param mass: mass as float
        """
        self.name = name
        self.position = position
        self.velocity = velocity
        self.mass = 1

    def __str__(self):
        """
        Define output format.
        For particle called 'tester' at position = (2.0, 0.5, 1.0) this will print as
        "tester 2.0 0.5 1.0"
        """
        return self.name +" " + str( self.position[0])+ " " + str(self.position[1]) + " " + str(self.position[2])

    
    def kinetic_energy(self):
        """
        Return kinetic energy using:
        (1/2)*mass*(v.v)
        """
        return 0.5*self.mass*float(np.inner(self.velocity, self.velocity))
        

    # Time integration methods
    def leap_velocity(self, dt, f):
        """
        First-order velocity update,
        v(t+dt) = v(t) + dt*f(t)/m ;  f(t) = (f1, f2, f3)     

        :param dt: timestep as float
        :param f: force vector f(t) = (f1, f2, f3), stored as NumPy array
        """
        self.velocity = self.velocity + dt*f/self.mass


    def leap_pos2nd(self, dt, f):
        """
        Second-order position update,
        r(t+dt) = r(t) + dt*v(t) + dt^2*f(t)/(2m),  f(t) = (f1, f2, f3)     

        :param dt: timestep as float
        :param f: force vector f(t) = (f1, f2, f3), stored as NumPy array
        """        
        self.position = self.position + dt*self.velocity + 0.5*dt**2*f/self.mass
          
        
    @staticmethod   
    def particle_separation(particle1, particle2):
        """
        Returns the distance between particle 1 and particle 2 with position 
        vectors position1 and position2 respectively.

        :param particle1: Particle3D instance of first particle
        :param particle2: Particle3D instance of second particle
        """
        # Extract position of particles 1 and 2
        position1 = particle1.position
        position2 = particle2.position
        
        # Subtract one position from the other and calculate the modulus of this
        separation = np.linalg.norm(position1 - position2)
        
        return separation


