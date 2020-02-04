"""
CompMod Checkpoint 1: Periodic Boundary conditions, pbc.py
This program:
    - prompts for the coordinates of a particle, and the length, l, of a cube in the positive 
      octant of the xyz axes, with one vertex at (0,0,0). 
    - returns the coordinates of the image of the particle within the cube.
    - returns the coordinates of the image of the particle closest to the other particle.

Author: Damaris Tan and Ray Honeysett
Student number: s1645055 and s1711116
Version: 04/02/20
"""
import numpy as np

def image_in_cube(particle, l):
    """
    Computes image of the particle within the cube of given length l.
    
    :param particle: a Particle3D isntance, the co-ordinates of which can be retrieved by the .position function.
    :param l: length of cube
    :updates the position of the Particle3D object so that it is within the cube
    """
    # Using the modulo operator ensures that this works for negative coordinates as well:
    image_in_cube = np.mod(particle.position, l)
    
    particle.position=image_in_cube
       
def image_closest_to_particle(particle1, particle2, l):   
    """ 
    Computes image of the particle2 closest to particle1.
    
    :param particle1: a Particle3D instance, the co-ordinates of which can be found using the .position function from the Particle3D class.
    :param particle2: another Particle3D instance, the co-ordinates of which can be found using the .position function from the Particle3D class.
    :param l: length of cube
    :return: coordinates of particle's image closest to the other particle, stored as NumPy array [[x1''. x2''. x3''.]], -l/2 < xi'' < l/2.
    """
    # The most efficient method: add on l/2 before using the modulo operator
    # Correct for this by subtracting l/2 after:
    image = np.mod((particle2.position - particle1.position)+l/2, l) - l/2
    
    return image
