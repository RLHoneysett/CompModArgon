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
    
    :param particle: the data stored on a particle from the Particle3D class, the co-ordinates of
    which can be retrieved by the .position function.
    :param l: length of cube
    :return: coordinates of particle's image inside cube, stored as NumPy array [[x1'. x2'. x3'.]], 0 < xi' < l.
    """
    # Using the modulo operator ensures that this works for negative coordinates as well:
    image_in_cube = np.mod(particle.position, l)
    
    particle.position=image_in_cube
       
def image_closest_to_particle(particle1, particle2, l):   
    """ 
    Computes image of the particle2 closest to particle 1.
    
    :param particle1: the data stored on the first particle, the co-ordinates of which can be
    found using the .position function from the Particle3D class.
    :param particle2: the data stored on the second particle, the co-ordinates of which can be
    found using the .position function from the Particle3D class.
    :param l: length of cube
    :return: coordinates of particle's image closest to the other particle, stored as NumPy array [[x1''. x2''. x3''.]], -l/2 < xi'' < l/2.
    """
    # The most efficient method: add on l/2 before using the modulo operator
    # Correct for this by subtracting l/2 after:
    image = np.mod((particle2.position - particle1.position)+l/2, l) - l/2
    
    return image

"""
Note: if the particle's image in the cube has a coordinate of exactly xi = l/2, the above method will return xi = -l/2 for the 'closest' coordinate. 
      However, there will be at least 2 images which are equally close to the origin (and 8 if image_in_cube = [[l/2. l/2. l/2.]]. 
      Ideally, the function should return the positive position, xi=l/2, since this is within the original cube.
      This is not achieved using the above (very efficient) method. 
      Therefore, I include a second, longer method below, which does this (image_closest_to_origin_long_way()).
      However, the first function (image_closest_to_origin()) is used in the main(), for two reasons:
          i) It is more efficient.
          ii) In real situations, the particle will never be exactly in at xi = l/2, so this situation does not arise.
"""
    
    
