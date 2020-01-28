"""
CompMod Checkpoint 1: Periodic Boundary conditions, pbc.py
This program:
    - prompts for the coordinates of a particle, and the length, l, of a cube in the positive 
      octant of the xyz axes, with one vertex at (0,0,0). 
    - returns the coordinates of the image of the particle within the cube.
    - returns the coordinates of the image of the particle closest to the origin.

Author: Damaris Tan
Student number: s1645055
Version: 22/10/19
"""
import numpy as np

def image_in_cube(x, l):
    """
    Computes image of the particle within the cube of given length l.
    
    :param x: coordinates of particle, stored as NumPy array [[x1. x2. x3.]]
    :param l: length of cube
    :return: coordinates of particle's image inside cube, stored as NumPy array [[x1'. x2'. x3'.]], 0 < xi' < l.
    """
    # Using the modulo operator ensures that this works for negative coordinates as well:
    image_in_cube = np.mod(x, l)
    
    return image_in_cube
       
def image_closest_to_origin(x, l):   
    """ 
    Computes image of the particle closest to the origin.
    
    :param x: coordinates of particle, stored as NumPy array [[x1. x2. x3.]]
    :param l: length of cube
    :return: coordinates of particle's image closest to the orgin, stored as NumPy array [[x1''. x2''. x3''.]], -l/2 < xi'' < l/2.
    """
    # The most efficient method: add on l/2 before using the modulo operator
    # Correct for this by subtracting l/2 after:
    closest_to_origin = np.mod(x+l/2, l) - l/2
    
    return closest_to_origin

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

def image_closest_to_origin_long_way(x, l):
    """ 
    Computes image of the particle closest to the origin through a less efficient method.
    
    :param x: coordinates of particle, stored as NumPy array [[x1. x2. x3.]]
    :param l: length of cube
    :return: coordinates of particle's image closest to the orgin, stored as NumPy array [[x1''. x2''. x3''.]], -l/2 < xi'' < l/2.
    """
    # Compute coordinates of particle's image within the cube.
    image_in_cube = np.mod(x, l)
    
    # Create an array with dummy components. 
    # These components will be changed to hold the the coordinates of the image closest to the origin.
    closest_to_origin = np.array([0.,0.,0.])
    
    # Compute closest image to origin by checking each coordinate of image_in_cube separately.
    # If coordinate is greater than l/2, subtract the length. If not, keep coordinate the same.
    for i in range(0,3):
        if image_in_cube[i] > l/2:
            closest_to_origin[i] = image_in_cube[i] - l
        else:
            closest_to_origin[i] = image_in_cube[i]
    
    return closest_to_origin
            
# Main method:        
def main():
    # Prompt for position of particle and length of cube.
    print("For a particle at position x = (x1, x2, x3), and a cube of length l: ")
    x1 = float(input("Type the value of the x1 coordinate: "))
    x2 = float(input("Type the value of the x2 coordinate: "))
    x3 = float(input("Type the value of the x3 coordinate: "))
    l = float(input("Type the value of l, the length of the cube: ")) 
    
    # Turn coordinates into a vector stored as a NumPy array.
    x = np.array([x1, x2, x3])
    
    # Print the location of the particle's image in the cube.
    print("\nThe location of the particle's image within the cube is: ", image_in_cube(x,l)) 
    
    # Print the location of the particle's image closest to the origin.
    print("\nThe location of the particle's image closest to the origin is: ", image_closest_to_origin(x,l))
    
    
# Execute main method, but only if it is invoked directly
if __name__ == "__main__":
    main()
