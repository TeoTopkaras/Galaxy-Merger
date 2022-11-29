
This code aims to simulate the collision of two galaxies in 3D space,
and also create a animation with the merge. 
We used a dynamical friction approach in which we consider all 
particles (stars) to be massless, and only the centres to have a mass. 
A halo approximation was also taken into account
for the movement of the particles and the interactions of the two black holes. 
Simple Euler method was used for the force computation
of the new positions and velocities of the particles and the centers. 
The user has to specify the initial positions of the two centers, 
the radius of each galaxy, the masses, the number of particles that wants to be simulated, 
and finally the two quantities, one for the inclination 
and one for the factor projection angle of each galaxy.



The Functions.py and the main.py have to be on the same directory. 
Functions.py contains all the relevant functions that we utilize in the code.
The code needs the python libraries numpy, scipy, astropy, and tqdm in order 
to run properly. The main.py give as an output in a .txt file all the positions 
for the stars and for the centers, and shows the projection figure of the specified frame. 
If one wants to run the 3D simulation, a separate file has created, galaxy_orbits.ipynb, due to the fact 
that you need the jupyter environment to see the animation. The black holes 
at each center are also specified with a black circle. 
