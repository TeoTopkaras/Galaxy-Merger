from Functions import *

#-------------------------------------------------------------
#Input Parameters
#-------------------------------------------------------------
num_part=[3000,3000] #Number of particles for each galaxy 
disc_size=[50,30] #Disk radius for each galaxy in kpc
gal_mass=[10.,5.] #Mass of each galaxy in 10^8 M_sol
cntr_loc=[[0.,0.,0.],[300.,0.,0.]] #Initial positions of the centers
cntr_vel=[[1.,0.,1.],[-1.,0.,-1.]] #Initial velocities of the centers
proj_angle=[np.pi/2.,0.] #Disk projection angle in the ra-dec
incl_angle=[0.,np.pi/4.] #Inclination angle in the ra-dec

#-------------------------------------------------------------
#Simulation Constants
#-------------------------------------------------------------
delta_t = 0.01 #Timestep of the evolution
n_iter = 100 #Number of Euler iteretions per animation frame
simulation_time = 1000 #Time of evolution of the system 
stellar_position_file_name = './particle_positions.txt'
central_position_file_name = './cntr_positions.txt'
generated_particles = './num_particles_generated.txt'
frame = 3 #Frame of choise for the projection figure
telescope_res = 10 #Resolution of the telescope at target redshift
#--------------------------------------------------------------

merger_engine(cntr_loc,cntr_vel,num_part,disc_size,gal_mass,proj_angle,incl_angle,grav_const,delta_t,n_iter,simulation_time,stellar_position_file_name,central_position_file_name,generated_particles)

plot_projection(stellar_file_name,frame,telescope_res)