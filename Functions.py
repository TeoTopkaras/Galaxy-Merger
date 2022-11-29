import numpy as np 
import scipy.constants as cst
from matplotlib import *
import matplotlib.pyplot as plt
from astropy import convolution as conv
from scipy.signal import convolve as scipy_convolve
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import time

grav_const = 449.8502 #kpc3/1e8M_sol/(1e9yr)2


def write_txt(data,file_name):
    '''
    Parameters
    ---------------
    data : ndarray 
        x-coordinates for each timestep
    file_name : string
        Name of file

    Returns
    ---------------
    File containing all the positions specified to store
    '''
    
    file = open(file_name, "a")
    file.write(str(data[0][0])+','+str(data[0][1])+','+str(data[0][2]))
    for i in range(1,len(data)):
        file.write(','+str(data[i][0])+','+str(data[i][1])+','+str(data[i][2]))
    file.write('\n')
    file.close()

    


def merger_engine(cntr_pos,cntr_vel,num_part,disc_size,gal_mass,x_angle,z_angle,grav_const,delta_t,n_iter,simulation_time,particle_file,cntr_file,generated_particles):
    '''
    Parameters
    ---------------
    cntr_pos : ndarray 
        The radial norm for the corresponging vector distance
    cntr_vel : ndarray
        Velocities of the centers
    num_part : scalar
        Number of particles    
    disc_size : scalar
        Radius of the galactic disk 
    gal_mass : scalar
        Total mass of the corresponding galaxy
    x_angle: scalar
        b/a factor 
    z_angle: scalar 
        Rotation angle 
    grav_const : scalar
        grav_constravitational constant in units kpc^3/1e8M_sol/(1e9yr)^2
    delta_t : scalar
        Timestep of the simulation 
    n_iter : scalar
        Evolution time of the Euler function 
    simulation_time : scalar
        Number of frames for the simulation
    
    particle_file : string
        Name of file for the particle positions
    cntr_file : string
        Name of file for the black hole positions
    generated_particles : string
	File containing the total number of plotted particles for each galaxy
    
    Returns
    ---------------
    n_tot : ndarray
        Total number of plotted particles for each galaxy 
    '''
    stellar_pos=[]
    stellar_vel=[]
    n_tot=[0.,0.]

    for i in range(0,len(num_part)):
        pos,vel,n_tot[i]=stellar_disc(cntr_pos[i],cntr_vel[i],num_part[i],disc_size[i],gal_mass[i],x_angle[i],z_angle[i],grav_const)
        stellar_pos.extend(pos)
        stellar_vel.extend(vel)
    
    file = open(generated_particles, "a")
    file.write(str(n_tot[i]))
    for i in range(1,len(n_tot)):
        file.write(','+str(n_tot[i]))
    file.close()
                   
    merger_flag = False

    for frames in tqdm(range(simulation_time)):
        stellar_pos,stellar_vel,cntr_pos,cntr_vel,merger_flag=expl_Euler(\
                stellar_pos,stellar_vel,cntr_pos,cntr_vel,disc_size,gal_mass,delta_t,n_iter,merger_flag)
        write_txt(stellar_pos,particle_file)
        write_txt(cntr_pos,cntr_file)
                   

        time.sleep(0.5)




def read_txt(file_name):
    '''
    Parameters
    ---------------
    file_name : string 
        Name of the file you want to read
    
    Returns
    ---------------
    data : ndarray
        Positions of all the stars 
    '''
    data=[]
    with open(file_name, 'r') as f:
        lines = f.readlines() 
    for raw_line in lines:
        split_line = raw_line.strip().split(",")
        num_ls = [float(x) for x in split_line]
        data.append([[num_ls[3*i+0],num_ls[3*i+1],num_ls[3*i+2]] for i in range(int(len(num_ls)/3))])

    return data







def plot_projection(file_name,frame,telescope_res):
    '''
    Parameters
    ---------------
    file_name : string 
        Name of the file you want to read
    frame : int
        Number of frame for the projection
    telescope_res : float
        Number used to replicate the telescope resolution
    Returns
    ---------------
    Plots the isolight curves of the corresponing projected frame on the ra-dec plane
    '''
    

    pos=read_txt(file_name)

    x = [pos[frame][i][0] for i in range(len(pos[0]))]
    z = [pos[frame][i][2] for i in range(len(pos[0]))]
    x_fit=[]
    z_fit=[]

    xmean, zmean = np.median(x), np.median(z)
    x -= xmean
    z -= zmean

    isolight_curves = np.zeros((300,300))

    for i in range(0,len(pos[0])):
        if (np.abs(x[i])<=150.) and (np.abs(z[i])<=150.):
            x_fit.append(x[i])
            z_fit.append(z[i])
            isolight_curves[150+int(z[i])][150+int(x[i])]+=1

    psf = conv.Gaussian2DKernel(telescope_res)
    isolight_curves = scipy_convolve(isolight_curves, psf, mode='same', method='direct')



    fig=plt.figure(figsize=(7,7), constrained_layout=True)
    plt.imshow(isolight_curves, origin='lower',cmap='hot')      
    plt.contour(isolight_curves, levels=[0.2*np.max(isolight_curves),0.5*np.max(isolight_curves),
                                         0.8*np.max(isolight_curves)], colors='blue')

    plt.plot([220,270],[20,20], color="white") 
    plt.text(222,22,'50 kpc',fontsize=12,color='white')
    plt.axis('off')
    plt.show()


def Internal_Mass(dist,disc_size,gal_mass,grav_const):
    '''
    Parameters
    ---------------
    dist : ndarray 
        The radial norm for the corresponging vector distance
    disc_size : scalar
        Radius of the galactic disk 
    gal_mass : scalar
        Total mass of the corresponding galaxy
    grav_const : scalar
        Gravitational constant in units kpc^3/1e8M_sol/(1e9yr)^2
    
    Returns
    ---------------
    int_mass : scalar
        The internal mass depending on the location
    '''
    halo_radius = 2*disc_size
    halo_velocity = 2*np.sqrt(grav_const*gal_mass/halo_radius)
    if dist < halo_radius:
        int_mass=halo_velocity**2*dist**3/(grav_const*(dist + halo_radius)**2)
    else:
        int_mass = gal_mass
        
    return int_mass 

def stellar_disc(cntr_pos,cntr_vel,num_part,disc_size,gal_mass,x_angle,z_angle,grav_const):
    '''
    Parameters
    ---------------
    cntr_pos : ndarray 
        The radial norm for the corresponging vector distance
    cntr_vel : ndarray
        Velocities of the centers
    disc_size : scalar
        Radius of the galactic disk 
    num_part : scalar
        Number of particles
    gal_mass : scalar
        Total mass of the corresponding galaxy
    x_angle: scalar
        b/a factor 
    z_angle: scalar 
        Rotation angle 
    grav_const : scalar
        Gravitational constant in units kpc^3/1e8M_sol/(1e9yr)^2
    
    Returns
    ---------------
    np.array(stellar_pos) : ndarray
        Initial positions for all particles 
    np.array(stellar_vel) : ndarray
        Initial velocities for  all particles
    len(stellar_pos) : ndarray
        Total number of particles plotted
    '''
    stellar_pos=[]
    stellar_vel=[]
    
    num_part_0=num_part*3.2/(disc_size*(1-np.exp(-3.2)))
    radius=np.round(np.arange(1,disc_size,1),1)
    
    x_rot=np.array(((1.,0.,0.),(0.,np.cos(x_angle),-np.sin(x_angle)),(0.,np.sin(x_angle),np.cos(x_angle))))
    z_rot=np.array(((np.cos(z_angle),0.,np.sin(z_angle)),(0.,1.,0.),(-np.sin(z_angle),0.,np.cos(z_angle))))
    rot=np.dot(x_rot,z_rot)

    for j in range(0,len(radius)):
        n_segment=int(num_part_0*np.exp(-3.2*radius[j]/disc_size)+0.5)
        for i in range(0,n_segment):
            angle=(i*(2*np.pi)/n_segment)+(j*0.2)
            pos=[radius[j]*np.cos(angle),radius[j]*np.sin(angle),0]
            vel=[np.sqrt(grav_const*Internal_Mass(radius[j],disc_size,gal_mass,grav_const)/radius[j])*np.cos(angle+np.pi/2.)\
                 ,np.sqrt(grav_const*Internal_Mass(radius[j],disc_size,gal_mass,grav_const)/radius[j])*np.sin(angle+np.pi/2.),0.]
            stellar_pos.append([cntr_pos[k]+np.dot(rot,pos)[k] for k in range(3)])
            stellar_vel.append([cntr_vel[k]+np.dot(rot,vel)[k] for k in range(3)])

    return (np.array(stellar_pos),np.array(stellar_vel),len(stellar_pos))


def galaxy_density_profile(dist,disc_size,gal_mass,grav_const):
    '''
    Parameters
    ---------------
    dist : ndarray 
        The radial norm for the corresponging vector distance
    disc_size : scalar
        Radius of the galactic disk 
    gal_mass : scalar
        Total mass of the corresponding galaxy
    grav_const : scalar
        Gravitational constant in units kpc^3/1e8M_sol/(1e9yr)^2
    
    Returns
    ---------------
    dM/dV : scalar
        The density of the thin layer containing mass equal to M_outer - M_inner
    '''
    inner_radius=0.99*dist
    outer_radius=1.01*dist
    M_inner=Internal_Mass(inner_radius,disc_size,gal_mass,grav_const)
    M_outer=Internal_Mass(outer_radius,disc_size,gal_mass,grav_const)
    
    dM = M_outer-M_inner
    dV = (4/3)*np.pi*(outer_radius**3-inner_radius**3)
    return dM/dV # In M_{\odot}*kpc^-3


def dynamical_friction(dist,v_i,v_j,disc_size,gal_mass_i,gal_mass_j,grav_const):
    '''
    Parameters
    ---------------
    dist : ndarray 
        The radial norm for the corresponging vector distance
    v_i : ndarray
        Velocity vector of the main galaxy black hole
    v_j : ndarray
        Velocity vector of the companion galaxy black hole
    disc_size : scalar
        Radius of the galactic disk 
    gal_mass_i : scalar
        Total mass of the main galaxy
    gal_mass_j : scalar
        Total mass of the companion galaxy
    grav_const : scalar
        Gravitational constant in units kpc^3/1e8M_sol/(1e9yr)^2
    
    Returns
    ---------------
    friction : ndarray
        Dymanical friction accordning to the density profile of the timestep
    '''
    rho = galaxy_density_profile(dist,disc_size,gal_mass_j,grav_const)
    vij = [(v_i[k]-v_j[k]) for k in range(3)]
    v_length = np.linalg.norm(vij)
    friction = [-12*np.pi*grav_const*gal_mass_i*rho*vij[k]/((1.+v_length)**3) for k in range(3)]
    return friction




def particle_force(r_i,r_j,disc_size,gal_mass,grav_const):
    '''
    Parameters
    ---------------
    r_i : ndarray
        Position vector of the main galaxy particles
    r_j : ndarray
        Position vector of the companion galaxy particles
    disc_size : scalar
        Radius of the galactic disk 
    gal_mass : scalar
        Total mass of the corresponding galaxy
    grav_const : scalar
        Gravitational constant in units kpc^3/1e8M_sol/(1e9yr)^2
    
    Returns
    ---------------
    Force_particles : ndarray
        Force vector for the massless particles 
    '''
    Force_particles = np.zeros(3)
    rij=[(r_i[k]-r_j[k]) for k in range(3)]
    r_length=np.linalg.norm(rij)
    Force_particles = [-(grav_const*Internal_Mass(r_length,disc_size,gal_mass,grav_const)*rij[k]/r_length**3.) for k in range(3)]
    return Force_particles




def cntr_force(r_i,r_j,v_i,v_j,disc_size,gal_mass_i,gal_mass_j,grav_const):
    '''
    Parameters
    ---------------
    r_i : ndarray
        Position vector of the main galaxy black hole
    r_j : ndarray
        Position vector of the companion galaxy black hole
    v_i : ndarray
        Velocity vector of the main galaxy black hole
    v_j : ndarray
        Velocity vector of the companion galaxy black hole
    disc_size : scalar
        Radius of the galactic disk 
    gal_mass_i : scalar
        Total mass of the main galaxy
    gal_mass_j : scalar
        Total mass of the companion galaxy
    grav_const : scalar
        Gravitational constant in units kpc^3/1e8M_sol/(1e9yr)^2
    
    Returns
    ---------------
    Force_cntr : ndarray
        Force vector for the galaxy cntr 
    '''
    Force_cntr = np.zeros(3)
    rij=[(r_i[k]-r_j[k]) for k in range(3)]
    r_length=np.linalg.norm(rij)
    friction=dynamical_friction(r_length,v_i,v_j,disc_size,gal_mass_i,gal_mass_j,grav_const)
    Force_cntr =[-(grav_const*Internal_Mass(r_length,disc_size,gal_mass_j,grav_const)*rij[k]/r_length**3.)+friction[k] for k in range(3)]

    return Force_cntr


def expl_Euler(stellar_pos,stellar_vel,cntr_pos,cntr_vel,disc_size,gal_mass,delta_t,n_iter,merger_flag):
    '''
    Parameters
    ---------------
    stellar_pos : ndarray
        Position vector of all particles
    stellar_vel : ndarray
        Velocity vector of all particles
    cntr_pos : ndarray
        Position vector of the black holes
    cntr_vel : ndarray
        Velocity vector of the black holes
    disc_size : scalar
        Radius of the galactic disk 
    gal_mass : scalar
        Total mass of the corresponding galaxy
    delta_t : scalar
        Timestep of the simulation 
    n_iter : scalar
        Evolution time of the Euler function 
    merger_flag : Boolean
        False if there isn't a collision, True if the central black holes collide
        
    
    Returns
    ---------------
    stellar_pos : ndarray
        New stellar positions according to Euler method
    stellar_vel : ndarray
        New stellar velocities according to Euler method
    cntr_pos : ndarray
        New black hole positions according to Euler method
    cntr_vel : ndarray
        New black hole velocities according to Euler method
    merger_flag : Boolean
        Force vector for the central black hole of the main galaxy due to the companion
    '''

    for t in range(0,n_iter):
    
        Force_stars = np.zeros((len(stellar_pos),3))
        Force_cntr = np.zeros((len(cntr_pos),3))
        
        if merger_flag==False:
            for i in range(0,len(cntr_pos)):
                for j in range(0,len(cntr_pos)):
                    if i!=j:
                        Fij = cntr_force(cntr_pos[i],cntr_pos[j],cntr_vel[i],cntr_vel[j],disc_size[j],gal_mass[i],gal_mass[j],grav_const)
                        Force_cntr[i] = np.add(Force_cntr[i],Fij)
    
        for i in range(0,len(stellar_pos)):
            for j in range(0,len(cntr_pos)):
                Fij = particle_force(stellar_pos[i],cntr_pos[j],disc_size[j],gal_mass[j],grav_const)
                Force_stars[i] = np.add(Force_stars[i],Fij)
    

        stellar_pos=[[(stellar_pos[i][k]+(stellar_vel[i][k]*delta_t)) for k in range(3)] for i in range(len(stellar_pos))]
        cntr_pos=[[(cntr_pos[i][k]+(cntr_vel[i][k]*delta_t)) for k in range(3)] for i in range(len(cntr_pos))]
        stellar_vel=[[stellar_vel[i][k]+delta_t*Force_stars[i][k] for k in range(3)] for i in range(len(stellar_pos))]
        cntr_vel=[[cntr_vel[i][k]+delta_t*Force_cntr[i][k] for k in range(3)] for i in range(len(cntr_pos))]
        
        if merger_flag==False:
            cntr_dist=[(cntr_pos[0][k]-cntr_pos[1][k]) for k in range(3)]
            if np.linalg.norm(cntr_dist)<= 0.5 :
                merger_flag=True
                cntr_pos=[[(gal_mass[0]*cntr_pos[0][k]+gal_mass[1]*cntr_pos[1][k])/np.sum(gal_mass) for k in range(3)]\
                        for i in range(len(cntr_pos))]
                cntr_vel=[[(gal_mass[0]*cntr_vel[0][k]+gal_mass[1]*cntr_vel[1][k])/np.sum(gal_mass) for k in range(3)]\
                        for i in range(len(cntr_pos))]
        
        
        
    return (stellar_pos,stellar_vel,cntr_pos,cntr_vel,merger_flag)