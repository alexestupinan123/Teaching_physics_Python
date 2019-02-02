from visual import*
import numpy as np

'''initial physics parameters'''
g = 9.8 # Gravity acceleration [m/s^2]
b = 1.333 # Bar's mass [kg]
d = 1.661 # Disk's mass [kg]
rd = 1 # Disk's radius [m]
theta_ini = np.pi/6  #initial angle
L = 20.0 # rod length
Ly = input('A que longitud de la barra desea fijar el disco, si la barra mide %0.1f = ' % L) # point to fixed disk

'''initial display parameters'''
pivot = vector(0, 0, 0) # initial coordinates
display (width =600, height =600, center=vector(0,-L/2,0), background=color.white)
rr_ini = vector(L*np.sin(theta_ini), -L*(np.cos(theta_ini)), 0) #final rod position from pivot
rc_ini = rr_ini - (L-Ly)*vector(np.sin(theta_ini), -(np.cos(theta_ini)), 0) #center mass position of disk
roof = box(pos=pivot, size=vector(10, 0.5, 10), color=color.blue) #top square
rod = cylinder(pos=pivot, axis=rr_ini, length=L, radius=0.1, color=color.green) #rod definition
disco = cylinder(pos=rc_ini, axis=(0, 0, 1), length=0.1, radius=rd, color=color.red, make_trail=True, trail_type="points", trail_color=color.orange) # , disk definition
curva=curve(radius=0.1, color=color.orange)

'''initial conditions'''
t = 0 # initial time
dt = 0.01 # step
theta = theta_ini # initial angle
angularvel = 0.0 # initial angle velocity


while(t<100):
    rate(100) # number of steps
    '''cinematic vars in function of time'''
    acc = -g*np.sin(theta)*(b*L/2+d*Ly)/(1/3*b*L**2+d*Ly**2+1/2*d*rd**2) #angular acceleration physics pendulum
    #acc = -(g/L)*np.sin(theta) #angular acceleration simple pendulum
    angularvel = angularvel+acc*dt # angular velocity
    theta = theta + angularvel*dt + 0.5*acc*dt**2 # angular position
    '''updating positions rod and disk'''
    rod.axis = vector(L*np.sin(theta), -L*(np.cos(theta)), 0)
    disco.pos = rod.axis - (L-Ly)*vector(np.sin(theta), -(np.cos(theta)), 0)
    curva.append(disco.pos)
    t = t+dt
