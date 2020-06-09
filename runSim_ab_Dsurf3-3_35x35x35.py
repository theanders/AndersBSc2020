import hoomd
import hoomd.md
import numpy as np
import sys
import os

import gsd
import gsd.hoomd

hoomd.context.initialize("")

def sim_vol(L,surf,ds): 
    ''' Choose surf to be 1,2 or 3 for P-surface, D-surface or G-surface
        L is a box around the figure '''
    N_points = 100000
    inside = 0
    if surf == 1: #Choose P-surface
        surface = lambda x,y,z : (np.cos(x*(2*np.pi)/L) + np.cos(y*(2*np.pi)/L) + np.cos(z*(2*np.pi)/L))
    if surf == 2: #Choose D-surface
        surface = lambda x,y,z : (np.cos(x*(2*np.pi)/L)*np.cos(y*(2*np.pi)/L)*np.cos(z*(2*np.pi)/L) - np.sin(x*(2*np.pi)/L)*np.sin(y*(2*np.pi)/L)*np.sin(z*(2*np.pi)/L))
    if surf == 3: # Choose G-surface
        surface = lambda x,y,z : (np.sin(x*(2*np.pi)/L)*np.cos(y*(2*np.pi)/L) + np.sin(z*(2*np.pi)/L)*np.cos(x*(2*np.pi)/L) + np.sin(y*(2*np.pi)/L)*np.cos(z*(2*np.pi)/L))
    
    for i in range(N_points):
        x_test, y_test, z_test = np.random.uniform(0,L,3) #Test random x,y,z values
        if surface(x_test,y_test,z_test) >= -ds and surface(x_test,y_test,z_test) <= ds:
            inside += 1 #Count on how many test coordinates is between two surfaces
    return L**3*(inside/N_points) # Find the volume between the two surfaces

def read_off(file):
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')
    n_verts, n_faces, n_dontknow = tuple(int(s) for s in file.readline().strip().split(' '))
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces

#User inputs
bond_k = 4.0 #Harmonic spring constant
bond_r0 = 1.0 #r_0
temperature=1.0 
r_cut = 1.0
dss = '0.14'
ds = float(dss)

#Beads per arm
N_A = 3
N_B = 3

#parameters for LJ potential of the constraining walls
walls_active = True
wallSig = 0.5
wallEps = 1.0
wallr0 = wallSig*pow(2, 1./6.)

L = 35

Gm03 = open("D--%s-0.01.off" % dss)
Gp03 = open("D-%s-0.01.off" % dss)
vm, fm = read_off(Gm03)
vp, fp = read_off(Gp03)
v = ((np.array(vm + vp) % 1.0) - 0.5)*L

Wall = v.tolist()

sim_vol = sim_vol(L,2,ds)
dpd_density = 3
#dpd A coefficent, center bead to all other particles
A_center=25.0
#dpd A coefficent, particles with same color
A_diff=40.0
#dpd A coefficent, particles with different color
A_same=25.0
#warumUp
A_warmup=25.0

N_tot = sim_vol * dpd_density #Total number of particles
N_per_mol = N_A + N_B #Number of particles per molecule
N_mol = int(N_tot/N_per_mol) #Number of molecules
N_wall = len(Wall)
print(sim_vol,N_mol)

sim_steps = 3.5e8
warmup_steps = 5e5
dump_period = 5e5

def initConfig(snapshot,Wall):
    #Get random point in box
    def GetPointOnSurf(s=0):
        while(True):
            x,y,z = np.random.uniform(-L/2+0.2, L/2-0.2, 3)
            res = (np.cos(x*(2*np.pi)/L)*np.cos(y*(2*np.pi)/L)*np.cos(z*(2*np.pi)/L) - np.sin(x*(2*np.pi)/L)*np.sin(y*(2*np.pi)/L)*np.sin(z*(2*np.pi)/L))
            if res > s-0.001 and res < s+0.001:
                break
        return [x,y,z]

    Wall_type = [3] * len(Wall)

    #Particle positions
    part_pos = []
    part_type = []

    #Max offset in any direction
    offset = 0.05

    #Create molecules
    for ii in range(0, N_mol):
        x_c, y_c, z_c = GetPointOnSurf()

        #First particle A bead
        part_pos.append([x_c, y_c, z_c])
        part_type.append(1)

        x_last = x_c
        y_last = y_c
        z_last = z_c

        #A arm
        for jj in range(0,N_A-1):
            #Get offsets
            x_o, y_o, z_o = np.random.uniform(-offset,offset,3)
            #Add the offset
            x_last += x_o
            y_last += y_o
            z_last += z_o

            #Append new particle
            part_pos.append([x_last, y_last, z_last])
            #Append type A=1
            part_type.append(1)
            #print(np.cos(x_last) + np.cos(y_last) + np.cos(z_last))
        
        #Go back to first particle
        x_last = x_c
        y_last = y_c
        z_last = z_c

        for jj in range(0,N_B):
            #Get offsets
            x_o, y_o, z_o = np.random.uniform(-offset,offset,3)
            #Add the offset
            x_last += x_o
            y_last += y_o
            z_last += z_o

            #Append new particle
            part_pos.append([x_last, y_last, z_last])
            #Append type B=2
            part_type.append(2)
            #print(np.cos(x_last) + np.cos(y_last) + np.cos(z_last))
    part_pos = part_pos + Wall
    part_type = part_type + Wall_type
        
    #Assign particle positions
    snapshot.particles.position[:] = part_pos

    #Assign particle types
    snapshot.particles.typeid[:] = part_type

    #Setup bonds
    snapshot.bonds.resize(N_mol* (N_A - 1 + N_B))

    bonds = []

    #Iterate through all center molecules
    for ii in range(0,N_mol):
        #get the index of the center molecule in the pos array
        i=ii*N_per_mol
        #Arm A
        for a in range(0, N_A-1):
            bonds.append([i+a,i+1+a])
            #arm type B
        for a in range(0, N_B):
            if( a==0 ):
                bonds.append([i+a,i+1+a+N_A-1])
            else:
                bonds.append([i+a+N_A-1,i+1+a+N_A-1])
    snapshot.bonds.group[:] = bonds

 
#get a suffix to tell several simruns apart
suffix = "Dsurf_NANB" + str(N_A) + "_ab_" + str(L) + "x" + str(L) +"x" + str(L) + "_"
if( len(sys.argv) <= 1 ):
    suffix += str(int(np.random.uniform(1000, 9999)))
else:
    suffix += sys.argv[1]

dump_file = "trajectory_" + suffix + ".gsd"
print( "dumping simulation to:", dump_file )


#check if this run is a pickup from earlier runs or if we need to create a new configuration
restart = os.path.isfile( dump_file )

init_state = None

if not restart:
    #create a new inital configuration and use that one

    print("creating new random initial configuration")
    
    #init snapshot in all ranks
    snapshot = hoomd.data.make_snapshot(N=N_mol*N_per_mol+N_wall,
                                    box=hoomd.data.boxdim(Lx=L, Ly=L, Lz=L),
                                    particle_types=['A', 'B', 'D', 'W'],
                                    bond_types=['harmonic'])

    initConfig(snapshot,Wall)

    #load initial configuration in each rank
    init_state = hoomd.init.read_snapshot(snapshot)  

    groupA = hoomd.group.type(type='A')
    groupB = hoomd.group.type(type='B')
    groupD = hoomd.group.type(type='D')
    groupW = hoomd.group.type(type='W')

    #dump the initial configuration 
    hoomd.dump.gsd(dump_file, period=None, group=hoomd.group.union('name1',hoomd.group.union('name2', groupA, groupB), groupD), overwrite=False, truncate=False, phase=0, dynamic=['property', 'momentum'])
    hoomd.dump.gsd("trajectory_%s_load.gsd" % suffix, period=None, group=hoomd.group.all(), overwrite=False, truncate=False, phase=0, dynamic=['property', 'momentum'])
else:
    #find last frame, since we're using an older version of hoomd, we
    #need this workaround

    print("continuing simulation")
    
    traj = gsd.hoomd.open( "trajectory_%s_load.gsd" % suffix, 'rb' )
    last_frame = len(traj) - 1
    print("opening %s at frame %i" % ("trajectory_%s_load.gsd" % suffix, last_frame) )
    init_state = hoomd.init.read_gsd( "trajectory_%s_load.gsd" % suffix, frame=last_frame )
    groupA = hoomd.group.type(type='A')
    groupB = hoomd.group.type(type='B')
    groupD = hoomd.group.type(type='D')
    groupW = hoomd.group.type(type='W')



#hoomd.dump.gsd("initial_%s.gsd" % suffix, period=None, group=hoomd.group.union('name1',hoomd.group.union('name2', groupA, groupB),groupD), overwrite=True, truncate=False, phase=0, dynamic=['property', 'momentum'])
#hoomd.dump.gsd("initial_%s_Wall.gsd" % suffix, period=None, group=hoomd.group.type(type='W'), overwrite=True, truncate=False, phase=0, dynamic=['property', 'momentum'])

nl = hoomd.md.nlist.cell()

dpd = hoomd.md.pair.dpd(r_cut=r_cut, nlist=nl, kT=temperature, seed=1)
dpdLJ = hoomd.md.pair.force_shifted_lj(r_cut=wallr0, nlist=nl)


nl.reset_exclusions(exclusions = [])

harmonic = hoomd.md.bond.harmonic()

harmonic.bond_coeff.set('harmonic', k=bond_k, r0=bond_r0)

hoomd.md.integrate.mode_standard(dt=0.005)

all = hoomd.group.all()
groupA = hoomd.group.type(type='A')
groupB = hoomd.group.type(type='B')
groupD = hoomd.group.type(type='D')
groupW = hoomd.group.type(type='W')
groupParticles = hoomd.group.union('name1',hoomd.group.union('name2', groupA, groupB),groupD)
groupWall = hoomd.group.type(type='W')

#hoomd.md.integrate.nve(group=all)

#integrator = hoomd.md.integrate.nve(group=all)

integrator = hoomd.md.integrate.nve(group = groupParticles)
hoomd.md.integrate.nve(group = groupWall, zero_force=True)

#Add walls r_cut=False excludes interactions from simulation
dpd.pair_coeff.set('W', 'W', A=0, gamma=0, r_cut=False)
dpd.pair_coeff.set('W', 'A', A=0, gamma=0, r_cut=False)
dpd.pair_coeff.set('W', 'B', A=0, gamma=0, r_cut=False)
dpd.pair_coeff.set('W', 'D', A=0, gamma=0, r_cut=False)

dpdLJ.pair_coeff.set('A', 'A', sigma=0, epsilon=0, r_cut=False)
dpdLJ.pair_coeff.set('A', 'B', sigma=0, epsilon=0, r_cut=False)
dpdLJ.pair_coeff.set('A', 'D', sigma=0, epsilon=0, r_cut=False)

dpdLJ.pair_coeff.set('B', 'B', sigma=0, epsilon=0, r_cut=False)
dpdLJ.pair_coeff.set('D', 'D', sigma=0, epsilon=0, r_cut=False)

dpdLJ.pair_coeff.set('B', 'D', sigma=0, epsilon=0, r_cut=False)

dpdLJ.pair_coeff.set('W', 'W', sigma=0, epsilon=0, r_cut=False)
dpdLJ.pair_coeff.set('W', 'A', sigma=wallSig, epsilon=wallEps)
dpdLJ.pair_coeff.set('W', 'B', sigma=wallSig, epsilon=wallEps)
dpdLJ.pair_coeff.set('W', 'D', sigma=wallSig, epsilon=wallEps)

#Warmup
if (restart and traj[-1].configuration.step < warmup_steps) or not restart:
    dpd.pair_coeff.set('A', 'A', A=A_warmup, gamma = 4.5)
    dpd.pair_coeff.set('A', 'B', A=A_warmup, gamma = 4.5)
    dpd.pair_coeff.set('A', 'D', A=A_warmup, gamma = 4.5)

    dpd.pair_coeff.set('B', 'B', A=A_warmup, gamma=4.5)
    dpd.pair_coeff.set('D', 'D', A=A_warmup, gamma=4.5)

    dpd.pair_coeff.set('B', 'D', A=A_warmup, gamma=4.5)


    #limit forces, so it can equilibrate without too big forces
    integrator.set_params(limit=0.02)
    hoomd.run_upto( int(0.1*warmup_steps) )

    #lift force limit and finish warm up
    integrator.set_params(limit=None)
    hoomd.run_upto( warmup_steps )
else:
    print("Skipping warmup")

#Actual sim
dpd.pair_coeff.set('A', 'A', A=A_center, gamma = 4.5)
dpd.pair_coeff.set('A', 'B', A=A_center, gamma = 4.5)
dpd.pair_coeff.set('A', 'D', A=A_center, gamma = 4.5)

dpd.pair_coeff.set('B', 'B', A=A_same, gamma=4.5)
dpd.pair_coeff.set('D', 'D', A=A_same, gamma=4.5)

dpd.pair_coeff.set('B', 'D', A=A_diff, gamma=4.5)


hoomd.analyze.log(filename="log-output_%s.log" % suffix,
                  quantities=['potential_energy', 'temperature'],
                  period=500,
                  overwrite=True)

hoomd.dump.gsd(dump_file, period=dump_period, group=groupParticles, overwrite=True, truncate=False, phase=0, dynamic=['property', 'momentum'])
hoomd.dump.gsd("trajectory_%s_load.gsd" % suffix, period=dump_period, group=all, overwrite=True, truncate=False, phase=0, dynamic=['property', 'momentum'])

hoomd.run(sim_steps + warmup_steps)

hoomd.dump.gsd("final_%s.gsd" % suffix, period=None, group=groupParticles, overwrite=True, truncate=False, phase=0, dynamic=['property', 'momentum'])