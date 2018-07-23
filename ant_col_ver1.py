#this piece of code will just let the ants
#choose their paths and then send them to
#mpi processes to call the lstm algorithm
#to calculate the fitness, then it will collect
#those fintnesses and compare them to get the lowest (best) one



import numpy as np
from mpi4py import MPI
from random import random
from net_mesh import network_mesh
from LSTM_4_1_3_itr_ver04_ant import art_i


#an array for the initial interlayer connections
pheromons_0 = np.ones(16)

#an array for the first interlayer connections
pheromons_1 = np.ones((16, 16))


PHORMONS_THERESHOLD = 20
PHORMONS_REDUCTION_INTERVAL = 100

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
status = MPI.Status()


def construct_colony():
    colony = network_mesh()
    for ant in range(ANTS):                     #Let ants generate the paths based on pheromones
        pherm_sum = np.sum(pheromons_0)
        rand_gen = random() * pherm_sum

        for a0 in range(len(pheromons_0)):      #Ant moves from initial position to the input an neuron
            if rand_gen < pheromons_0[a0]:
                break
            else:
                rand_gen-= pheromons_0[a0]

        pherm_sum = np.sum(pheromons_1[a0])
        rand_gen = random() * pherm_sum

        for a1 in range(len(pheromons_1[a0])):  #Ant moves from an input neuron to a hidden neuron
            if rand_gen < pheromons_1[a0,a1]:
                break
            else:
                rand_gen-= pheromons_1[a0,a1]


        colony.mesh_1[a0,a1]    = 1.0
        colony.mesh_2[a1]       = 1.0

    return colony

def modify_pheromones(mesh1, mesh2, condition):
    dum_pheromons_1 = np.zeros(len(mesh1))
    for p1 in range(len(mesh1)):
        if np.sum(mesh1[p1])>0:
            if condition=='G':
                pheromons_0[p1]*=1.15
            elif condition=='B':
                pheromons_0[p1]*=0.85
            dum_pheromons_1+=mesh1[p1]
    indecies_dum = dum_pheromons_1>0
    if condition=='G':
        pheromons_1[indecies_dum]*=1.15
    elif condition=='B':
        pheromons_1[indecies_dum]*=0.85

    indecies_dum = pheromons_0>PHORMONS_THERESHOLD
    pheromons_0[indecies_dum] = PHORMONS_THERESHOLD
    indecies_dum = pheromons_1>PHORMONS_THERESHOLD
    pheromons_1[indecies_dum] = PHORMONS_THERESHOLD


#Worker Processes
if rank!=0:
    while True:
        slave_recv_package = comm.recv(source=0, tag=13)
        if slave_recv_package[0]==-1:
            break

        #The below two lines are for illustration and they can be replaced by directly altering the below third line:
        #fitness = art_i('ant', slave_recv_package[0], slave_recv_package[1]).res_err
        itr = slave_recv_package[0]
        colony = slave_recv_package[1]
        fitness = art_i('ant', itr, colony).res_err     #The parameter 'ant' is used to tell the object art_i that the...
                                                        #weight matrices will be modified by the ACO generated meshes
        slave_send_package = [itr, fitness, np.sum(colony.mesh_1) + np.sum(colony.mesh_2)]
        comm.send(slave_send_package, dest=0, tag=11)

#Master Process
if rank==0:
    #number of ants
    ANTS   = 200

    #colony iterations
    ITERs = 500 + 1

    #fitnessses collection
    FITNESS_REC = []

    #colonies
    COLONYs = []

    #max fitness
    max_fit = [0,100000000]


    for itr in range(1, size):                          #Send colonies to worker process
        colony = construct_colony()
        COLONYs.append([itr, colony])
        master_send_package = [itr, colony]
        comm.send(master_send_package, dest=itr, tag=13)
        if itr%PHORMONS_REDUCTION_INTERVAL==0:
            pheromons_0*=0.9
            pheromons_1*=0.9

    for itr in range(size, ITERs):                                                      #Recieve results and send more colonies
        master_recv_package = comm.recv(source=MPI.ANY_SOURCE, tag=11, status=status)
        FITNESS_REC.append(master_recv_package)

        if  master_recv_package[1] in range(0, np.sort(np.array(FITNESS_REC)[:,1])[10]):         #Modify pheromones based on ants' paths in a best fit network
            if max_fit[1]>master_recv_package[1]:
                max_fit = master_recv_package[0:2]
            modify_pheromones(COLONYs[max_fit[0]-1][1].mesh_1, COLONYs[max_fit[0]-1][1].mesh_2, 'G')
        else:
            modify_pheromones(COLONYs[int(max_fit[0])-1][1].mesh_1, COLONYs[int(max_fit[0])-1][1].mesh_2, 'B')
        if itr%PHORMONS_REDUCTION_INTERVAL==0:
            pheromons_0*=0.9
            pheromons_1*=0.9

        colony = construct_colony()
        COLONYs.append([itr, colony])
        master_send_package = [itr, colony]
        comm.send(master_send_package, dest= status.Get_source() , tag=13)
        if itr%PHORMONS_REDUCTION_INTERVAL==0:
            pheromons_0*=0.9
            pheromons_1*=0.9



    for itr in range(size-1):                          #Recive the last batch
        master_recv_package = comm.recv(source=MPI.ANY_SOURCE, tag=11, status=status)
        FITNESS_REC.append(master_recv_package)

        if max_fit[1]>master_recv_package[1]:          #Modify pheromones based on ants' paths in a best fit network
            max_fit = master_recv_package[0:2]
            modify_pheromones(COLONYs[max_fit[0]-1][1].mesh_1, COLONYs[max_fit[0]-1][1].mesh_2, 'G')
        else:
            modify_pheromones(COLONYs[max_fit[0]-1][1].mesh_1, COLONYs[max_fit[0]-1][1].mesh_2, 'B')
        if itr%PHORMONS_REDUCTION_INTERVAL==0:
            pheromons_0*=0.9
            pheromons_1*=0.9

    np.save("FITNESS_REC", FITNESS_REC)
    np.save("COLONYs", COLONYs)

    master_send_package = [-1, -1]

    for i in range(1, size):                       #Terminate Worker Processes
        comm.send(master_send_package, dest=i, tag=13)
