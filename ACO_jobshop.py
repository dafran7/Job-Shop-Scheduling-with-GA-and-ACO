def extract_data(file_target):
    file_read = []
    with open(file_target, 'r') as file:
        file_read += file.readlines()

    extract_data = []
    for i in range(0, len(file_read)):
        if file_read[i][0] != "#":
            extract_data.append(file_read[i])

    jmlh_job = int(extract_data[0].split()[0])
    jmlh_mesin = int(extract_data[0].split()[1])

    job_shop = [[0 for _ in range(jmlh_mesin)] for _ in range(jmlh_job)]
    order_mch = [[0 for _ in range(jmlh_mesin)] for _ in range(jmlh_job)]
    for i in range(1, jmlh_job + 1):
      for j in range(0, jmlh_mesin):
            ind_mesin = int(extract_data[i].split()[j * 2])
            job_shop[i - 1][j] = int(extract_data[i].split()[j * 2 + 1])

            order_mch[i - 1][j] = ind_mesin + 1

    return job_shop, order_mch, jmlh_job, jmlh_mesin

def calculateProbability(tour, flip, edges, pheromones, a, b):
    phero_matrix = []
    for p in pheromones:
        if pheromones.index(p) in tour:
            if p==0:
                phero_matrix.append(0)
            else:
                phero_matrix.append((p*0.00001)**a)
        else:
            phero_matrix.append(p**a)

    #phero_matrix = [x**a for x in pheromones]
    if (flip==1):
        edges = [(1/x)**b for x in edges]
    else:
        edges = [x**b for x in edges]
    top = []
    for i in range(len(edges)):
        top.append(phero_matrix[i]*edges[i])
    bottom = sum(top)

    return [x/bottom for x in top]

def select_from_wheel(probability_matrix):
    rand_num = random.randrange(0,100000)/100000
    prob_cumm = 0
    k = 0
    for prob in probability_matrix:
        prob_cumm += prob
        #print(str(rand_num)+"->"+str(prob_cumm))

        if rand_num < prob_cumm:
            return k #probability_matrix.index(prob)
        k += 1

def selectJob(tour, flip, edges, pheromones, count_op, alpha, beta, q_threshold):
    pheroNow = []
    edgeNow = []
    counter = count_op
    for i in range(num_job):
        #print("pero"+str(pheromones[i]))
        #print("cont" + str(count_op[i]))
        if counter[i] > (num_mach-1):
            pheroNow.append(0)
            edgeNow.append(0.1)
        else:
            pheroNow.append(pheromones[i][counter[i]])
            edgeNow.append(edges[i][counter[i]])

    probability = calculateProbability(tour, flip, edgeNow, pheroNow, alpha, beta)
    #print(probability)
    rand_num = random.randrange(0,100000)/100000
    result = 0
    txt = ""
    if rand_num <= q_threshold:
        result = probability.index(max(probability))
        txt = "MX"+str(result)
    else:
        result = select_from_wheel(probability)
        txt = "WH"+str(result)
    #print(txt)
    return result

def calculateMakespan(num_job, num_mach, tour, counter, pt, ms):
    #time_spent = [[0 for _ in range(num_job)] for _ in range(num_mach)]
    time_spent = [0 for _ in range(num_mach)]
    for i in range(num_job*num_mach):
        ind_machNow = ms[tour[i]][counter[i]]
        ind_machBef = ms[tour[i]][counter[i]-1]
        if time_spent[ind_machNow-1] < time_spent[ind_machBef-1]:
            time_spent[ind_machNow-1] = time_spent[ind_machBef-1] + pt[tour[i]][counter[i]]
        else:
            time_spent[ind_machNow-1] += pt[tour[i]][counter[i]]
        #time_spent[ms[selectedJob][count_op[selectedJob]]][count_op[selectedJob]] = pt[selectedJob][count_op[selectedJob]]

    makespan = max(time_spent)
    #if makespan < 55:
        #print(time_spent)
    return makespan

def createAnt(num_job, num_mach, edge, ms, phero_matrix, alpha, beta, q_threshold):
    count_op = [0 for _ in range(num_job)]
    tour = []
    counter = []
    flip_order = 1
    for i in range(num_job*num_mach):
        # Select Job
        selectedJob = selectJob(tour, flip_order, edge, phero_matrix, count_op, alpha, beta, q_threshold)
        tour.append(selectedJob)
        counter.append(count_op[selectedJob])
        #edge[selectedJob][count_op[selectedJob]+1:] = [x+edge[selectedJob][count_op[selectedJob]] for x in edge[selectedJob][count_op[selectedJob]+1:]]  # Add job machine process time to the latter process time
        count_op[selectedJob] += 1
        flip_order *= (-1)

    fitness = calculateMakespan(num_job, num_mach, tour, counter, edge, ms)
    return tour, fitness

def createColony(num_ant, num_job, num_mach, pt, ms, phero_matrix, alpha, beta, q_threshold):
    colony_tour = []
    colony_makespan = []
    for i in range(num_ant):
        ant_tour, makespan = createAnt(num_job, num_mach, pt, ms, phero_matrix, alpha, beta, q_threshold)
        colony_tour.append(ant_tour)
        colony_makespan.append(makespan)

    return colony_tour, colony_makespan

def updatePheromone(phero_matrix, num_job, num_mach, colony_tour, colony_makespan):
    num_ant = len(colony_tour)

    for i in range(num_ant):
        j=0
        job_priority = []
        job_priority_order = 1
        flips = 1
        job_count = [0 for _ in range(num_job)]
        while j != num_job*num_mach:
            currentJob = colony_tour[i][j]
            currentMach = job_count[currentJob]

            phero_matrix[currentJob][currentMach] += 1/(colony_makespan[i] * job_priority_order**flips)
            phero_matrix[currentJob][currentMach] = round(phero_matrix[currentJob][currentMach],5)
            if currentJob not in job_priority:
                job_priority.append(currentJob)
                job_priority_order += 1

            j += 2
            job_count[currentJob] += 1
            flips *= (-1)

    return phero_matrix

'''==========Solving job shop scheduling problem by gentic algorithm in python======='''
# importing required modules
import pandas as pd
import numpy as np
import time
import copy

''' ================= initialization setting ======================'''

# pt_tmp=pd.read_excel("JSP_dataset.xlsx",sheet_name="Processing Time",index_col =[0])
# ms_tmp=pd.read_excel("JSP_dataset.xlsx",sheet_name="Machines Sequence",index_col =[0])

pt, ms, num_job, num_mach = extract_data("./instances/ft06")

# dfshape=pt_tmp.shape
# num_mc=dfshape[1] # number of machines
# num_job=dfshape[0] # number of jobs
num_gene = num_mach * num_job  # number of genes in a chromosome
print("Testing\t:\n\t"+str(pt)+"\n\t"+str(ms)+"\n\t"+"Number of genes in chromosomes :"+str(num_gene)+"\n")

# pt=[list(map(int, pt_tmp.iloc[i])) for i in range(num_job)]
# ms=[list(map(int,ms_tmp.iloc[i])) for i in range(num_job)]


# raw_input is used in python 2
"""population_size = int(input('Please input the size of population: ') or 30)  # default value is 30
crossover_rate = float(input('Please input the size of Crossover Rate: ') or 0.8)  # default value is 0.8
mutation_rate = float(input('Please input the size of Mutation Rate: ') or 0.2)  # default value is 0.2
mutation_selection_rate = float(input('Please input the mutation selection rate: ') or 0.2)
num_mutation_jobs = round(num_gene * mutation_selection_rate)
num_iteration = int(input('Please input number of iteration: ') or 1000)  # default value is 2000
"""
start_time = time.time()

'''----------start----------'''
import random
import operator
from math import *
from statistics import *



# Inisialisasi parameter ACO
antNo = 25
iterasiNo = 100

rho = 0.2
alpha = 1
beta = 1
q_threshold = 0.9

phero_matrix = []
for i in range(num_job):
    tau_0 = [0 for _ in range(num_mach)]
    for j in range(num_mach):
        tau_0[j] = round(1 / (num_job * (abs(sum(pt[i][:]) - sqrt(pt[i][1])))), 5) # Initial Pheromones
    phero_matrix.append(tau_0)
"""for i in range(num_job):
    tau_0 = round(1 / (num_job / abs(sum(pt[i][:(num_job//2)]) + sqrt(pt[i][1]*2))), 5)
    #tau_0 = round(1 / abs((num_job / sum(pt[i][:(num_job//2)])) - sqrt(pt[i][1]*2)),5)
    phero_matrix.append([tau_0 for _ in range(num_mach)])
    '''for j in range(a):
        if i == j:
            phero_matrix[i].append(0)
        else:
            phero_matrix[i].append( tau_0 )'''"""

#print(str(len(phero_matrix))+" "+str(len(phero_matrix[0])))

# Main Loop of ACO
bestOptimum = 999999999999999
bestTour = []
makespan_iterasi = []
lastOptimum = bestOptimum
count_op = [0 for _ in range(num_job)]

for it in range(iterasiNo):
    colony_tour, colony_makespan = createColony(antNo, num_job, num_mach, pt, ms, phero_matrix, alpha, beta, q_threshold)

    bestMakespan = min(colony_makespan)
    if (it%20)==0:
        if (bestMakespan-lastOptimum) < (lastOptimum/num_job):
            q_threshold += (0.0005*sqrt(it))
        else:
            q_threshold -= (0.0005*sqrt(it))

    if bestMakespan < bestOptimum:
        bestOptimum = bestMakespan
        lastOptimum = bestOptimum
        bestTour = colony_tour[colony_makespan.index(bestMakespan)]
        print(phero_matrix)

    # Update Pheromones
    phero_matrix = updatePheromone(phero_matrix, num_job, num_mach, colony_tour, colony_makespan)

    # Evaporation
    for i in range(num_job):
        for j in range(num_mach):
            phero_matrix[i][j] *= (1 - rho)
            phero_matrix[i][j] = round(phero_matrix[i][j], 5)

    """if (it==(iterasiNo-1)):
        print(phero_matrix)"""
    makespan_iterasi.append(colony_makespan[0])

'''----------result----------'''
print("Optimal tour",bestTour)
print("Optimal value:%f"%bestOptimum)
print('the elapsed time:%s'% (time.time() - start_time))
print(q_threshold)

import matplotlib.pyplot as plt
#%matplotlib inline
plt.plot([i for i in range(len(makespan_iterasi))],makespan_iterasi,'b')
plt.ylabel('makespan',fontsize=15)
plt.xlabel('iteration',fontsize=15)
plt.show()

'''--------plot gantt chart-------'''
import pandas as pd
import plotly.plotly as py
import plotly.figure_factory as ff
import datetime

m_keys = [j for j in range(num_mach)]
j_keys = [j for j in range(num_job)]
key_count = {key: 0 for key in j_keys}
j_count = {key: 0 for key in j_keys}
m_count = {key: 0 for key in m_keys}
j_record = {}
for i in bestTour:
    gen_t = int(pt[i][key_count[i]])
    gen_m = int(ms[i][key_count[i]]-1)
    j_count[i] = j_count[i] + gen_t
    m_count[gen_m] = m_count[gen_m] + gen_t

    if m_count[gen_m] < j_count[i]:
        m_count[gen_m] = j_count[i]
    elif m_count[gen_m] > j_count[i]:
        j_count[i] = m_count[gen_m]

    start_time = str(
        datetime.timedelta(seconds=j_count[i] - pt[i][key_count[i]]))  # convert seconds to hours, minutes and seconds
    end_time = str(datetime.timedelta(seconds=j_count[i]))

    j_record[(i, gen_m)] = [start_time, end_time]

    key_count[i] = key_count[i] + 1


df=[]
for m in m_keys:
    for j in j_keys:
        df.append(dict(Task='Machine %s'%(m), Start='2018-07-14 %s'%(str(j_record[(j,m)][0])), Finish='2018-07-14 %s'%(str(j_record[(j,m)][1])),Resource='Job %s'%(j+1)))
"""
fig = ff.create_gantt(df, index_col='Resource', show_colorbar=True, group_tasks=True, showgrid_x=True, title='Job shop Schedule')
py.plot(fig, filename='GA_job_shop_scheduling', world_readable=True)
"""
