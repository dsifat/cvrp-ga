import math
import random
from operator import itemgetter
from copy import deepcopy
import numpy
import csv


depot = [0, 0, 0, 0]



ins=open("input1.txt","r+")
customers=[[int(n) for n in line.split()] for line in ins]

customerCount = 32
vehicleCount = 5
vehicleCapacity = 100
assigned = [-1] * customerCount

cluster_no = 0

bin_matrix = []
centroids = []
tot_demand = []
members = []
xy_members = []

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def sweep_cluster():
    global cluster_no, bin_matrix, centroids, tot_demand
    global members, xy_members, prev_members

    tot_demand = sum([c[3] for c in customers])
    cluster_no = int(math.ceil(float(tot_demand) / vehicleCapacity))
    print('Number of Vehicle', cluster_no)


    d_customers = sorted(customers, key=itemgetter(3), reverse=True)
    centroids, tot_demand, members, xy_members = [], [], [], []
    for i in range(cluster_no):
        centroids.append(d_customers[i][1:3])

        tot_demand.append(0)
        members.append([])
        xy_members.append([])

    bin_matrix = [[0] * cluster_no for i in range(len(customers))]

    converged = False
    while not converged:
        prev_matrix = deepcopy(bin_matrix)

        for i in range(len(customers)):
            edist = []
            if assigned[i] == -1:
                for k in range(cluster_no):
                    p1 = (customers[i][1], customers[i][2]) # x,y
                    p2 = (centroids[k][0], centroids[k][1])
                    edist.append((distance(p1, p2), k))

                edist = sorted(edist, key=itemgetter(0))

            closest_centroid = 0
            while assigned[i] == -1:  
                max_prior = (0, -1)   # value, index
                for n in range(len(customers)):
                    pc = customers[n]

                    if assigned[n] == -1:
                        c = edist[closest_centroid][1]
                        cen = centroids[c]     

                        p = distance((pc[1], pc[2]), cen) / pc[3]

                        if p > max_prior[0]:
                            max_prior = (p, n)  

                if max_prior[1] == -1:   
                    break

                hpc = max_prior[1]    
                c = edist[closest_centroid][1]

                if tot_demand[c] + customers[hpc][3] <= vehicleCapacity:
                    members[c].append(hpc)

                    xy = customers[hpc][0],customers[hpc][1], customers[hpc][2] 
                    xy_members[c].append(xy)

                    tot_demand[c] += customers[hpc][3]
                    assigned[hpc] = c
                    with open("c",'w') as f:
                        writer=csv.writer(f,delimiter=',')
                        writer.writerows(xy_members)
                    bin_matrix[hpc][c] = 1

                if assigned[i] == -1:
                    if closest_centroid < len(edist)-1:
                        closest_centroid += 1

                    else:
                        break

        for j in range(cluster_no):
            xj = sum([cn[1] for cn in xy_members[j]])
            yj = sum([cn[2] for cn in xy_members[j]])
            xj = float(xj) / len(xy_members[j])
            yj = float(yj) / len(xy_members[j])
            centroids[j] = (xj, yj)

        converged = numpy.array_equal(numpy.array(prev_matrix), numpy.array(bin_matrix))

def clustering():
    sweep_cluster()

def write_file(xy_members):
    for i in range(0,len(xy_members)):
        file=open("%s.txt" %i,"w+")
        for j in range(0,len(xy_members[i])):
            file.write(str(xy_members[i][j])+"\n")
def modify_file():
    for i in range(0,cluster_no):
        with open("%s.txt" %i,"r+") as file:
            with open("%s.csv" %i,"w+") as out_file:
                for line in file:
                    line=str(line)
                    line=line.replace("(","")
                    line=line.replace(")","")
                    out_file.write(line)
def main():
    idx = clustering()
    print('centroids', centroids)
    print('Clustered Nodes', members)
    print('Vehicle demands', tot_demand)
    print('Nodes Value',xy_members)
    write_file(xy_members)
    modify_file()

if __name__ == '__main__':
    main()