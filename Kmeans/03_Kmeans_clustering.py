#!/usr/bin/env python
# coding: utf-8

# In[39]:


# Sanket Badjate

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
get_ipython().run_line_magic('matplotlib', 'inline')

class UsingLyb:
    def __init__(self):
        print('____________Using Lyb Function_____________')
    
    def working(self):
        
        df=pd.DataFrame({'X':[0.1,0.15,0.08,0.16,0.2,0.25,0.24,0.3],
                     'y':[0.6,0.71,0.9,0.85,0.3,0.5,0.1,0.2]})
        f1 = df['X'].values
        f2 = df['y'].values
        X = np.array(list(zip(f1, f2)))
        print(X)

        #centroid points
        C_x=np.array([0.1,0.3])
        C_y=np.array([0.6,0.2])
        centroids=C_x,C_y

        #plot the given points
        colmap = {1: 'r', 2: 'b'}
        plt.scatter(f1, f2, color='k')
        plt.show()

        #for i in centroids():
        plt.scatter(C_x[0],C_y[0], color=colmap[1])
        plt.scatter(C_x[1],C_y[1], color=colmap[2])
        plt.show()

        C = np.array(list((C_x, C_y)), dtype=np.float32)
        print (C)

        #plot given elements with centroid elements
        plt.scatter(f1, f2, c='#050505')
        plt.scatter(C_x[0], C_y[0], marker='*', s=200, c='r')
        plt.scatter(C_x[1], C_y[1], marker='^', s=200, c='b')
        plt.show()


        #import KMeans class and create object of it
        from sklearn.cluster import KMeans
        model=KMeans(n_clusters=2,random_state=0)
        model.fit(X)
        labels=model.labels_
        print(labels)
        print(model.predict([[0.25,0.5]]))

        #using labels find population around centroid
        count=0
        for i in range(len(labels)):
            if (labels[i]==1):
                count=count+1

        print('No of population around cluster 2:',count-1)

        #Find new centroids
        new_centroids = model.cluster_centers_

        print('Previous value of m1 and m2 is:')
        print('M1==',centroids[0])
        print('M1==',centroids[1])

        print('updated value of m1 and m2 is:')
        print('M1==',new_centroids[0])
        print('M1==',new_centroids[1])

   
class fromScratch:
    
    def __init__(self):
        
        print('____________From Scratch_____________')
        self.x=np.array([0.1,0.15,0.08,0.16,0.2,0.25,0.24,0.3])
        self.y=np.array([0.6,0.71,0.9,0.85,0.3,0.5,0.1,0.2])
        plt.plot(self.x,self.y,"o")
        plt.show()
    
    def eucledian_distance(self,x1,y1,x2,y2):
        return math.sqrt((self.x1-self.x2)**2+(self.y1-self.y2)**2)

    def manhattan_distance(self,x1,y1,x2,y2):
        return math.fabs(x1-x2)+math.fabs(y1-y2)
    
    def returnCluster(self,m1,m2,x_co,y_co):
    #if we use manhattan_distance then clusters are classified more correctly..
        distance1=self.manhattan_distance(m1[0],m1[1],x_co,y_co)
    
        distance2=self.manhattan_distance(m2[0],m2[1],x_co,y_co)
    
        if(distance1<distance2):
            return 1
        else:
            return 2
    def working(self):
        #initial centroids for cluster1 nd cluster 2
        m1=[0.1,0.6]
        m2=[0.3,0.2]
        #difference and iteration is for controlling iteration
        difference = math.inf
        threshold=0.02
        iteration=0;
        while difference>threshold: #use any one condition #iteration one is easy
            print("Iteration ",iteration, " : m1=",m1, " m2=",m2)
            cluster1=[]
            cluster2=[]

            #step1 assign all points to nearest cluster
            for i in range(0,np.size(self.x)):
                clusterNumber=self.returnCluster(m1,m2,self.x[i],self.y[i])
                point=[self.x[i],self.y[i]]
                if clusterNumber==1:
                    cluster1.append(point)
                else:
                    cluster2.append(point)

            print("cluster 1", cluster1,"\nCLuster 2: ", cluster2)

            #step 2: Calculating new centriod for cluster1
            m1_old=m1
            m1=[]
            m1=np.mean(cluster1, axis=0) #axis=0 means columnwise 

            #calculating centroid for cluster2
            m2_old=m2
            m2=[];
            m2=np.mean(cluster2,axis=0)
            print("m1 = ",m1," m2=",m2)

            #adjusting diffrences of adjustment between m1 nd m1_old
            xAvg=0.0
            yAvg=0.0
            xAvg=math.fabs(m1[0]-m1_old[0])+math.fabs(m2[0]-m2_old[0])
            xAvg=xAvg/2

            yAvg=math.fabs(m1[1]-m1_old[1])+math.fabs(m2[1]-m2_old[1])
            yAvg=yAvg/2

            if(xAvg>yAvg):
                difference=xAvg
            else:
                difference=yAvg

            print("Difference  : ", difference)
            iteration+=1;
            print("")
            #final Output
        print("Cluster 1 centroid : m1 = ",m1)
        print("CLuster 1 points: ", cluster1)
        print("Cluster 2 centroid : m2 = ",m2)
        print("CLuster 2 points: ", cluster2)

        clust1=np.array(cluster1)
        clust2=np.array(cluster2)

        #cluster 1 points
        plt.plot(clust1[:,0],clust1[:,1],"o")

        #cluster2 points
        plt.plot(clust2[:,0], clust2[:,1],"*")

        #centroids
        plt.plot([m1[0],m2[0]],[m1[1],m2[1]],"^")
        plt.show()
        
        #same code
        plt.scatter(clust1[:,0],clust1[:,1])
        plt.scatter(clust2[:,0],clust2[:,1])
        plt.scatter([m1[0],m2[0]],[m1[1],m2[1]],marker="*")
        plt.show()
        
    
    
    
    
u=UsingLyb()
u.working()

f=fromScratch() 
f.working()
    


# In[ ]:




