#IMPORTS
import csv;
import numpy as np;
from numpy.linalg import inv;
import scipy.cluster.vq as vq;
import math;
import random;
class dataSet():
	training={'x_matrix':[],'target_vector':[]}
	validation={'x_matrix':[],'target_vector':[]}
	testing={'x_matrix':[],'target_vector':[]}

def partition(x_matrix,target_vector):
	tem=dataSet()
	temp=[[],[],[]]
	for i in range(0,len(x_matrix)):
		t=x_matrix[i]
		t.append(target_vector[i])
		temp[int(target_vector[i])].append(t)
	t_set=[[],[],[]]
	for i in range(0,3):
		m=len(temp[i])
		for j in range(0,m):
			if(j<0.8*m):
				t_set[0].append(temp[i][j])
			elif(j<0.9*m):
				t_set[1].append(temp[i][j])
			else:
				t_set[2].append(temp[i][j])
	for i in range(0,3):
		random.shuffle(t_set[i])
		m=len(t_set[i])
		for j in range(0,m):
			if i==0:
				tem.training['target_vector'].append(t_set[i][j].pop())
				tem.training['x_matrix'].append(t_set[i][j])
			elif i==1:
				tem.validation['target_vector'].append(t_set[i][j].pop())
				tem.validation['x_matrix'].append(t_set[i][j])
			elif i==2:
				tem.testing['target_vector'].append(t_set[i][j].pop())
				tem.testing['x_matrix'].append(t_set[i][j])
	return tem

def getInverseVariance(x_matrix, mu_vector,n):
	var_array=[]
	for x in x_matrix[0]:
		var_array.append(0)
	for x_vector in x_matrix:
		var_array=np.add(var_array,np.power(np.subtract(x_vector,mu_vector),2))
	var_array=np.divide(var_array,n*len(x_matrix))
	var_matrix=[]
	for i in range(0,len(var_array)):
		temp=[]
		for j in range(0,len(var_array)):
			if(i==j):
				if(var_array[i]==0):
					temp.append(0.0000000000000001)
				else:	
					temp.append(var_array[i])
			else:
				temp.append(0)
		var_matrix.append(temp)
	return inv(np.array(var_matrix))

def get_inv_var_matrix(x_matrix, mu_matrix,n):
	inv_var_matrix=[]
	for mu_vector in mu_matrix:
		inv_var_matrix.append(getInverseVariance(x_matrix,mu_vector,n))
	return inv_var_matrix
def phi_xn(x_vector, mu_vector, inv_var_matrix):
	x_transpose=np.transpose(np.subtract(x_vector,mu_vector))
	temp=np.dot(inv_var_matrix,x_transpose)
	temp=np.dot(np.subtract(x_vector,mu_vector),temp)
	temp=(-0.5)*temp
	temp=math.exp(temp)
	return temp
def phi_x(x_vector, mu_matrix, inv_var_matrix_vector):
	result=[]
	for i in range(0,len(mu_matrix)+1):
		if i==0:
			result.append(1)
		else:
			x_transpose=np.transpose(np.subtract(x_vector,mu_matrix[i-1]))
			temp=np.dot(inv_var_matrix_vector[i-1],x_transpose)
			temp=np.dot(np.subtract(x_vector,mu_matrix[i-1]),temp)
			temp=(-0.5)*temp
			temp=math.exp(temp)
			result.append(temp)
	return result

def phi(x_matrix, mu_matrix, inv_var_matrix):
	phi_matrix=[]
	for x_vector in x_matrix:
		phi_matrix.append(phi_x(x_vector,mu_matrix,inv_var_matrix))
	return phi_matrix

def delta_w(ro, e_delta):
	return np.dot(-ro,e_delta)

def delta_e(e_d, e_w, lamb):
	return np.add(e_d, np.dot(lamb,e_w))

def delta_e_d(target, weight_vector, phi_x_n):
	phi_x_n_t=np.transpose(phi_x_n)
	temp=np.dot(weight_vector,phi_x_n_t)
	temp=np.subtract(target,temp)
	temp=np.dot(temp,-1)
	temp=np.dot(temp,phi_x_n_t)
	return temp
	
def find_w_star(phi_matrix, lamb, target_vector):
	phi_transpose=np.transpose(phi_matrix)
	temp=np.dot(phi_transpose,phi_matrix)
	temp=np.add(np.dot(lamb,np.identity(len(phi_transpose))),temp)
	temp=inv(temp)
	temp=np.dot(temp,np.dot(phi_transpose,target_vector))
	return temp

def closedFormTrain(x_matrix,target_vector,clusterNum,n,lamb):
	x=vq.kmeans(np.array(x_matrix),clusterNum-1)
	mu_matrix=x[0]
	sigma_matrix=get_inv_var_matrix(x_matrix, mu_matrix,n)
	phi_matrix=phi(x_matrix,mu_matrix,sigma_matrix)
	w_star = find_w_star(phi_matrix,lamb=0.1, target_vector=target_vector)
	return [w_star,mu_matrix,sigma_matrix]

def validationError(x_matrix,target_vector,w_star, mu_matrix, sigma_matrix):
	err=0
	for i in range(0,len(x_matrix)):
		err=err+(target_vector[i] - np.dot(w_star,np.transpose(phi_x(x_matrix[i],mu_matrix,sigma_matrix))))**2
	return math.sqrt(err/len(x_matrix))
def testError(x_matrix,target_vector,w_star, mu_matrix, sigma_matrix):
	err=0
	for i in range(0,len(x_matrix)):
		err=err+(target_vector[i] - np.dot(w_star,np.transpose(phi_x(x_matrix[i],mu_matrix,sigma_matrix))))**2
	return math.sqrt(err/len(x_matrix))

x_matrix=[]
target_vector=[]
with open('input.csv', 'rU') as csvfile:
	spamreader = csv.reader(csvfile)
	for row in spamreader:
		temp=[]
		for item in row:
			temp.append(float(item))
		x_matrix.append(temp)
with open('output.csv', 'rU') as csvfile:
	spamreader = csv.reader(csvfile)
	for row in spamreader:
		for item in row:
			target_vector.append(float(item))
part = partition(x_matrix,target_vector)
clusterNum=8
lamb=0
writer=open('synthetic_closed_output.csv','w')
writer.write('M,Lambda,ValidationERMS,TestingERMS\n')
while(lamb<1):
	writer.write(str(clusterNum)+','+str(lamb)+',')
	newTrainResult = closedFormTrain(part.training['x_matrix'],part.training['target_vector'], clusterNum=clusterNum,n=0.5,lamb=lamb)
	newErr=validationError(part.validation['x_matrix'],part.validation['target_vector'],newTrainResult[0],newTrainResult[1], newTrainResult[2])
	writer.write(str(newErr)+',')
	newErr=testError(part.testing['x_matrix'],part.testing['target_vector'],newTrainResult[0],newTrainResult[1], newTrainResult[2])
	writer.write(str(newErr)+'\n')
	lamb=lamb+0.1