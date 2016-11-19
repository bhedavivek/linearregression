def partition(x_matrix,target_vector):
	tem=dataSet()
	temp_x=[[],[],[]]
	temp_t=[[],[],[]]
	for i in range(0,len(x_matrix)):
		temp_x[int(target_vector[i])].append(x_matrix[i])
		temp_t[int(target_vector[i])].append(target_vector[i])
	for i in range(0,3):
		random.shuffle(temp_x[i])
		random.shuffle(temp_t[i])
		m=len(temp_x[2])
		for j in range(0,m):
			if(j<0.8*m):
				tem.training['x_matrix'].append(temp_x[2].pop())
				tem.training['target_vector'].append(temp_t[2].pop())
				tem.training['x_matrix'].append(temp_x[1].pop())
				tem.training['target_vector'].append(temp_t[1].pop())
				tem.training['x_matrix'].append(temp_x[0].pop())
				tem.training['target_vector'].append(temp_t[0].pop())
			elif(j<0.9*m):
				tem.validation['x_matrix'].append(temp_x[2].pop())
				tem.validation['target_vector'].append(temp_t[2].pop())
			else:
				tem.testing['x_matrix'].append(temp_x[2].pop())
				tem.testing['target_vector'].append(temp_t[2].pop())
		for i in range(0,2):
			m=len(temp_x[i])
			for j in range(0,m):
				if(j<0.5*m):
					tem.validation['x_matrix'].append(temp_x[i][j])
					tem.validation['target_vector'].append(temp_t[i][j])
				else:
					tem.testing['x_matrix'].append(temp_x[i][j])
					tem.testing['target_vector'].append(temp_t[i][j])
	return tem



def partition(x_matrix,target_vector):
	tem=dataSet()
	temp_x=[[],[],[]]
	temp_t=[[],[],[]]
	for i in range(0,len(x_matrix)):
		temp_x[int(target_vector[i])].append(x_matrix[i])
		temp_t[int(target_vector[i])].append(target_vector[i])
	for i in range(0,3):
		random.shuffle(temp_x[i])
		random.shuffle(temp_t[i])
	for i in range(0,3):
		m=len(temp_x[i])
		for j in range(0,m):
			if(j<0.8*m):
				tem.training['x_matrix'].append(temp_x[i][j])
				tem.training['target_vector'].append(temp_t[i][j])
			elif(j<0.9*m):
				tem.validation['x_matrix'].append(temp_x[i][j])
				tem.validation['target_vector'].append(temp_t[i][j])
			else:
				tem.testing['x_matrix'].append(temp_x[i][j])
				tem.testing['target_vector'].append(temp_t[i][j])
	return tem



def closedFormTrain(x_matrix,target_vector,clusterNum,n,lamb):
	temp_x=[]
	for i in range(0,len(target_vector)):
		temp=[]
		for each in x_matrix[i]:
			temp.append(each)
		temp.append(target_vector[i])
		temp_x.append(temp)
	x=KMeans(n_clusters=clusterNum-1, random_state=0, init="k-means++", n_jobs=-1).fit(temp_x)
	mu_matrix=np.delete(x.cluster_centers_, len(x.cluster_centers_[0])-1,1)
	sigma_matrix=get_inv_var_matrix(x_matrix, mu_matrix,n)
	phi_matrix=phi(x_matrix,mu_matrix,sigma_matrix)
	w_star = find_w_star(phi_matrix,lamb=0.1, target_vector=target_vector)
	return [w_star,mu_matrix,sigma_matrix]

def closedFormTrain(x_matrix,target_vector,clusterNum,n,lamb):
	x=KMeans(n_clusters=clusterNum-1, random_state=0, init="k-means++", n_jobs=-1).fit(x_matrix)
	mu_matrix=x.cluster_centers_
	sigma_matrix=get_inv_var_matrix(x_matrix, mu_matrix,n)
	phi_matrix=phi(x_matrix,mu_matrix,sigma_matrix)
	w_star = find_w_star(phi_matrix,lamb=0.1, target_vector=target_vector)
	return [w_star,mu_matrix,sigma_matrix]


def stochasticTrain(x_matrix,target_vector,clusterNum,n,lamb,eta):
	w_star=[]
	w_temp=[]
	x=KMeans(n_clusters=clusterNum-1, random_state=0, init="k-means++", n_jobs=-1).fit(x_matrix)
	mu_matrix=x.cluster_centers_
	sigma_matrix=get_inv_var_matrix(x_matrix, mu_matrix,n)
	for i in range(0,clusterNum):
		w_temp.append(0.5)
	for i in range(0,len(x_matrix)):
		phi_vector=phi_x(x_matrix[i],mu_matrix,sigma_matrix)
		e_d=delta_e_d(target_vector[i],w_temp,phi_vector)
		w_d=delta_w(eta,delta_e(e_d,w_temp,lamb))
		w_temp=np.add(w_temp,w_d)
	return [w_temp,mu_matrix,sigma_matrix]


def stochasticTrain(x_matrix,target_vector,clusterNum,n,lamb,eta):
	temp=[]
	w_temp=[]
	x=vq.kmeans2(np.array(x_matrix),clusterNum-1)
	mu_matrix=x[0]
	sigma_matrix=get_inv_var_matrix(x_matrix, mu_matrix,n)
	for i in range(0,clusterNum):
		w_temp.append(random.random())
	w_star=w_temp
	prevErr=1
	prevMax=float('-inf')
	currentMax=float('-inf')
	for i in range(0,len(x_matrix)):
		phi_vector=phi_x(x_matrix[i],mu_matrix,sigma_matrix)
		e_d=delta_e_d(target_vector[i],w_temp,phi_vector)
		d_e=delta_e(e_d,w_temp,lamb)
		temp.append(d_e)
		if(i%100==0):
			temp=np.mean(temp,axis=0)
			w_d=delta_w(eta,temp)
			w_temp=np.add(w_temp,w_d)
			currentMax=temp[0]
			for each in temp:
				if(each>currentMax):
					currentMax=each
			if(prevMax>currentMax):
				#eta=eta*0.9
				prevMax=currentMax
				w_star=w_temp
			if(math.fabs(currentMax)<0.0001):
				print 'Broke'
				break;
			temp=[]
		w_star=w_temp
	return [w_star,mu_matrix,sigma_matrix]

def stochasticTrain(x_matrix,target_vector,clusterNum,n,lamb,eta):
	w_temp=[]
	temp=[]
	x=vq.kmeans2(np.array(x_matrix),clusterNum-1)
	mu_matrix=x[0]
	sigma_matrix=get_inv_var_matrix(x_matrix, mu_matrix,n)
	for i in range(0,clusterNum):
		w_temp.append(random.random())
	while(len(x_matrix)!=0):
		phi_vector=phi_x(x_matrix.pop(random.random()*len(x_matrix)),mu_matrix,sigma_matrix)
		e_d=delta_e_d(target_vector[i],w_temp,phi_vector)
		d_e=delta_e(e_d,w_temp,lamb)
		temp.append(d_e)
		if(i%100==0):
			temp=np.mean(temp,axis=0)
			print temp
			w_d=delta_w(eta,temp)
			max_w=w_d[0]
			w_temp=np.add(w_temp,w_d)
			temp=[]
			for each in w_d:
				if(max_w<each):
					max_w=each
			if(max_w<0.001):
				break
	return [w_temp,mu_matrix,sigma_matrix]
