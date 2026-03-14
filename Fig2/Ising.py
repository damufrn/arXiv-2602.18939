#!/usr/bin/env python
# coding: utf-8

# In[1]:


#any  qubits
import  qiskit.quantum_info as qi
import numpy as np
import pickle
import matplotlib.pyplot as plt
import qiskit.quantum_info as qi
from itertools import product
import cvxpy as cp
import re
import time

LISTA_DE_Ns=[3,6,9,12]


η5_z1_z2_x1=[]  ## It works with any number of qubits. The reduction gives the same polytope.
with open('polytope_sigma_z1_sigma_z2_x1_periodic5.pkl', 'rb') as f:
    η5_z1_z2_x1 = pickle.load(f)
    print("Politopo Reduzido 5 qubits lido de arquivo.")

def has_Z_at_extremes(s):
   return s[0] == 'Z' and s[-1] == 'Z'

def has_one_X(s):
    return bool(re.search(r'X', s))
    
def has_two_Z_neighbors(s):
    return bool(re.search(r'ZZ', s))
  


def reduced_tomographic_representation_z1_z2_x1(ρ): #Versão 17 de maio - Para qualquer número de qubits
    NUM_TERMOS_PAULI=2
    lista_valores_esperados=[]
    paulis=['I','X','Y','Z']
    qtd_qubits=int(np.log2(ρ.dim))
    M=np.zeros([ρ.dim,ρ.dim])
    
    s = 'I' * (qtd_qubits-2)
    pauli_matrix='ZZ' + s
    pauli_matrix=qi.Pauli(pauli_matrix).to_matrix()
    
    lista_valores_esperados.append(np.real(np.trace(pauli_matrix @ ρ.data)))
    
    s = 'I' * (qtd_qubits-1)
    pauli_matrix='X'+s
    pauli_matrix=qi.Pauli(pauli_matrix).to_matrix()
    lista_valores_esperados.append(np.real(np.trace(pauli_matrix @ ρ.data)))
    
    return(lista_valores_esperados)


def ROMreduced_z1_z2_x1(ρ):
    
    if not isinstance(ρ, list):        
        ρ = reduced_tomographic_representation_z1_z2_x1(ρ)
    ρ = np.array(ρ) 
    
    σ_dict = {2: η5_z1_z2_x1}
    σ = σ_dict[len(ρ)]
    σ = np.array(σ)  # Ensure σ is a NumPy array
    
    n = len(σ)
    x = cp.Variable(n)
    
    objective = cp.Minimize(cp.norm(x, 1))
    
    if σ.ndim == 2:
        constraints = [ρ == σ.T @ x]  # σ.T @ x = sum(x_i * σ_i)
    else:
        constraints = [ρ == cp.sum(cp.multiply(x, σ))]
    constraints.append(cp.sum(x) == 1)
    prob = cp.Problem(objective, constraints)
    sol = prob.solve()
    
    return sol
#===============



#=======================
def has_two_Z_neighbors_periodic(s):
    #Esta rotina espera apenas dois Z. Qualquer vizinho ZZ ou condição períodica ela avisa.
    return bool(re.search(r'ZZ|^Z.*Z$', s))




# In[2]:


def is_hermitian(matrix):
    conjugate_transpose = np.conj(matrix).T    
    return np.allclose(matrix, conjugate_transpose)

def gerar_vizinhos(N,vizinho=1):
    lista=[]
    
    for i in range(N):
        s='I' * N
        s = s[:i] + 'Z' + s[i+1:]
        j=(i+vizinho)%N
        s = s[:j] + 'Z' + s[j+1:]
        
        lista.append(s)
    return(lista)

def gerar_transversos(N):
    lista=[]
    
    for i in range(N):
        s='I' * N
        s = s[:i] + 'X' + s[i+1:]
        lista.append(s)
    return(lista)

first_neighbours= {}
transverse = {}
for N in LISTA_DE_Ns:
    first  =  np.zeros((2**N,2**N))
    trans  =  np.zeros((2**N,2**N))
    

    for palavra in gerar_vizinhos(N,1):
        first = first + qi.Pauli(palavra).to_matrix()
    
    for palavra in gerar_transversos(N):
        trans = trans + qi.Pauli(palavra).to_matrix()

    first_neighbours[N]=first
    transverse[N]=trans
     
def H(N,g):
    return(-first_neighbours[N] -g*transverse[N])
           



def printResultado(hamil):
    val,vec=np.linalg.eigh(hamil) #tem que usar eigh - Hamiltonianos hermitianos e o resultado sai em ordem crescente.
    tam=len(val)
    for i in range(tam):
        print(val[i])
        display(qi.Statevector(vec[:,i]).draw('latex')) #autovetores estão nas colunas!

def groundState(hamil):
    val,vec=np.linalg.eigh(hamil)
    return(qi.Statevector(vec[:,0])) #resultado em ordem crescente vindo do eigh. Não usar eig aqui.
    


# In[3]:


#eigenvalues, eigenvectors = np.linalg.eigh(H(0.5))
#printResultado(eigenvalues,eigenvectors)


# In[4]:


#pa
pontos_grounds_states={}
g_INI=0
g_FIN=30
delta_g=0.1
for N in LISTA_DE_Ns:
    pontos_grounds_states[N]=[]
    tempo_inicial=time.time()
    TODOS_Gs=np.arange(g_INI,g_FIN,delta_g)
    num_Gs=len(TODOS_Gs)
    calc_Gs=0
    for g in TODOS_Gs:
        #3 qubits
        Hamiltonian=H(N,g)
        calc_Gs=calc_Gs+1
        if is_hermitian(Hamiltonian):          
            ρ_groundstate=qi.DensityMatrix( groundState(Hamiltonian)  )        
            tempo_atual=time.time()
            print('Previsão '+str(N)+' qubits: ' + str((num_Gs-calc_Gs)*(tempo_atual - tempo_inicial)/calc_Gs),' s')
            red =ROMreduced_z1_z2_x1(ρ_groundstate)
            pontos_grounds_states[N].append(reduced_tomographic_representation_z1_z2_x1(ρ_groundstate))
        else:
            print('Um hamiltoniano não era hermitiano!')
    
with open('data_3_6_9_12.pkl', 'wb') as f:
    pickle.dump((np.array(η5_z1_z2_x1),pontos_grounds_states), f)   
        
    




# In[ ]:





# In[17]:


