#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.spatial import ConvexHull

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",   # Computer Modern
})

cores = [     "lime",     "blue",     "green",     "black",     "yellow",     "purple",
    "brown",     "pink",     "gray",     "olive",     "cyan",     "magenta",
    "gold",     "teal",     "navy",     "maroon",     "turquoise",     "indigo",
    "coral",     "darkgreen"
]
TAMFONTLABEL=18
TAMFONTLEGEND=12
TAMFONTTITLE=18
TAMFONTINSET=18
ESPESSURA_LINHA=1.5

plt.figure(figsize=(6, 6))  # Figura quadrada



with open('data_3_6_9_12.pkl', 'rb') as f:
    points_poly, pontos_grounds_states = pickle.load(f)    

hull = ConvexHull(points_poly)
hull_points = points_poly[hull.vertices]
points_line={}    



for key in pontos_grounds_states.keys():
        cor_linha=cores.pop(1)
        points_line = np.array(pontos_grounds_states[key])  
        plt.plot(points_line[:, 0], points_line[:, 1], '-', label= r"$\mathcal{N}$ =" + str(key) , 
                 color=cor_linha, linewidth=ESPESSURA_LINHA)
        
# Pontos não ordenados de η3_z1_z2_x1
plt.scatter(points_poly[:, 0], points_poly[:, 1], marker='s', 
             color='black', s=20)

# Polígono da convex hull
plt.fill(hull_points[:, 0], hull_points[:, 1], alpha=0.3)

# Labels e formatação

plt.xlabel(r"$\langle Z_1 Z_2 \rangle$", fontsize=TAMFONTLABEL)
plt.ylabel(r"$\langle X_1 \rangle$",fontsize=TAMFONTLABEL)
plt.title("$RoM_\mathcal{M}[\psi_\mathrm{gs}(g)]$",fontsize=TAMFONTTITLE)
plt.legend(fontsize=TAMFONTLEGEND)
# plt.grid(True)
plt.axis('equal')
plt.xlim(0.01, 1.08)
plt.ylim(-0.01, 1.08)

ax = plt.gca()

ax.text(
    0.25, 1.00,                      
    r"$g\to \infty$",    
    transform=ax.transAxes,          
    ha='right',                      
    va='top',                        
    fontsize=TAMFONTINSET
)
ax.text(
    0.86, 0.10,                      
    r"$g = 0$",    
    transform=ax.transAxes,          
    ha='right',                      
    va='top',                        
    fontsize=TAMFONTINSET
)
plt.savefig("figure_3_6_9_12.png", dpi=300, bbox_inches="tight")
plt.show()


# In[ ]:





# In[ ]:




