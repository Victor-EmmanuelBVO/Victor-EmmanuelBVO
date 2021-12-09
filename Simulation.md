#import des modules nécessaires

import numpy as np                  #tableaux de données
import scipy.interpolate as spip    #interpolation
import matplotlib.pyplot as plt     #graphiques

#fonctions reprises de Path3d (programme fourni dans le cadre du projet)
def path_points(points, steps=None):
    """
    Calcule un chemin courbe à partir de points de passage.
    Les courbes sont des splines cubiques.
    
    Paramètres:
        points: array[3,N]
            N points de passage en 3 dimensions
        steps: int
            nombre de points à générer, par défaut 10 * N
    Retourne:
        XPath: array[3,steps]
            coordonnées des points sur le chemin en 3 dimensions
    """

    # paramètre: racine carrée de la corde accumulée:
    deltaTau = np.sqrt(np.sum(np.diff(points)**2, axis=0))
    tauPoints = np.hstack(((0), np.cumsum(deltaTau)))
    tauEnd = tauPoints[-1]

    # Interpoler avec des spline cubiques:
    spline = spip.splprep(points, u=tauPoints, s=0)[0]
    
    # Echantillonner à intervalles réguliers:
    if not steps: steps = 100*points.shape[1]
    tau = np.linspace(0, tauEnd, steps)
    XPath = np.array(spip.splev(tau, spline))
    
    return XPath

def path_vectors(XPath):
    """
    Calcule la distance curvilinéaire, le vecteur tangent et
    le vecteur de courbure le long du chemin.
    
    Paramètres:
        XPath: array[3,N]
            coordonnées de N points en 3 dimensions
    Retourne: (sPath, TPath, CPath)
        sPath: array[N]
            distance curvilinéaire des points
        TPath: array[3,N]
            vecteur tangent aux points (|T| = 1)
        CPath: array[3,N]
            vecteur de courbure aux points (|C| = courbure = 1/rayon)
    """    
    ds = np.sqrt(np.sum(np.diff(XPath)**2, axis=0))
    s = np.hstack(((0), np.cumsum(ds)))

    dX_ds = np.gradient(XPath, s, axis=1, edge_order=2)
    d2X_ds = np.gradient(dX_ds, s, axis=1, edge_order=2)
    
    return s, dX_ds, d2X_ds
    
def path(points, steps=None):
    """
    Calcule les éléments d'un chemin courbe à partir de points de passage.
    
    Paramètres:
        points: array[3,N]
            N points de passage en 3 dimensions
        steps: int
            nombre de points à générer, par défaut 10 * N
    Retourne: (sPath, TPath, CPath)
        XPath: array[3,steps]
            coordonnées des points sur le chemin en 3 dimensions
        sPath: array[N]
            distance curvilinéaire des points
        TPath: array[3,N]
            vecteur tangent aux points (|T| = 1)
        CPath: array[3,N]
            vecteur de courbure aux points (|C| = courbure = 1/rayon)
    """    
    XPath = path_points(points, steps)
    sPath, TPath, CPath = path_vectors(XPath)
    return sPath, XPath, TPath, CPath

def ainterp(x, xp, ap, **kwargs):
    """
    Interpolation en x sur plusieurs fonctions xp -> ap[k].
    
    Paramètres:
        x: float ou array[M]
            abscisses x_m où les fonctions sont évaluées
        xp: array[N]
            abscisses x_n des points des fonctions
        ap: array[K, N]
            ordonnées f_k(x_n) des points des fonctions
            
    Résultat:
        a: array[K] ou array[K, M]
            ordonnées évaluées f_k(x_m)
    """
    a = np.array([np.interp(x, xp, fp, **kwargs) for fp in ap])
    return a

def path_at(s, path, **kwargs):
    """
    Retourne les éléments en un point donné d'un chemin,
    par interpolation des données du chemin.
    
    Paramètres:
        s: float
            distance curviligne du point
        path: (sPath, XPath, TPath, CPath)
            éléments du chemin, tels que retournés par path().
            
    Retourne: X, T, C
        X: array[3]
            coordonnées du point
        T: array[3]
            vecteur tangent au point
        C: array[3]
            vecteur de courbure au point
    """   
    sPath, XPath, TPath, CPath = path
    X = ainterp(s, sPath, XPath, **kwargs)
    T = ainterp(s, sPath, TPath, **kwargs)
    C = ainterp(s, sPath, CPath, **kwargs)
    return X, T, C


#points de passage chargés du document Points_passage.txt
Points_passage = np.loadtxt('Points_passage.txt').T #.T pour prendre la transposée (plus facile pour les calculs)

# chemin et vecteurs via la fonction path
sPath, xyzPath, TPath, CPath = path(Points_passage)
#sPath = distance curviligne
#xyzPath = coordonnées des points sur le chemin en 3D
#TPath = vecteurs tangents
#CPath = vecteurs courbures

length = sPath[-1] #dernière élément de sPath donc la longueur totale

nbre_jalon= 109 #points jalons à afficher sur le graphique
sMarks = np.linspace(0, length, nbre_jalon) #Permet d'afficher chaque point jalon à égale distance


xyzMarks = np.array([np.interp(sMarks, sPath, uPath) for uPath in xyzPath]) #coordonnées XYZ de chaque point jalon
TMarks = np.array([np.interp(sMarks, sPath, uPath) for uPath in TPath])     #direction des vecteurs tangents
CMarks = np.array([np.interp(sMarks, sPath, uPath) for uPath in CPath])     #direction des vecteurs courbures

# graphique 3D : chemin et vecteurs
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_box_aspect(np.ptp(xyzPath, axis=1)) #ratio entre les axes XYZ
ax.plot(Points_passage[0],Points_passage[1],Points_passage[2],'bo', label='points') #affichage des points de passage
ax.plot(xyzPath[0],xyzPath[1],xyzPath[2],'k.', ms=0.5) #affichage des points générés via interpolation
ax.plot(xyzMarks[0],xyzMarks[1],xyzMarks[2],'r.', ms=2) #affichage des points jalons
scale = 0.005 #échelle pour réduire la taille des vecteurs
ax.quiver(xyzMarks[0],xyzMarks[1],xyzMarks[2], #affichage des vecteurs tangents
          scale*TMarks[0],scale*TMarks[1],scale*TMarks[2],
          color='r', linewidth=0.5, label='T')
ax.quiver(xyzMarks[0],xyzMarks[1],xyzMarks[2], #affichage des vecteurs courbures
          scale*CMarks[0],scale*CMarks[1],scale*CMarks[2],
          color='g', linewidth=0.5, label='C')
ax.legend() #affichage de la légende
plt.show()  #affichage du graphique final en 3D


#Simulation physique

#définitions des paramètres :

tEnd=20 #temps maximum de la simulation
dt=0.01 #temps entre chaque point
steps = int(tEnd/dt) #nombre maximum d'étapes
g = 9.81 #constante de gravité
b = 0.009 # écart des rails [m]
r = 0.0079375 # diamètre de la bille [m]
h = np.sqrt(r**2 - b**2/4) # hauteur du centre de la bille sur les rails [m]
e = 0.00058 # coefficient de frottement linéaire [m/(m/s)]
M = 1 + 2/5*r**2/h**2 # coefficient d'inertie [1]
tSim = np.zeros(steps+1) # temps: array[steps+1] * [s]
sSim = np.zeros(steps+1) # distance curviligne: array[steps+1] * [m]
VsSim = np.zeros(steps+1) # vitesse tangentielle: array[steps+1] * [m/s]
Gn = np.zeros(steps+1) # accélération normale: array[steps+1] * [m/s²]

#boucle de simulation
i = 0
while i < steps and sSim[i] < length: #tant que le nombre maximum d'étape n'est pas dépassé et que la distance
                                      #actuelle est inférieure à la distance totale, la simulation continue
    
    vect_X_T_C = path_at(sSim[i], (sPath, xyzPath, TPath, CPath))  #vecteur tangent et courbure via path_at
    
    vectg=np.array([0,0,-g]) #vecteur gravité
    v=VsSim[i]               #vitesse actuelle
    vectC=vect_X_T_C[2]      #vecteur courbure
    vectT=vect_X_T_C[1]      #vecteur tangent
    gs = (vectT * vectg)[2]  #formule physique
    Gn[i] = np.linalg.norm(v**2 * vectC - (vectg - gs * vectT)) #accélération normale
    As = (gs-e*v/h *Gn[i])/M #formule physique de As

    #calcul de la nouvelle vitesse, distance et du temps
    VsSim[i+1] = VsSim[i] + As * dt         
    sSim[i+1] = sSim[i] + VsSim[i+1] * dt
    tSim[i+1] = tSim[i] + dt    
    i = i+1
tSim = tSim[:i-1]  #réduction des listes au nombre d'étape effectuée
sSim = sSim[:i-1]  
VsSim = VsSim[:i-1]
Gn = Gn[:i-1]

zPath = path_points(Points_passage)[2]  #coordonnée Z des points de passage
zSim = np.interp(sSim, sPath, zPath) #coordonnée Z des points générés par interpolation

#graphiques de distance,vitesse et hauteur
plt.figure()
plt.subplot(311)
plt.plot(tSim, sSim, label='s') #distance en fonction du temps
plt.ylabel('s [m]')
plt.xlabel('t [s]')
plt.subplot(312)
plt.plot(tSim, VsSim, label='vs') #vitesse en fonction du temps
plt.ylabel('Vs [m/s]')
plt.xlabel('t [s]')
plt.subplot(313)
plt.plot(tSim, zSim, label='z') #hauteur en fonction du temps
plt.ylabel('z [m]')
plt.xlabel('t [s]')
plt.show() #affichage des 3 graphes

#graphiques d'énergie et d'accélération normale
EpSim = zSim*g          # énergie potentielle spécifique [m**2/s**2]
EkSim = M*0.5*VsSim**2  # énergie cinétique spécifique [m**2/s**2]

fig, ax1 = plt.subplots()
ax1.set_xlabel('t [s]') #axe du temps
ax1.set_ylabel('Energie/masse [J/kg]') #axe de l'énergie
ax1.plot(tSim, EpSim, 'b-', label='Epot/m') #Energie potentielle en fonction du temps
ax1.plot(tSim, EkSim, 'r-', label='Ecin/m') #Energie cinétique en fonction du temps
ax1.plot(tSim, EpSim+EkSim, 'k-', label='Energie/m') #Energie mécanique (=EpSim + EkSim) en fonction du temps
plt.legend(loc='upper left')
ax2 = ax1.twinx()
ax2.set_ylabel('Accélération normale [m/s²]') #axe de l'accélération  
ax2.plot(tSim, Gn, 'g--', label='Gn') #accélération normale en fonction du temps
fig.tight_layout()  
plt.legend(loc='upper right')
plt.show() #affichage du graphe final
