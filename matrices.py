import math
from matplotlib.pylab import diff, eig
import numpy as np
import matplotlib.pyplot as plt


def valeurs_propres(A):
    A = np.array(A, dtype=float)
    vp = eig(A)[0]
    print("Valeurs propres :")
    print(vp)
    return vp

def rayon_spectral(A):
    vp = valeurs_propres(A)   
    rayon = np.max(np.abs(vp))
    print("Rayon spectral :", rayon)
    return rayon

def saisir_matrice(n):
    A = []
    print("Entrer la matrice A :")
    for i in range(n):
        ligne = list(map(float, input(f"Ligne {i+1} : ").split()))
        A.append(ligne)
    return np.array(A)

def saisir_vecteur(n, nom):
    print(f"Entrer le vecteur {nom} :")
    return np.array(list(map(float, input().split())))

def is_DDS(A):
    n = len(A)

    for i in range(n):
        somme = 0
        for j in range(n):
            if j != i:
                somme += abs(A[i][j])

        if abs(A[i][i]) <= somme:
            return False

    return True

def is_symmetric(A):
    n = len(A)
    for i in range(n):
        for j in range(n):
            if A[i][j] != A[j][i]:
                return False
    return True

def transpose(A):
    n = len(A)
    T = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            T[i][j] = A[j][i]

    return T

def is_positive_definite(A):
    return np.all(np.linalg.eigvals(A) > 0)

def norme_1(matrice):
    return np.max(np.sum(np.abs(matrice), axis=0))

def norme_infini(matrice):
    return np.max(np.sum(np.abs(matrice), axis=1))

def norme_2(matrice):
    return np.sqrt(np.sum(np.square(matrice)))

def norme_p(matrice, p):
    return np.power(np.sum(np.abs(matrice) ** p), 1 / p)

def determinant(A):
    A = np.array(A, dtype=float)
    return np.linalg.det(A)

def conditionnement(A, norme="1", p=2):
    A = np.array(A, dtype=float)
    
    if determinant(A) == 0:
        print("La matrice n'est pas inversible.")
        return None
    
    A_inv = np.linalg.inv(A)

    if norme == "1":
        return norme_1(A) * norme_1(A_inv)
    elif norme == "inf":
        return norme_infini(A) * norme_infini(A_inv)
    elif norme == "2":
        return norme_2(A) * norme_2(A_inv)
    elif norme == "p":
        return norme_p(A, p) * norme_p(A_inv, p)
    else:
        print("Norme non reconnue")
        return None
    
def choisir_norme(M, type_norme, p=None):
    if type_norme == "1":
        return norme_1(M)
    elif type_norme == "inf":
        return norme_infini(M)
    elif type_norme == "2":
        return norme_2(M)
    elif type_norme == "p":
        if p is None:
            raise ValueError("Vous devez fournir p pour la norme p")
        return norme_p(M, p)
    else:
        raise ValueError("Norme inconnue")
    
def get_operations():
    return ["valeurs_propres", "rayon_spectral", "norme_1", "norme_infini", 
            "norme_2", "determinant", "conditionnement", "is_DDS", 
            "is_symmetric", "is_positive_definite"]