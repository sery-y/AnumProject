import math
import numpy as np
import matplotlib.pyplot as plt
from matrices import is_positive_definite, is_symmetric, saisir_matrice, saisir_vecteur

def pivot_partiel(A, b, k):
    n = len(A)

    max_index = k
    max_val = abs(A[k][k])

    for i in range(k+1, n):
        if abs(A[i][k]) > max_val:
            max_val = abs(A[i][k])
            max_index = i

    if max_val == 0:
        return False

    if max_index != k:
        A[[k, max_index]] = A[[max_index, k]]
        b[k], b[max_index] = b[max_index], b[k]

    return True

def pivot_total(A,b ,k,perm):
    n = len(A)

    max_val = abs(A[k][k])
    max_i, max_j = k, k

    for i in range(k, n):
        for j in range(k, n):
            if abs(A[i][j]) > max_val:
                max_val = abs(A[i][j])
                max_i, max_j = i, j

    if max_val == 0:
        return False

    # swap lignes
    if max_i != k:
        A[[k, max_i]] = A[[max_i, k]]
        b[k], b[max_i] = b[max_i], b[k]

    # swap colonnes
    if max_j != k:
        A[:, [k, max_j]] = A[:, [max_j, k]]
        perm[k], perm[max_j] = perm[max_j], perm[k]

    return True

def gauss(A, b):
    n = len(A)
    perm = list(range(n))

    A = A.astype(float)
    b = b.astype(float)

    historiques_A = [A.copy()]
    historiques_b = [b.copy()]

    for k in range(n-1):

        # Gestion pivot nul
        if A[k][k] == 0:
           print("Pivot nul détecté à l'étape", k)
           print("Choisir méthode de pivot :")
           print("1 - Partiel")
           print("2 - Total")

           choix = input("Votre choix : ")

           if choix == "1":
            ok = pivot_partiel(A, b, k)
           else:
            ok = pivot_total(A, b, k, perm)
           if not ok:
                print("Aucune méthode de pivotage n'a pu résoudre le problème.")
                return historiques_A, historiques_b, None


        #  Élimination
        for i in range(k+1, n):
            m = A[i][k] / A[k][k]
            A[i, k:] -= m * A[k, k:]
            b[i] -= m * b[k]

        historiques_A.append(A.copy())
        historiques_b.append(b.copy())

    #  Substitution arrière
    x = np.zeros(n)

    for i in range(n-1, -1, -1):
        s = sum(A[i][j] * x[j] for j in range(i+1, n))
        x[i] = (b[i] - s) / A[i][i]

    #  Réorganisation si pivot total
    x_final = np.zeros(n)
    for i in range(n):
        x_final[perm[i]] = x[i]

    return historiques_A, historiques_b, x_final

def gauss_console():
    n = int(input("Entrer la taille de la matrice A : "))
    A = saisir_matrice(n)
    b = saisir_vecteur(n, "b")
    histA, histB, solution = gauss(A, b)
    print("\n=== HISTORIQUE DES ÉTAPES ===")
    for k in range(len(histA)):
        print(f"\n--- Étape {k} ---")
        print("Matrice A :")
        print(histA[k])
        print("Vecteur b :")
        print(histB[k])
    
    print("\n=== SOLUTION FINALE ===")
    print("X =", solution)
    
    return histA, histB, solution
    
def afficher_matrices(L, U, k):
    print(f"Étape k = {k}")
    print("Matrice L :")
    print(L)
    print("\nMatrice U :")
    print(U)

def LU(A, b):
    n = len(A)

    A = A.astype(float)
    L = np.eye(n)
    perm = list(range(n))
    U = A.copy()

    for k in range(n):

        # ===== PIVOT =====
        if A[k][k] == 0:
            if A[k][k] == 0:
              print("Pivot nul détecté à l'étape", k)
              print("Choisir méthode de pivot :")
              print("1 - Partiel")
              print("2 - Total")

              choix = input("Votre choix : ")
              
              if choix == "1":
               ok = pivot_partiel(A, b, k)
              else:
               ok = pivot_total(A, b, k, perm)

              if not ok:
                print("Pivot impossible")
                return None, None, None

        # ===== ELIMINATION (Gauss) =====
        for i in range(k+1, n):
            m = A[i][k] / A[k][k]
            L[i][k] = m
            A[i, k:] = A[i, k:] - m * A[k, k:]

        afficher_matrices(L, U, k)
    return L, U, perm

def resoudre_LU(L, U, b):
    n = len(b)

    # Ly = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - sum(L[i][j] * y[j] for j in range(i))

    # Ux = y
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i+1, n))) / U[i][i]

    return x

def LU_console():
    n = int(input("Entrer la taille de la matrice A : "))
    A = saisir_matrice(n)
    b = saisir_vecteur(n, "b")

    L, U, perm = LU(A, b)
    x = resoudre_LU(L, U, b)

    print("\n=== SOLUTION ===")
    print("X =", x)

    return L, U, x

def choleski(A):
    n = len(A)
    L = np.zeros((n, n))
    if not is_symmetric(A) or not is_positive_definite(A):
        print("La matrice n'est pas symétrique définie positive.")
        return None
    
    
    for i in range(n):
        for j in range(i+1):
            somme =0
            # somme Σ L[i][k] * L[j][k]
            for k in range(j):
                somme += L[i][k] * L[j][k]
            
            if i == j:
                L[i][j] = math.sqrt(A[i][i] - somme)
            else:
                L[i][j] = (A[i][j] - somme) / L[j][j]
        print(f"Étape i = {i}")
        print("Matrice L :")
        print(L)
    return L

#L*Y=b puis L^T * X =Y  A= L * L^T (A = S * S^T)
def resoudre_choleski(L,b):
    n = len(b)
    #L*Y=b
    y = np.zeros(n)
    for i in range(n):
        somme = 0
        for j in range(i):
            somme += L[i][j] * y[j]   
        y[i] = (b[i] - somme) / L[i][i]
    #L^T X = Y
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        somme = 0
        for j in range(i+1, n):
            somme += L[j][i] * x[j]  
        x[i] = (y[i] - somme) / L[i][i]
    return x

def choleski_console():
    n = int(input("Entrer la taille de la matrice A : "))
    A = saisir_matrice(n)
    b = saisir_vecteur(n, "b")
    L=choleski(A)
    if L is not None:
        x = resoudre_choleski(L, b)
        print("\n=== SOLUTION ===")
        print("X =", x)
        return L, x
    
def recommander_methodes_directes(A):
    
    print("\n--- Analyse pour méthodes directes ---")

    if np.linalg.det(A) == 0:
        print("Matrice non inversible => aucune méthode directe possible")
        return

    if is_symmetric(A) and is_positive_definite(A):
        print("Cholesky recommandé (rapide et stable)")
        return

    print("LU recommandé (cas général)")

    print("Gauss avec pivot recommandé")
    print("   - Pivot partiel : standard")
    print("   - Pivot total : si instabilité")

def get_direct_methodes():
    return {
        "gauss": gauss_console,
        "lu": LU_console,
        "cholesky": choleski_console
    }