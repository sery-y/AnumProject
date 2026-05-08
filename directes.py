import math
from tkinter import messagebox
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

    steps = []

    for k in range(n - 1):

        steps.append(f"\n===== Étape {k+1} =====")

        # ===== Gestion pivot nul =====
        if abs(A[k][k]) < 1e-14:

            steps.append(
                f"Pivot nul détecté en position ({k+1},{k+1})"
            )

            ret = choix_utilisateur_gauss(k + 1, A, b, k, perm)

            if ret == False:
                steps.append("Pivot impossible.")
                return historiques_A, historiques_b, None, steps

            steps.append("Pivot appliqué avec succès.")

        # ===== Élimination =====
        for i in range(k + 1, n):

            m = A[i][k] / A[k][k]

            steps.append(
                f"m_{i+1}{k+1} = "
                f"{A[i][k]:.4f} / {A[k][k]:.4f} = {m:.4f}"
            )

            A[i, k:] -= m * A[k, k:]
            b[i] -= m * b[k]

            steps.append(
                f"L{i+1} ← L{i+1} - ({m:.4f}) × L{k+1}"
            )

            steps.append(
                f"Nouvelle ligne {i+1} : {A[i]}"
            )

            steps.append(
                f"Nouveau b[{i+1}] = {b[i]:.4f}"
            )

        historiques_A.append(A.copy())
        historiques_b.append(b.copy())

        steps.append(f"Matrice A après étape {k+1} :\n{A}")
        steps.append(f"Vecteur b après étape {k+1} :\n{b}")

    # ===== Substitution arrière =====

    x = np.zeros(n)

    steps.append("\n===== Substitution arrière =====")

    for i in range(n - 1, -1, -1):

        s = sum(A[i][j] * x[j] for j in range(i + 1, n))

        x[i] = (b[i] - s) / A[i][i]

        steps.append(
            f"x{i+1} = ({b[i]:.4f} - {s:.4f}) / "
            f"{A[i][i]:.4f} = {x[i]:.4f}"
        )

    # ===== Réorganisation pivot total =====

    x_final = np.zeros(n)

    for i in range(n):
        x_final[perm[i]] = x[i]

    steps.append(f"\nSolution finale : {x_final}")

    return historiques_A, historiques_b, x_final, steps


    
def afficher_matrices(L, U, k):
    print(f"Étape k = {k}")
    print("Matrice L :")
    print(L)
    print("\nMatrice U :")
    print(U)

def LU(A, b, affiche=False):
    n = len(A)
    A = A.astype(float)
    b = b.astype(float)  
    L = np.eye(n)
    perm = list(range(n))

    for k in range(n):
        if abs(A[k][k]) < 1e-14:
            ret = choix_utilisateur_LU(k+1, A, b, k, perm)  # ← passer b
            if ret == False:
                return None, None, None
        for i in range(k+1, n):
            m = A[i][k] / A[k][k]
            L[i][k] = m
            A[i, k:] = A[i, k:] - m * A[k, k:]
        if affiche:
            afficher_matrices(L, A, k)
    return L, A.copy(), perm

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
    recs = []
    if abs(np.linalg.det(A)) < 1e-14: # if determinant == 0
        return ["Matrice non inversible — aucune méthode directe possible."]
    if is_symmetric(A) and is_positive_definite(A):
        recs.append("Cholesky recommandé (matrice SDP — rapide et stable).")
    recs.append("Décomposition LU recommandée (cas général).")
    recs.append("Gauss avec pivot partiel : standard. Pivot total : si instabilité.")
    return recs



def choix_utilisateur_LU(i, A, b, k, perm):
    reponse = messagebox.askyesno(
        "Pivot nul",
        f"Pivot nul détecté à l'étape {i}.\n\n"
        "Oui  → Pivot partiel\n"
        "Non → Pivot total"
    )

    if reponse:
        ok = pivot_partiel(A, b, k)
        
    else:
        ok = pivot_total(A, b, k, perm)
        
    if not ok:
        messagebox.showerror(
            "Erreur",
            "Pivot impossible."
        )
        return False
    return True   

def choix_utilisateur_gauss(i, A, b, k, perm):
    reponse = messagebox.askyesno(
        "Pivot nul",
        f"Pivot nul détecté à l'étape {i}.\n\n"
        "Oui  → Pivot partiel\n"
        "Non → Pivot total"
    )

    if reponse:
        ok = pivot_partiel(A, b, k)
        
    else:
        ok = pivot_total(A, b, k, perm)
        
    if not ok:
        messagebox.showerror(
            "Erreur",
            "Aucune méthode de pivotage n'a pu résoudre le problème.."
        )
        return False
    return True   