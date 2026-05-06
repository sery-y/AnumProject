import math
import numpy as np
import matplotlib.pyplot as plt
from matrices import choisir_norme, is_DDS, is_positive_definite, is_symmetric, rayon_spectral ,saisir_matrice, saisir_vecteur



def jacobi(A,b,x0,max_iter=100,epsilon=1e-6,visualiser=False):

    D=np.diag(np.diag(A))
    E=-np.tril(A,-1)
    F=-np.triu(A,1)

    D_inv=np.diag(1/np.diag(D))
    Js=D_inv @ (E+F)
    c=D_inv @ b
    
    type_norme = input("Choisir norme (1 / inf / 2 / p) : ")
    p_norm= None
    if type_norme == "p":
     p_norm = float(input("Entrer la valeur de p : "))

    M_norm = choisir_norme(Js, type_norme, p_norm)
    print("Matrice d'itération Jacobi:\n", Js)

    p=rayon_spectral(Js)
    print("Rayon spectral =", p)

    if p < 1:
        print("la methode de jacobi converge")
    else:
       print("la methode de jacobi ne converge pas ")

    X = x0
    if M_norm < 1:
      X1 = Js @ X + c
      err_apriori = np.linalg.norm(X - X1) * (M_norm / (1 - M_norm))
      print("Estimation derreure  =", err_apriori)
    else:
     print("||M|| >= 1 → estimation derreur non applicable")
    suite_vectorielle=[X.copy()]
    for i in range(max_iter):
       X_new = Js @ X + c
       suite_vectorielle.append(X_new.copy())

       erreur = np.linalg.norm(A @ X_new - b)
       diff = np.linalg.norm(X_new - X)
       if M_norm < 1:
          err_estime = diff * (M_norm / (1 - M_norm))
          print("Erreur estimée =", err_estime)
       else:
          err_estime = diff
          print("Erreur estimée (a posteriori) =", err_estime)

       if M_norm < 1:
        if diff * (M_norm / (1 - M_norm)) < epsilon:
         print("Convergence")
         break
       else:
        if diff < epsilon:
         print("Convergence")
         break

       X = X_new
    if visualiser:
     visualiser_convergence(suite_vectorielle, "Jacobi")
     visualiser_solution(suite_vectorielle, b, A, "Jacobi")
    return Js, suite_vectorielle , p

def jacobi_console():

    n = int(input("Entrer la taille de la matrice A : "))

    A = saisir_matrice(n)
    b = saisir_vecteur(n, "b")
    x0 = saisir_vecteur(n, "x0")

    max_iter = int(input("Nombre max d'itérations : "))
    epsilon = float(input("Tolérance : "))
    visualiser = input("Voulez-vous voir les graphiques ? (o/n) : ").lower() == 'o'
    return jacobi(A, b, x0, max_iter, epsilon,visualiser=visualiser)

def gauss_seidel(A, b, x0, max_iter=100, epsilon=1e-6,visualiser=False):

    D = np.diag(np.diag(A))
    E = -np.tril(A, -1)
    F = -np.triu(A, 1)

    DE_inv = np.linalg.inv(D - E)

    Gs = DE_inv @ F
    c = DE_inv @ b
    type_norme = input("Choisir norme (1 / inf / 2 / p) : ")
    p_norm = None
    if type_norme == "p":
        p_norm = float(input("Entrer la valeur de p : "))

    M_norm = choisir_norme(Gs, type_norme, p_norm)
    print("Matrice d'itération GS :\n", Gs)

    p = rayon_spectral(Gs)
    print("Rayon spectral =", p)

    if p < 1:
        print("Gauss-Seidel converge")
    else:
        print("Gauss-Seidel ne converge pas")
    
    if M_norm < 1:
     print("Estimation derreur disponible")
    else:
     print("||M|| >= 1 → estimation derreur non applicable")
    X = x0
    suite_vectorielle = [X.copy()]

    for i in range(max_iter):
        X_new = Gs @ X + c
        suite_vectorielle.append(X_new.copy())

        erreur = np.linalg.norm(A @ X_new - b)
        diff = np.linalg.norm(X_new - X)
        if M_norm < 1:
            err_estime = diff * (M_norm / (1 - M_norm))
            print("Erreur estimée =", err_estime)
        else:
            err_estime = diff
            print("Erreur estimée =", err_estime)

        if M_norm < 1:
         if err_estime < epsilon:
          print("Convergence")
          break
        else:
         if diff < epsilon:
          print("Convergence")
          break

        X = X_new
    if visualiser :
     visualiser_convergence(suite_vectorielle, "Gauss-Seidel")
     visualiser_solution(suite_vectorielle, b, A, "Gauss-Seidel")
    return Gs, suite_vectorielle, p

def gauss_seidel_console():
    n = int(input("Entrer la taille de la matrice A : "))

    A = saisir_matrice(n)
    b = saisir_vecteur(n, "b")
    x0 = saisir_vecteur(n, "x0")

    max_iter = int(input("Nombre max d'itérations : "))
    epsilon = float(input("Tolérance : "))
    visualiser = input("Voulez-vous voir les graphiques ? (o/n) : ").lower() == 'o'
    return gauss_seidel(A, b, x0, max_iter, epsilon, visualiser=visualiser)

def analyser_matrice(A):
    res = {}
    res["DDS"] = is_DDS(A)
    res["symétrique"] = is_symmetric(A)
    res["définie positive"] = is_positive_definite(A)
    
    return res

def recommander_methodes(A):
    info = analyser_matrice(A)
    recommandations = []
    if info["DDS"]:
        recommandations.append("Jacobi recommandé (matrice DDS → convergence garantie)")
        recommandations.append("Gauss-Seidel recommandé (matrice DDS → convergence garantie)")
    
    if info["symétrique"] and info["définie positive"]:
        recommandations.append("Gauss-Seidel recommandé (matrice symétrique et définie positive → convergence garantie)")
    return recommandations , info

def afficher_recommandations(A):
    recommandations, info = recommander_methodes(A)
    print("Analyse de la matrice A :")
    for prop, val in info.items():
        print(f" - {prop} : {'Oui' if val else 'Non'}")
    
    if recommandations:
        print("\nMéthodes recommandées pour résoudre Ax=b :")
        for methode in recommandations:
            print(f" - {methode}")
    else:
        print("\nAucune méthode itérative recommandée pour cette matrice.")

def comparer_methodes(A, b, x0, max_iter=100, epsilon=1e-6):
    
    print("\n--- Comparaison Jacobi vs Gauss-Seidel ---")

    # Jacobi
    Js, suite_jacobi, p_j = jacobi(A, b, x0, max_iter, epsilon, visualiser=False)
    X_j = suite_jacobi[-1]
    err_j = np.linalg.norm(A @ X_j - b)

    # Gauss-Seidel
    Gs, suite_gs, p_gs = gauss_seidel(A, b, x0, max_iter, epsilon, visualiser=False)
    X_gs = suite_gs[-1]
    err_gs = np.linalg.norm(A @ X_gs - b)

    print("\n--- Résultats ---")
    print(f"Jacobi : p = {p_j}, erreur = {err_j}")
    print(f"Gauss-Seidel : p = {p_gs}, erreur = {err_gs}")

    # Comparaison vitesse
    if p_j < p_gs:
        print("Jacobi converge plus rapidement (p plus petit)")
    else:
        print("Gauss-Seidel converge plus rapidement")

    # Comparaison précision
    if err_j < err_gs:
        print("Jacobi est plus précis")
    else:
        print("Gauss-Seidel est plus précis")

    # Recommandation finale
    if err_j < err_gs and p_j <= p_gs:
        print("\n Méthode recommandée : Jacobi")
    else:
        print("\n Méthode recommandée : Gauss-Seidel")

def visualiser_convergence(suite_iterative, titre="Convergence", methode="Jacobi"):
    # Calculer l'erreur relative entre itérations successives
    erreurs = []
    for i in range(len(suite_iterative) - 1):
        err = np.linalg.norm(suite_iterative[i+1] - suite_iterative[i])
        erreurs.append(err)
    
    # Créer le graphique
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(erreurs)+1), erreurs, 'b-o', linewidth=2, markersize=4)
    plt.title(f"{titre} - {methode}", fontsize=14)
    plt.xlabel("Numéro d'itération", fontsize=12)
    plt.ylabel("Erreur entre deux itérations consécutives", fontsize=12)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Afficher la dernière erreur
    if erreurs:
        plt.text(0.7, 0.95, f"Erreur finale: {erreurs[-1]:.2e}", 
                 transform=plt.gca().transAxes, fontsize=10)
    
    plt.show()

def visualiser_solution(suite_iterative, b, A, titre="Convergence"):
    residus = []
    for x in suite_iterative:
        residu = np.linalg.norm(A @ x - b)
        residus.append(residu)
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(len(residus)), residus, 'r-o', linewidth=2, markersize=4)
    plt.title(f"{titre} - Évolution du résidu(erreur)", fontsize=14)
    plt.xlabel("Numéro d'itération", fontsize=12)
    plt.ylabel("||Ax - b||", fontsize=12)
    plt.show() 

def get_iterative_methodes():
 return {
    "jacobi": jacobi_console,
    "gauss_seidel": gauss_seidel_console
}