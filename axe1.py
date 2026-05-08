import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


#######################------------------------##############################################################################################

#######################         DICOTOMOEI #################

def dichotomie(f_expr, a, b, tol, nmax):

    x = sp.Symbol('x')
    f = sp.sympify(f_expr)
    f_num = sp.lambdify(x, f, "numpy")

    fa = f_num(a)
    fb = f_num(b)

    # tvi 
    if fa * fb > 0:
        print("f(a) et f(b) ont le même signe")
        return False,None,[],[]

    iterations = []
    Erreurs=[]

    for i in range(nmax):

        m = (a + b) / 2
        fm = f_num(m)

        iterations.append( m)
        Erreurs.append(fm)

        if abs(fm) < tol or (b - a) / 2 < tol:
            break

        # mise à jour intervalle
        if fa * fm < 0:
            b = m
            fb = fm
        else:
            a = m
            fa = fm

    return True, m, iterations,Erreurs

##########################################################################################################################################

####point fixe

######################### #################################


def verifier_stabilite(phi, x, a, b):
    xs, vals = evaluer_phi_safe(phi, x, a, b)
    finite_vals = vals[np.isfinite(vals)]
    # Trop de NaN → phi instable sur l'intervalle
    if len(finite_vals) < 10:
        return False
    return np.all((finite_vals >= a) & (finite_vals <= b))



# k<1 k=|max (f')|

def verifier_contractante(phi, x, a, b):
    dphi = sp.diff(phi, x)
    g = sp.lambdify(x, dphi, "numpy")
    xs = np.linspace(a, b, 300)
    try:
        with np.errstate(all='ignore'):
            raw = g(xs)
        if np.isscalar(raw):
            raw = np.full(300, float(raw))
        else:
            raw = np.asarray(raw, dtype=complex)
            imag = np.abs(raw.imag)
            raw = np.where(imag < 1e-10, raw.real, np.nan)
            raw = raw.astype(float)
        vals = np.abs(raw)
        finite_vals = vals[np.isfinite(vals)]
        if len(finite_vals) == 0:
            return False, float('inf')
        k = float(np.max(finite_vals))
        return k < 1, k
    except Exception:
        return False, float('inf')


def point_fixe_avec_phi(f_expr, phi_exprs, a, b, x0, tol, nmax):
    """
    Méthode du point fixe avec phi(s) fournie(s) manuellement.

    Paramètres
    ----------
    f_expr   : str  – la fonction f(x) dont on cherche f(x)=0  (pour TVI et rapport)
    phi_exprs: str ou list[str]  – une ou plusieurs fonctions phi(x)
    a, b     : float – intervalle [a, b]
    x0  
    rapport        : list[(phi_sym, stable, contractante, k)]
    """

    x = sp.Symbol('x')

    # TVI 
    f_sym = sp.sympify(f_expr)
    f_num = sp.lambdify(x, f_sym, "numpy")
    if f_num(a) * f_num(b) > 0:
        print("f(a) et f(b) ont le même signe ")
        return False, None, None, [], [], []

    # phi_exprs en liste
    if isinstance(phi_exprs, str):
        phi_exprs = [phi_exprs]

    
    rapport = []
    meilleur_phi = None
    meilleur_k   = 1.0          # on cherche k < 1, le plus petit possible
    #tes phis

    for phi_str in phi_exprs:
        try:
            phi_sym = sp.sympify(phi_str)
        except Exception:
            print(f"  [ERREUR] impossible de parser : {phi_str}")
            continue

        stable= verifier_stabilite(phi_sym, x, a, b)
        contractant, k = verifier_contractante(phi_sym, x, a, b)

        rapport.append((phi_sym, stable, contractant, k))
        if stable and contractant and k < meilleur_k :
            meilleur_k   = k
            meilleur_phi = phi_sym

    #pas de phi valid 
    if meilleur_phi is None:
        print("\n Aucune phi donner  n'est stable et contractante sur [a, b].")
        print(" Méthode du point fixe non applicable avec ces phi.")
        return False, None, None, [], [], rapport


    # x SUV+ phi(x) 
    phi_num  = sp.lambdify(x, meilleur_phi, "numpy")
    curr_x   = x0
    iterations = [x0]
    erreurs    = []
    sol        = None
    success    = False

    for i in range(nmax):
        try:
            with np.errstate(all='ignore'):
                x_next = phi_num(curr_x)

            # Protection NaN / inf / complexe
            if np.iscomplex(x_next):
                x_next = float(np.real(x_next))
            x_next = float(x_next)
            if not np.isfinite(x_next):
                print(f"  [iter {i}] x_next infini ")
                break

            err = (meilleur_k / (1 - meilleur_k)) * abs(x_next - curr_x)
            iterations.append(x_next)
            erreurs.append(err)

            if err < tol:
                success = True
                sol = x_next
                break

            curr_x = x_next

        except Exception as e:
            print(f"  [iter {i}] Exception : {e}" )
            break
     # recommandations
    print("\n" + "="*58)
    print("  Analyse — Point Fixe ")
    print("="*58)

    n_valides   = sum(1 for _, s, c, _ in rapport if s and c)
    n_invalides = len(rapport) - n_valides

    if not success:
        print(f"\n  La méthode n'a pas convergé en {nmax} itérations.")
        print("  Essayez une phi avec un k plus petit, ou resserrez l'intervalle [a, b].")
    else:
        print(f"\n  Convergence obtenue en {len(iterations)-1} itérations paramatres bien choisi ")
    if success and len(iterations) > 10:
        print("  il est recommader de utuliser la methode newton")

    # Comparaison
    print("  Newton reste le choix le plus rapide et le plus fiable.")
    print("="*58 + "\n")

    return success, meilleur_phi, sol, iterations, erreurs, rapport

def point_fixe(f_expr, a, b, x0, tol, nmax):

    x = sp.Symbol('x')
    f = sp.sympify(f_expr)
    f_num = sp.lambdify(x, f, "numpy")

    phi_list = generer_phi(f,x)
    meilleur_phi = None
    meilleur_k = 1
    rapport = []
    
    fa = f_num(a)
    fb = f_num(b)

    # tvi 
    if fa * fb > 0:
        print("f(a) et f(b) ont le même signe ")
        return False,None,None,None,None,None
    

    # verfier la stablite et la conver des phi genere
    for phi in phi_list:

        stable = verifier_stabilite(phi, x, a, b)
        contractante, k = verifier_contractante(phi, x, a, b)

        rapport.append((phi, stable, contractante, k))

        if stable and contractante:
            if k < meilleur_k:
                meilleur_k = k
                meilleur_phi = phi

   #tout les phi non stable ou non contractante j arrte 
    if meilleur_phi is None:
         print(" Aucune fonction phi(x) est stable et contractante sur [a,b]")
         print(" Méthode du point fixe non applicable")

         return False,meilleur_phi,None,None,None,rapport

                
    #calcule xn+1=phi(xn)
   
    phi = sp.lambdify(x, meilleur_phi, "numpy")

    x = x0
    iterations = [x]
    erreurs=[]

    for i in range(nmax):
        try:
            with np.errstate(all='ignore'):
                x_next = phi(x)
            if np.iscomplex(x_next):
                x_next = float(np.real(x_next))
            x_next = float(x_next)
            if not np.isfinite(x_next):
                break
            erreur = (meilleur_k / (1 - meilleur_k)) * abs(x_next - x)

            erreurs.append(erreur)
            iterations.append(x_next)
            
            if erreur < tol:
                break

            x = x_next
        except Exception:
            break

    return True, meilleur_phi, x_next, iterations, erreurs,rapport

#point fixe avec relaxation 
def point_fixe_avec_relaxation(f_expr, a, b, x0, tol, nmax):

    x = sp.Symbol('x')
    f = sp.sympify(f_expr)
    f_num = sp.lambdify(x, f, "numpy")

    phi_list = generer_phi_avec_relaxation_et_newton(f,x)
    meilleur_phi = None
    meilleur_k = 1
    rapport = []
    
    fa = f_num(a)
    fb = f_num(b)

    # tvi 
    if fa * fb > 0:
        print("f(a) et f(b) ont le même signe ")
        return False,None,None,None,None,None
    

    # verfier la stablite et la conver des phi genere
    for phi in phi_list:

        stable = verifier_stabilite(phi, x, a, b)
        contractante, k = verifier_contractante(phi, x, a, b)

        rapport.append((phi, stable, contractante, k))

        if stable and contractante:
            if k < meilleur_k:
                meilleur_k = k
                meilleur_phi = phi

   #tout les phi non stable ou non contractante j arrte 
    if meilleur_phi is None:
         print(" Aucune fonction phi(x) est stable et contractante sur [a,b]")
         print(" Méthode du point fixe non applicable")

         return False,meilleur_phi,None,None,None,rapport

                
    #calcule xn+1=phi(xn)
   
    phi = sp.lambdify(x, meilleur_phi, "numpy")

    x = x0
    iterations = [x]
    erreurs=[]

    for i in range(nmax):
        try:
            with np.errstate(all='ignore'):
                x_next = phi(x)
            if np.iscomplex(x_next):
                x_next = float(np.real(x_next))
            x_next = float(x_next)
            if not np.isfinite(x_next):
                break
            erreur = (meilleur_k / (1 - meilleur_k)) * abs(x_next - x)

            erreurs.append(erreur)
            iterations.append(x_next)
            
            if erreur < tol:
                break

            x = x_next
        except Exception:
            break

    return True, meilleur_phi, x_next, iterations, erreurs,rapport


# genére phi(x)=x
def generer_phi(f, x):
    phi_list = []
    

    if f.is_Add:
        args = f.args
        for t in args:
            cible = t
            reste = sp.simplify(-(f - t))
            
          
            coeff = cible.coeff(x)
            if coeff != 0:
                phi_list.append(sp.simplify(reste / coeff))
            
            
            try:
                inv_sols = sp.solve(sp.Eq(cible, reste), x)
                for s in inv_sols:
                    phi_list.append(s)
            except:
                pass

            
            if isinstance(cible, sp.exp) and cible.args[0] == x:
                phi_list.append(sp.log(reste))
            
            if isinstance(cible, sp.Pow) and cible.args[0] == x:
                n = cible.args[1]
                phi_list.append(reste**(1/n))
        
            if (cible.is_Pow and cible.args[1] == -1) or (isinstance(cible, sp.Rational) == False and x in cible.atoms() and cible.as_powers_dict().get(x) == -1):
                
                phi_list.append(sp.simplify(1 / reste))

            if isinstance(cible, (sp.sin, sp.cos, sp.tan)) and cible.args[0] == x:
                if isinstance(cible, sp.sin):
                    phi_list.append(sp.asin(reste))
                elif isinstance(cible, sp.cos):
                    phi_list.append(sp.acos(reste))
                elif isinstance(cible, sp.tan):
                    phi_list.append(sp.atan(reste))

    # elimine double 
    unique_phi = []
    for p in phi_list:
        try:
            ps = sp.simplify(p)
            if ps.has(x) and ps != x and ps not in unique_phi:
                unique_phi.append(ps)
        except:
            continue
            
    return unique_phi

def generer_phi_avec_relaxation_et_newton(f, x):
    phi_list = []
    
    
    f_prime = sp.diff(f, x)
    if f_prime != 0:
        phi_list.append(sp.simplify(x - f/f_prime))
    for lam in [0.1, 0.5]:
        phi_list.append(sp.simplify(x - lam * f))
        phi_list.append(sp.simplify(x + lam * f))

    if f.is_Add:
        args = f.args
        for t in args:
            cible = t
            reste = sp.simplify(-(f - t))
            
            coeff = cible.coeff(x)
            if coeff != 0:
                phi_list.append(sp.simplify(reste / coeff))
            
            try:
   
                inv_sols = sp.solve(sp.Eq(cible, reste), x)
                for s in inv_sols:
                    phi_list.append(s)
            except:
                pass

            if isinstance(cible, sp.exp) and cible.args[0] == x:
                phi_list.append(sp.log(reste))

            
            if isinstance(cible, sp.Pow) and cible.args[0] == x:
                n = cible.args[1]
                phi_list.append(reste**(1/n))
          
            if (cible.is_Pow and cible.args[1] == -1) or (isinstance(cible, sp.Rational) == False and x in cible.atoms() and cible.as_powers_dict().get(x) == -1):
        
                phi_list.append(sp.simplify(1 / reste))

            if isinstance(cible, (sp.sin, sp.cos, sp.tan)) and cible.args[0] == x:
                if isinstance(cible, sp.sin):
                    phi_list.append(sp.asin(reste))
                elif isinstance(cible, sp.cos):
                    phi_list.append(sp.acos(reste))
                elif isinstance(cible, sp.tan):
                    phi_list.append(sp.atan(reste))

  
    unique_phi = []
    for p in phi_list:
        try:
            ps = sp.simplify(p)
            if ps.has(x) and ps != x and ps not in unique_phi:
                unique_phi.append(ps)
        except: continue
            
    return unique_phi




############################################################################################################




####################################################### algoritheme de newtoon ##########################################################



def verifier_derivative_non_nulle(f, a, b):
    x = sp.Symbol('x')
    # f(x) = 0
    solutions = sp.solve(f, x)

    # VER si  dans [a, b]
    solutions_dans_intervalle = [sol for sol in solutions if sol.is_real and a <= sol <= b]

    if solutions_dans_intervalle:
        return 0

    return 1


def verifier_ddf_signe_constant(f_expr, a, b):
    x = sp.Symbol('x')
    f = sp.lambdify(x, f_expr, 'numpy')
    x_range = np.linspace(a, b, 200)
    valeurs = f(x_range)

    # Vérifier si toutes les valeurs ont le même signe
    if np.all(valeurs > 0):
        return 1
    
    elif np.all(valeurs < 0):
        return 1
    else:
        return 0


def newton_(f_expr_str, a, b, x0, epsilon, max_iter):
    
    x = sp.Symbol('x')
    f_expr = sp.sympify(f_expr_str)
    f = sp.lambdify(x, f_expr, 'numpy')
        # calc les f' et f"
    df_expr = sp.diff(f_expr, x)
    ddf_expr = sp.diff(df_expr, x)
    df = sp.lambdify(x, df_expr, 'numpy')
    ddf = sp.lambdify(x, ddf_expr, 'numpy')

    #f(a)f(b)<0
    f_a ,f_b = f(a),f(b)
    if f_a*f_b >= 0:
        print ("pas de solution sur [a,b]")
        return False,None, [], [], f_expr, df_expr, ddf_expr



    # f'!=0 sur a b 
    cond1 = verifier_derivative_non_nulle(df_expr ,a, b)
    if cond1 ==0 :
        print("f' s'annule sur[a,b] donc algoritheme newtoon non applicable ")
        return False , None, [], [], f_expr, df_expr, ddf_expr

    #f" ne change pas de signe sur ab 
    cond2=verifier_ddf_signe_constant(ddf_expr, a, b)
    if cond2 ==0 :
        print("f''change de  signe sur [a,b] donc algoritheme newtoon non applicable ")
        return False , None, [], [], f_expr, df_expr, ddf_expr

    #|f(c)| / |df(c)| < b-a 
    df_a, df_b = df(a), df(b)
    if df_a < df_b:
        c = a
    else:
        c = b

    ratio = abs(f(c)) / abs(df(c))
    if ratio > (b - a):
        print("️ Critère |f(c)|/|f'(c)| <= b-a non respecté  Newton non applicable.")
        return  False ,None, [], [], f_expr, df_expr, ddf_expr
    
    #application newton
    
    x_val = x0
    iterations = [x_val]
    erreurs = [abs(f(x0))]

    x_range = np.linspace(a, b, 100)
    m = np.min(np.abs(df(x_range)))
    M = np.max(np.abs(ddf(x_range)))

    for i in range(max_iter):
        x_next = x_val - f(x_val) / df(x_val)
        erreur_estimee = (M / (2 * m)) * abs(x_next - x_val) ** 2 if m != 0 else np.inf

        erreurs.append(erreur_estimee)
        iterations.append(x_next)

        if erreur_estimee < epsilon:
            break

        x_val = x_next

    return  True, x_next, iterations, erreurs,f_expr, df_expr, ddf_expr
    


#################affichage ####################################################################
        
def afficher_tableau_plt(methode, iterations, erreurs):
  
    data = [['Iteration', 'Solution', 'Erreur Estimée']]
    for i, (sol, err) in enumerate(zip(iterations, erreurs)):
        data.append([
            i,
            f'{sol:.6f}',
            f'{err:.6e}'
        ])

    plt.figure(figsize=(8, 4))
    plt.axis('off')
    print(f"\n===== Tableau : {methode} =====")
    table = plt.table(
        cellText=data[1:],
        colLabels=data[0],
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    plt.show()
  



def trace_courbe(f_expr, iterations, solution,a,b,titre):
    x = sp.Symbol('x')
    f = sp.lambdify(x, f_expr, 'numpy')
    x = np.linspace(a, b, 500)
    y = f(x)

    plt.figure(figsize=(10,6))

    plt.plot(x, y, label="f(x)")
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')

    # points itérations
    yi = [f(v) for v in iterations]
    plt.plot(iterations, yi, marker='x', color='blue',
             label="Itérations")
    plt.plot(solution, f(solution), 'ro',
                 markersize=8,
                 label="Solution")

    plt.title(titre)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()
    plt.show()
def comparer_methodes(f_expr,newton_iter, pf_iter, dicho_iter,sol_newton, sol_pf,sol_dicho,a,b):
    x = sp.Symbol('x')
    f = sp.lambdify(x, f_expr, 'numpy')

    x = np.linspace(a, b, 500)
    

    fig, axs = plt.subplots(1,3, figsize=(18,5))

    # Newton
    axs[0].plot(x, f(x))
    axs[0].axhline(0,color='black')
    axs[0].plot(newton_iter,
                [f(v) for v in newton_iter],
                'bx')
    axs[0].plot(sol_newton, f(sol_newton), 'ro')
    axs[0].set_title("Newton")

    # -Point fixe
    axs[1].plot(x, f(x))
    axs[1].axhline(0,color='black')
    axs[1].plot(pf_iter,
                [f(v) for v in pf_iter],
                'gx')
    axs[1].plot(sol_pf, f(sol_pf), 'ro')
    axs[1].set_title("Point Fixe")

    # DICHOTOMIE
    axs[2].plot(x, f(x))
    axs[2].axhline(0,color='black')
    axs[2].plot(dicho_iter,
                [f(v) for v in dicho_iter],
                'mx')
    axs[2].plot(sol_dicho, f(sol_dicho), 'ro')
    axs[2].set_title("Dichotomie")

    for ax in axs:
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    print("\n recomnation")

    print("Newton rapide mais pas toujours efficace")
    print("Point fixe simple mais dépend de phi(x)")
    print("Dichotomie lente mais toujours stable")


# Calculer ordre
def calculer_ordre_point_fixe(phi_expr, solution):
    
    x = sp.Symbol('x')
    phi = sp.sympify(phi_expr)

    dphi  = sp.diff(phi, x)       # φ'(x)
    ddphi = sp.diff(dphi, x)      # φ''(x)

    val_dphi  = float(dphi.subs(x, solution))   # φ'(α)
    val_ddphi = float(ddphi.subs(x, solution))  # φ''(α)

    if abs(val_dphi) > 1e-6:
        return 1, abs(val_dphi)        # ordre 1

    if abs(val_ddphi) > 1e-6:
        return 2, abs(val_ddphi) / 2  # ordre 2

    return 3, 0.0                      # ordre >= 3

def evaluer_phi_safe(phi, x_sym, a, b, n=300):
    """Évalue phi(x) sur [a,b] en filtrant les NaN/inf/complexes."""
    f = sp.lambdify(x_sym, phi, "numpy")
    xs = np.linspace(a, b, n)
    try:
        with np.errstate(all='ignore'):
            vals = f(xs)
        # Gérer le cas scalaire (phi = constante)
        if np.isscalar(vals):
            vals = np.full(n, float(vals))
        else:
            vals = np.asarray(vals, dtype=complex)
            # Garder partie réelle seulement si partie imaginaire ≈ 0
            imag = np.abs(vals.imag)
            vals = np.where(imag < 1e-10, vals.real, np.nan)
            vals = vals.astype(float)
        return xs, vals
    except Exception:
        return xs, np.full(n, np.nan)


# Recommendation automatique selon l ordre
def recommander_methode(
    dicho_success,  dicho_sol,
    pf_success,     pf_sol,    meilleur_phi,
    newton_success, newton_sol
):
    candidats = []

    if dicho_success:
        candidats.append(("Dichotomie", 1, 0.5))

    if pf_success and meilleur_phi is not None and pf_sol is not None:
        ordre, taux = calculer_ordre_point_fixe(meilleur_phi, pf_sol)
        candidats.append(("Point fixe", ordre, taux))

    if newton_success:
        candidats.append(("Newton", 2, 0.0))

    if not candidats:
        return [], "Aucune méthode applicable sur cet intervalle."

    # ordre max
    ordre_max = max(c[1] for c in candidats)
    
    # tous ceux qui ont l'ordre max → recommandes directement, sans comparer le taux
    recommande = [c[0] for c in candidats if c[1] == ordre_max]

    # message
    ordre_labels = {1: "linéaire", 2: "quadratique", 3: "cubique+"}
    lignes = ["Methodes applicables :"]

    for nom, ordre, taux in candidats:
       
        lignes.append(f"  • {nom} : ordre {ordre} ({ordre_labels.get(ordre, '?')})")

    lignes.append("")
    if len(recommande) == 1:
        lignes.append(f"Recommandation : {recommande[0]}")
    else:
        lignes.append(f"Recommandation : {' et '.join(recommande)} (meme ordre de convergence)")

    return recommande, "\n".join(lignes)