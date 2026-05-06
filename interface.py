import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sympy as sp
import csv, os, json
from axe1 import dichotomie, point_fixe, newton_, point_fixe_avec_relaxation, recommander_methode

# ══════════════════════════════════════════════
#  PALETTE 
# ══════════════════════════════════════════════
C = {
    "bg":        "#111118",
    "panel":     "#18181f",
    "card":      "#1f1f2b",
    "card2":     "#26263a",
    "hover":     "#2a2a3c",
    "border":    "#2e2e42",
    "accent":    "#7c6fcd",
    "acc_light": "#a89fe8",
    "acc_bg":    "#1e1b36",
    "teal":      "#1db88a",
    "teal_bg":   "#0d2820",
    "amber":     "#d4880a",
    "white":     "#e8e6f2",
    "gray":      "#7a7990",
    "muted":     "#4a4958",
    "red":       "#e05252",
    "badge_bg":  "#2d2b45",
}

F = {
    "title":  ("Segoe UI", 18, "bold"),
    "h2":     ("Segoe UI", 12, "bold"),
    "h3":     ("Segoe UI", 10, "bold"),
    "body":   ("Segoe UI", 10),
    "small":  ("Segoe UI", 9),
    "mono":   ("Consolas", 10),
    "big":    ("Segoe UI", 22, "bold"),
    "sup":    ("Segoe UI", 14, "bold"),
}

def _forward_sub(L, b):
    """Substitution avant pour Ly = b."""
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    return y
 
 
def _back_sub(U, y):
    """Substitution arrière pour Ux = y."""
    n = len(y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
    return x
 
 
def gauss_pivot_partiel(A, b):
    """Élimination de Gauss avec pivot partiel.
    Retourne (solution, étapes, pivot_swaps)."""
    n = len(b)
    Ab = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])
    steps = []
    for k in range(n):
        # Pivot partiel : max dans la colonne k
        max_row = k + np.argmax(np.abs(Ab[k:, k]))
        if max_row != k:
            Ab[[k, max_row]] = Ab[[max_row, k]]
            steps.append(f"Échange lignes {k+1} ↔ {max_row+1}")
        if abs(Ab[k, k]) < 1e-14:
            raise ValueError(f"Pivot nul à l'étape {k+1} — système singulier.")
        for i in range(k + 1, n):
            factor = Ab[i, k] / Ab[k, k]
            Ab[i] -= factor * Ab[k]
            steps.append(f"L{i+1} ← L{i+1} − {factor:.4f}·L{k+1}")
    x = _back_sub(Ab[:, :n], Ab[:, n])
    return x, steps
 
 
def gauss_pivot_total(A, b):
    """Élimination de Gauss avec pivot total."""
    n = len(b)
    Ab = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])
    col_order = list(range(n))
    steps = []
    for k in range(n):
        # Pivot total : max sur la sous-matrice restante
        sub = np.abs(Ab[k:, k:n])
        ri, ci = np.unravel_index(np.argmax(sub), sub.shape)
        ri += k; ci += k
        if ri != k:
            Ab[[k, ri]] = Ab[[ri, k]]
            steps.append(f"Échange lignes {k+1} ↔ {ri+1}")
        if ci != k:
            Ab[:, [k, ci]] = Ab[:, [ci, k]]
            col_order[k], col_order[ci] = col_order[ci], col_order[k]
            steps.append(f"Échange colonnes {k+1} ↔ {ci+1}")
        if abs(Ab[k, k]) < 1e-14:
            raise ValueError(f"Pivot nul à l'étape {k+1} — système singulier.")
        for i in range(k + 1, n):
            factor = Ab[i, k] / Ab[k, k]
            Ab[i] -= factor * Ab[k]
    x_perm = _back_sub(Ab[:, :n], Ab[:, n])
    # Réordonner x selon la permutation des colonnes
    x = np.zeros(n)
    for i, ci in enumerate(col_order):
        x[ci] = x_perm[i]
    return x, steps
 
 
def decomp_lu(A, b):
    """Décomposition LU (Doolittle) sans pivot.
    Retourne (solution, L, U, étapes)."""
    n = len(b)
    L = np.eye(n)
    U = A.astype(float).copy()
    steps = []
    for k in range(n):
        if abs(U[k, k]) < 1e-14:
            raise ValueError(f"Pivot nul — essayez Gauss avec pivot.")
        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i] -= factor * U[k]
            steps.append(f"m_{i+1}{k+1} = {factor:.4f}")
    y = _forward_sub(L, b.astype(float))
    x = _back_sub(U, y)
    return x, L, U, steps
 
 
def decomp_cholesky(A, b):
    """Décomposition de Cholesky (A doit être SPD).
    Retourne (solution, L, étapes)."""
    n = len(b)
    A = A.astype(float)
    L = np.zeros((n, n))
    steps = []
    for j in range(n):
        sum_sq = np.dot(L[j, :j], L[j, :j])
        val = A[j, j] - sum_sq
        if val <= 0:
            raise ValueError(
                f"Matrice non définie positive (valeur négative à ({j+1},{j+1})).")
        L[j, j] = np.sqrt(val)
        steps.append(f"L[{j+1},{j+1}] = √{val:.4f} = {L[j,j]:.4f}")
        for i in range(j + 1, n):
            L[i, j] = (A[i, j] - np.dot(L[i, :j], L[j, :j])) / L[j, j]
            steps.append(f"L[{i+1},{j+1}] = {L[i,j]:.4f}")
    y = _forward_sub(L, b.astype(float))
    x = _back_sub(L.T, y)
    return x, L, steps
 
 
def jacobi(A, b, x0, tol, nmax):
    """Méthode de Jacobi itérative.
    Retourne (converged, x, iters_list, errors_list, spectral_radius)."""
    n = len(b)
    D = np.diag(A)
    if np.any(np.abs(D) < 1e-14):
        raise ValueError("Zéro sur la diagonale — réordonnez le système.")
    R = A - np.diag(D)
    x = x0.copy().astype(float)
    iters, errors = [], []
    for k in range(nmax):
        x_new = (b - R @ x) / D
        err = np.linalg.norm(x_new - x, np.inf)
        iters.append(x_new.copy())
        errors.append(err)
        x = x_new
        if err < tol:
            # Rayon spectral de la matrice d'itération B = -D^{-1}R
            B = -R / D[:, None]
            rho = max(abs(np.linalg.eigvals(B)))
            return True, x, iters, errors, rho
    B = -R / D[:, None]
    rho = max(abs(np.linalg.eigvals(B)))
    return False, x, iters, errors, rho
 
 
def gauss_seidel(A, b, x0, tol, nmax):
    """Méthode de Gauss-Seidel itérative.
    Retourne (converged, x, iters_list, errors_list, spectral_radius)."""
    n = len(b)
    x = x0.copy().astype(float)
    iters, errors = [], []
    for k in range(nmax):
        x_new = x.copy()
        for i in range(n):
            s = sum(A[i, j] * x_new[j] for j in range(n) if j != i)
            if abs(A[i, i]) < 1e-14:
                raise ValueError(f"Zéro sur la diagonale ligne {i+1}.")
            x_new[i] = (b[i] - s) / A[i, i]
        err = np.linalg.norm(x_new - x, np.inf)
        iters.append(x_new.copy())
        errors.append(err)
        x = x_new
        if err < tol:
            # Matrice d'itération de GS : (D+L)^{-1} U
            D_L = np.tril(A)
            U_mat = A - D_L
            try:
                B = -np.linalg.inv(D_L) @ U_mat
                rho = max(abs(np.linalg.eigvals(B)))
            except Exception:
                rho = float('nan')
            return True, x, iters, errors, rho
    try:
        D_L = np.tril(A)
        U_mat = A - D_L
        B = -np.linalg.inv(D_L) @ U_mat
        rho = max(abs(np.linalg.eigvals(B)))
    except Exception:
        rho = float('nan')
    return False, x, iters, errors, rho
 


# ══════════════════════════════════════════════
#  WIDGETS UTILITAIRES
# ══════════════════════════════════════════════

def make_sep(parent, color=None):
    tk.Frame(parent, bg=color or C["border"], height=1).pack(fill="x")

class Card(tk.Frame):
    def __init__(self, parent, title=None, **kw):
        super().__init__(parent, bg=C["card"],
                         highlightthickness=1,
                         highlightbackground=C["border"], **kw)
        if title:
            hdr = tk.Frame(self, bg=C["card"], pady=10, padx=14)
            hdr.pack(fill="x")
            tk.Label(hdr, text=title.upper(), font=("Segoe UI",8,"bold"),
                     bg=C["card"], fg=C["muted"]).pack(side="left")
            tk.Frame(self, bg=C["border"], height=1).pack(fill="x")
        self.body = tk.Frame(self, bg=C["card"], padx=14, pady=10)
        self.body.pack(fill="both", expand=True)

class StatCard(tk.Frame):
    def __init__(self, parent, label_text):
        super().__init__(parent, bg=C["card"],
                         highlightthickness=1,
                         highlightbackground=C["border"],
                         padx=16, pady=14)
        self._val = tk.Label(self, text="—", font=F["big"],
                             bg=C["card"], fg=C["white"])
        self._val.pack()
        tk.Label(self, text=label_text, font=F["small"],
                 bg=C["card"], fg=C["gray"]).pack()

    def set(self, v): self._val.configure(text=v)

def mk_entry(parent, default="", width=18):
    e = tk.Entry(parent, font=F["mono"],
                 bg=C["card2"], fg=C["white"],
                 insertbackground=C["white"],
                 relief="flat", bd=0,
                 highlightthickness=1,
                 highlightbackground=C["border"],
                 highlightcolor=C["accent"],
                 width=width)
    e.insert(0, default)
    return e

def mk_btn(parent, text, cmd, color=None, secondary=False):
    if secondary:
        return tk.Button(parent, text=text, font=F["small"],
                         bg=C["card2"], fg=C["gray"],
                         activebackground=C["hover"],
                         activeforeground=C["white"],
                         relief="flat", bd=0, cursor="hand2",
                         padx=10, pady=5, command=cmd)
    return tk.Button(parent, text=text, font=F["h3"],
                     bg=color or C["accent"], fg=C["white"],
                     activebackground=C["acc_light"],
                     activeforeground=C["white"],
                     relief="flat", bd=0, cursor="hand2",
                     padx=14, pady=7, command=cmd)

# ══════════════════════════════════════════════
#  AXE 1
# ══════════════════════════════════════════════

class Axe1Frame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=C["bg"])
        self._iters = []
        self._build()

    def _build(self):
       
         #  Initialisation des résultats 
        self._initialized = False # pour detecter si l utilisateur a entrer des parametres ou pas
        self._dicho_success  = False
        self._dicho_sol      = None
        self._pf_success     = False
        self._pf_sol         = None
        self._pf_phi         = None
        self._newton_success = False
        self._newton_sol     = None
         # Top bar(Title)
        top = tk.Frame(self, bg=C["bg"], padx=28, pady=18)
        top.pack(fill="x")
        tk.Label(top, text="Axe 1 — Resolution de fonctions non lineaires",
                 font=F["title"], bg=C["bg"], fg=C["white"]).pack(side="left")
        
        tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

        # Scrollable
        outer = tk.Canvas(self, bg=C["bg"], bd=0, highlightthickness=0)
        sb = ttk.Scrollbar(self, orient="vertical", command=outer.yview)
        outer.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        outer.pack(fill="both", expand=True)
        content = tk.Frame(outer, bg=C["bg"])
        win = outer.create_window((0,0), window=content, anchor="nw")
        outer.bind("<Configure>", lambda e: outer.itemconfig(win, width=e.width))
        content.bind("<Configure>", lambda e: outer.configure(
            scrollregion=outer.bbox("all")))

        p = dict(padx=28, pady=8)

        # ── Choisir un algorithme ──
        ac = Card(content, "Choisir un algorithme")
        ac.pack(fill="x", **p)
        grid = tk.Frame(ac.body, bg=C["card"])
        grid.pack(fill="x")
        grid.columnconfigure((0,1), weight=1)

        self.algo_var = tk.StringVar(value="Newton") #lalgo choisi
        self._algo_frames = {}
        algos = [
            ("Newton", "Convergence quadratique", 0, 0),
            ("Dichotomie",     "Convergence linéaire",         0, 1),
            ("Point fixe",     "Simple, conditionnel",    1, 0),
            
        ]
        for name, desc, r, c in algos:
            f = tk.Frame(grid, bg=C["card2"],
                         highlightthickness=1,
                         highlightbackground=C["border"],
                         padx=14, pady=12, cursor="hand2")
            f.grid(row=r, column=c, padx=5, pady=5, sticky="ew")
            nl = tk.Label(f, text=name, font=F["h3"], bg=C["card2"], fg=C["white"])
            nl.pack(anchor="w")
            dl = tk.Label(f, text=desc, font=F["small"], bg=C["card2"], fg=C["gray"])
            dl.pack(anchor="w")
            for w in [f, nl, dl]:
                w.bind("<Button-1>", lambda e, n=name: self._sel_algo(n))
            self._algo_frames[name] = f
        

        # ── Recommandation ──
        rec = tk.Frame(content, bg=C["teal_bg"],
                       highlightthickness=1, highlightbackground=C["teal"],
                       padx=16, pady=12)
        rec.pack(fill="x", **p)
        tk.Label(rec, text="Recommandation", font=F["h3"],
                 bg=C["teal_bg"], fg=C["teal"]).pack(anchor="w")
        self._rec = tk.Label(rec, text="", font=F["body"],
                             bg=C["teal_bg"], fg="#b0e8d5",
                             wraplength=900, justify="left")
        self._rec.pack(anchor="w", pady=(4,0))

        self._sel_algo("Newton")

        # ── Parametres ──
        pc = Card(content, "Parametres")
        pc.pack(fill="x", **p)
        row = tk.Frame(pc.body, bg=C["card"])
        row.pack(fill="x")

        fields = [
            ("f(x) =", "e_fx", "", 20),
            ("x0",     "e_x0", "", 7),
            ("a", "e_a", "", 5),
            ("b", "e_b", "", 5),
            ("Tolerance", "e_tol", "", 7),
        ]
        for lbl, attr, default, w in fields:
            col = tk.Frame(row, bg=C["card"])
            col.pack(side="left", padx=(0,10))
            tk.Label(col, text=lbl, font=F["small"],
                     bg=C["card"], fg=C["gray"]).pack(anchor="w", pady=(0,3))
            e = mk_entry(col, default, w)
            e.pack(ipady=5)
            setattr(self, attr, e) #stocke les entrees

        for attr in ["e_fx", "e_x0", "e_a", "e_b", "e_tol"]:
          getattr(self, attr).bind("<KeyRelease>", self._on_input_change)

        col_b = tk.Frame(row, bg=C["card"])
        col_b.pack(side="left")
        tk.Label(col_b, text=" ", font=F["small"], bg=C["card"], fg=C["gray"]).pack(pady=(0,3))
        mk_btn(col_b, "Calculer", self._run).pack(ipady=4)

        # ── Stats ──
        sr = tk.Frame(content, bg=C["bg"])
        sr.pack(fill="x", **p)
        sr.columnconfigure((0,1,2), weight=1)
        self.st_r = StatCard(sr, "Racine approchee")
        self.st_r.grid(row=0, column=0, sticky="ew", padx=(0,8))
        self.st_i = StatCard(sr, "Iterations")
        self.st_i.grid(row=0, column=1, sticky="ew", padx=(0,8))
        self.st_e = StatCard(sr, "Erreur finale")
        self.st_e.grid(row=0, column=2, sticky="ew")

        # ── Graphe ──
        gc = Card(content, "Visualisation")
        gc.pack(fill="x", **p)
        self.fig, self.ax = plt.subplots(figsize=(9, 3.2), facecolor=C["card"])
        self.ax.set_facecolor(C["bg"])
        self._style_ax(self.ax)
        self.canvas = FigureCanvasTkAgg(self.fig, gc.body)
        self.canvas.get_tk_widget().pack(fill="both")

        # ── Tableau ──
        tc = Card(content, "Tableau des iterations")
        tc.pack(fill="x", **p)

        sty = ttk.Style()
        sty.theme_use("clam")
        sty.configure("D.Treeview",
                      background=C["card"], foreground=C["white"],
                      fieldbackground=C["card"], rowheight=24, font=F["mono"])
        sty.configure("D.Treeview.Heading",
                      background=C["card2"], foreground=C["acc_light"],
                      font=("Segoe UI",9,"bold"), relief="flat")
        sty.map("D.Treeview",
                background=[("selected", C["acc_bg"])],
                foreground=[("selected", C["white"])])

        cols = ("n","xn","fxn","err")
        self.tree = ttk.Treeview(tc.body, columns=cols, show="headings",
                                  height=5, style="D.Treeview")
        for col, w, title in [("n",55,"n"),("xn",210,"x_n"),
                               ("fxn",210,"f(x_n)"),("err",160,"Erreur")]:
            self.tree.heading(col, text=title)
            self.tree.column(col, width=w, anchor="center")
        self.tree.pack(fill="x")

        btns = tk.Frame(tc.body, bg=C["card"], pady=8)
        btns.pack(fill="x")
        mk_btn(btns, " Exporter CSV",  self._export_csv,  secondary=True).pack(side="left", padx=(0,8))
       
        mk_btn(btns, " Exporter JSON", self._export_json, secondary=True).pack(side="left")

        tk.Frame(content, bg=C["bg"], height=20).pack()

        
        
        self._update_rec() 

    def _style_ax(self, ax):
        ax.tick_params(colors=C["gray"], labelsize=8)
        for sp_ in ax.spines.values(): sp_.set_edgecolor(C["border"])
        ax.set_xlabel("x", color=C["gray"], fontsize=9)
        ax.set_ylabel("f(x)", color=C["gray"], fontsize=9)

    def _sel_algo(self, name):
        self.algo_var.set(name)
        for n, f in self._algo_frames.items():
            is_sel = (n == name)
            bg = C["acc_bg"] if is_sel else C["card2"]
            brd = C["accent"] if is_sel else C["border"]
            f.configure(bg=bg, highlightbackground=brd)
            for w in f.winfo_children(): w.configure(bg=bg)
        

    def _update_rec(self):
      if not self._initialized:
        self._rec.configure(text="Veuillez entrer des parametres valides pour obtenir une recommandation.")
        return

      if not any([self._dicho_success, self._pf_success, self._newton_success]):
        self._rec.configure(text="Aucune methode applicable sur cet intervalle.")
        return

      _, message = recommander_methode(
        self._dicho_success, self._dicho_sol,
        self._pf_success,    self._pf_sol,    self._pf_phi,
        self._newton_success, self._newton_sol
    )
      self._rec.configure(text=message)

    def _on_input_change(self, event=None):
      try:
        
        f_str = self.e_fx.get().strip().replace('^', '**')
        a     = float(self.e_a.get())
        b     = float(self.e_b.get())
        x0    = float(self.e_x0.get())
        
        tol   = float(self.e_tol.get())
        nmax  = 200
        self._initialized = True

       

        # Dichotomie
        ok_d, sol_d, _, _ = dichotomie(f_str, a, b, tol, nmax)
        self._dicho_success = ok_d
        self._dicho_sol     = sol_d

        # Point fixe
        ok_pf, phi, sol_pf, _, _, _ = point_fixe(f_str, a, b, x0, tol, nmax)
        self._pf_success = ok_pf
        self._pf_sol     = sol_pf
        self._pf_phi     = phi

        # Newton
        ok_n, sol_n, _, _, *_ = newton_(f_str, a, b, x0, tol, nmax) 
        self._newton_success = ok_n
        self._newton_sol     = sol_n

      except:
        self._dicho_success  = False
        self._dicho_sol      = None
        self._pf_success     = False
        self._pf_sol         = None
        self._pf_phi         = None
        self._newton_success = False
        self._newton_sol     = None
        

      self._update_rec()



    def _run(self):
      algo = self.algo_var.get()
      f_str = self.e_fx.get().strip().replace('^', '**')

      try:
          tol  = float(self.e_tol.get())
          a    = float(self.e_a.get())
          b    = float(self.e_b.get())
          x0   = float(self.e_x0.get())
          nmax = 200
      except ValueError as e:
          messagebox.showerror("Erreur", f"Paramètre invalide : {e}"); return

      root_val = None
      iters_table = []   # liste de (n, xn, fxn, err)

      try:
          x_sym = sp.Symbol('x')
          f_sym = sp.sympify(f_str)
          f_num = sp.lambdify(x_sym, f_sym, 'numpy')

        # ── DICHOTOMIE ──────────────────────────────────────
          if algo == "Dichotomie":
              ok, sol, iters, erreurs = dichotomie(f_str, a, b, tol, nmax)
              if not ok:
                  messagebox.showwarning("Dichotomie",
                    "f(a) et f(b) ont le même signe — pas de racine garantie.")
                  return
              root_val = sol
              for i, (xn, err) in enumerate(zip(iters, erreurs)):
                  iters_table.append((i, xn, float(f_num(xn)), float(err)))

        # ── NEWTON ──────────────────────────────────────────
          elif algo == "Newton":
              ok, sol, iters, erreurs, _, _, _ = newton_(f_str, a, b, x0, tol, nmax)
              if not ok:
                  messagebox.showwarning("Newton",
                    "Conditions non vérifiées — Newton non applicable.")
                  return
              root_val = sol
              for i, (xn, err) in enumerate(zip(iters, erreurs)):
                  iters_table.append((i, xn, float(f_num(xn)), float(err)))

        # ── POINT FIXE ──────────────────────────────────────
          elif algo == "Point fixe":
              ok, phi, sol, iters, erreurs, rapport = point_fixe(
                  f_str, a, b, x0, tol, nmax)
              if not ok:
                  msg = ("Aucune φ(x) stable et contractante trouvée sur [a,b].\n"
                       "Point fixe non applicable.\n" 
                       "Voulez-vous essayer avec relaxation / Newton ?")
                  choix=messagebox.askyesno("Point fixe", msg)
                  if choix:
                  #Relancer avec la fonction relaxation/newton
                    ok, phi, sol, iters, erreurs, rapport = point_fixe_avec_relaxation(
                    f_str, a, b, x0, tol, nmax
                    )

                    if not ok:
                      messagebox.showerror(
                    "Échec",
                    "Même avec relaxation/Newton, aucune convergence trouvée."
                )
                    return
                  else:
                    return



                  
              root_val = sol
              for i, (xn, err) in enumerate(zip(iters, erreurs)):
                  iters_table.append((i, xn, float(f_num(xn)), float(err)))

      except Exception as e:
          messagebox.showerror("Erreur", str(e)); return

    # ── AFFICHAGE ───────────────────────────────────────
      self._iters = iters_table

      if root_val is not None:
          self.st_r.set(f"{root_val:.6f}")
          self.st_i.set(str(len(iters_table)))
          last_err = iters_table[-1][3] if iters_table else 0
          self.st_e.set(f"{last_err:.2e}")

      for row in self.tree.get_children():
          self.tree.delete(row)
      for it in iters_table[:100]:
          self.tree.insert("", "end", values=(
              it[0],
              f"{it[1]:.10f}",
              f"{it[2]:.6e}",
              f"{it[3]:.2e}"))
          
      self.tree.configure(height=len(iters_table))

    # ── GRAPHE ──────────────────────────────────────────
      self.ax.cla(); self._style_ax(self.ax)
      try:
          margin = 2.5
          center = root_val if root_val is not None else (a + b) / 2
          xs = np.linspace(center - margin, center + margin, 600)
          ys = np.clip(f_num(xs), -50, 50)
          self.ax.plot(xs, ys, color=C["acc_light"], lw=2, label="f(x)")
          self.ax.axhline(0, color=C["muted"], lw=0.7, ls="--")
          if root_val is not None:
              self.ax.axvline(root_val, color=C["teal"], lw=1.2, ls="--",
                            alpha=0.8, label=f"x = {root_val:.4f}")
              self.ax.scatter([root_val], [0],
                            color=C["teal"], s=60, zorder=6)
              self.ax.text(root_val + 0.05, max(ys) * 0.12,
                         f"x = {root_val:.2f}",
                         color=C["teal"], fontsize=8)
          if len(iters_table) > 1:
              xi_pts = [it[1] for it in iters_table]
              yi_pts = np.clip([it[2] for it in iters_table], -50, 50)
              self.ax.scatter(xi_pts, yi_pts,
                            color=C["accent"], s=18, zorder=5, alpha=0.7)
          self.ax.legend(facecolor=C["card"], edgecolor=C["border"],
                       labelcolor=C["white"], fontsize=8)
      except:
          pass
      self.fig.tight_layout(pad=0.8)
      self.canvas.draw()

    
    def _export_csv(self):
        if not self._iters: messagebox.showinfo("Info","Aucun resultat."); return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV","*.csv")],
            initialfile="resultats_axe1.csv")
        if not path: return
        with open(path,'w',newline='',encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(["Algorithme", self.algo_var.get()])
            w.writerow(["f(x)", self.e_fx.get()])
            w.writerow([])
            w.writerow(["Iteration","x_n","f(x_n)","Erreur"])
            for it in self._iters:
                w.writerow([it[0],f"{it[1]:.12f}",f"{it[2]:.12e}",f"{it[3]:.12e}"])
        messagebox.showinfo("Exporte", f"Sauvegarde :\n{path}")

    def _export_json(self):
        if not self._iters: messagebox.showinfo("Info","Aucun resultat."); return
        path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON","*.json")],
            initialfile="resultats_axe1.json")
        if not path: return
        data = {"algorithme": self.algo_var.get(), "f(x)": self.e_fx.get(),
                "iterations": [{"n":it[0],"x":it[1],"fx":it[2],"err":it[3]}
                                for it in self._iters]}
        with open(path,'w',encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        messagebox.showinfo("Exporte", f"JSON sauvegarde :\n{path}")

    
# ══════════════════════════════════════════════
#  AXE 2  placeholder
# ══════════════════════════════════════════════

class Axe2Frame(tk.Frame):
    def __init__(self, parent):
      super().__init__(parent, bg=C["bg"])
      self._result_data = []   # pour export
      self._n = 3
      self._matrix_entries = []
      self._b_entries = []
      self._x0_entries = []
      self._build()

    # ──────────────────────────────────────────
    def _build(self):
        # ── Top bar ──
        outer = tk.Canvas(self, bg=C["bg"], bd=0, highlightthickness=0)
        self._content = tk.Frame(outer, bg=C["bg"])
        self._iter_param_card = Card(self._content, "Paramètres itératifs")
        self._x0_frame = tk.Frame(self._iter_param_card.body, bg=C["card"])
        top = tk.Frame(self, bg=C["bg"], padx=28, pady=18)

        top.pack(fill="x")
        tk.Label(top, text="Axe 2 — Résolution de systèmes linéaires",
                 font=F["title"], bg=C["bg"], fg=C["white"]).pack(side="left")
        tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

        # ── Scrollable canvas ──
       
        sb = ttk.Scrollbar(self, orient="vertical", command=outer.yview)
        outer.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        outer.pack(fill="both", expand=True)
        
        win = outer.create_window((0, 0), window=self._content, anchor="nw")
        outer.bind("<Configure>", lambda e: outer.itemconfig(win, width=e.width))
        self._content.bind("<Configure>",
                           lambda e: outer.configure(scrollregion=outer.bbox("all")))

        p = dict(padx=28, pady=8)

        # ── Catégorie : directe / itérative ──
        cat_card = Card(self._content, "Catégorie de méthode")
        cat_card.pack(fill="x", **p)
        cat_row = tk.Frame(cat_card.body, bg=C["card"])
        cat_row.pack(fill="x")
        cat_row.columnconfigure((0, 1), weight=1)

        self._cat_var = tk.StringVar(value="Directe")
        self._cat_frames = {}
        cats = [
            ("Directe",    "Gauss, LU, Cholesky",         0),
            ("Itérative",  "Jacobi, Gauss-Seidel",         1),
        ]
        for name, desc, c in cats:
            f = tk.Frame(cat_row, bg=C["card2"],
                         highlightthickness=1,
                         highlightbackground=C["border"],
                         padx=14, pady=12, cursor="hand2")
            f.grid(row=0, column=c, padx=5, pady=5, sticky="ew")
            nl = tk.Label(f, text=name, font=F["h3"], bg=C["card2"], fg=C["white"])
            nl.pack(anchor="w")
            dl = tk.Label(f, text=desc, font=F["small"], bg=C["card2"], fg=C["gray"])
            dl.pack(anchor="w")
            for w in [f, nl, dl]:
                w.bind("<Button-1>", lambda e, n=name: self._sel_cat(n))
            self._cat_frames[name] = f

        # ── Algorithme ──
        algo_card = Card(self._content, "Choisir un algorithme")
        algo_card.pack(fill="x", **p)
        self._algo_row = tk.Frame(algo_card.body, bg=C["card"])
        self._algo_row.pack(fill="x")
        self._algo_row.columnconfigure((0, 1, 2), weight=1)

        self._algo_var = tk.StringVar(value="Gauss (pivot partiel)")
        self._algo_frames = {}

        self._direct_algos = [
            ("Gauss (pivot partiel)",  "Stabilité numérique standard",       0),
            ("Gauss (pivot total)",    "Stabilité maximale",                  1),
            ("Décomposition LU",       "Réutilisable pour plusieurs b",       2),
            ("Cholesky",               "Pour matrices SDP uniquement",         3),
        ]
        self._iter_algos = [
            ("Jacobi",        "Convergence si diag. dominante", 0),
            ("Gauss-Seidel",  "Convergence plus rapide",        1),
        ]

        self._sel_cat("Directe")   # construit les boutons initiaux

        
         # ── Recommandation ──
        rec = tk.Frame(self._content, bg=C["teal_bg"],
                       highlightthickness=1, highlightbackground=C["teal"],
                       padx=16, pady=12)
        rec.pack(fill="x", **p)
        tk.Label(rec, text="Recommandation", font=F["h3"],
                 bg=C["teal_bg"], fg=C["teal"]).pack(anchor="w")
        self._rec = tk.Label(rec, text="", font=F["body"],
                             bg=C["teal_bg"], fg="#b0e8d5",
                             wraplength=900, justify="left")
        self._rec.pack(anchor="w", pady=(4,0))

        self._sel_algo("Newton")

        # ── Taille de la matrice ──
        sz_card = Card(self._content, "Taille du système")
        sz_card.pack(fill="x", **p)
        sz_row = tk.Frame(sz_card.body, bg=C["card"])
        sz_row.pack(anchor="w")
        tk.Label(sz_row, text="n =", font=F["small"],
                 bg=C["card"], fg=C["gray"]).pack(side="left")
        self._n_var = tk.IntVar(value=3)
        for n in [2, 3, 4, 5]:
            rb = tk.Radiobutton(sz_row, text=str(n), variable=self._n_var, value=n,
                                font=F["body"], bg=C["card"], fg=C["white"],
                                activebackground=C["card"], activeforeground=C["acc_light"],
                                selectcolor=C["acc_bg"],
                                indicatoron=0,
                                relief="flat", bd=0,
                                highlightthickness=1,
                                highlightbackground=C["border"],
                                cursor="hand2", padx=12, pady=4,
                                command=self._rebuild_matrix)
            rb.pack(side="left", padx=4)
        self._n = 3


        # ── Saisie de la matrice ──
        self._mat_card = Card(self._content, "Matrice A et vecteur b")
        self._mat_card.pack(fill="x", **p)
        self._mat_body = self._mat_card.body
        self._rebuild_matrix()

        # ── Paramètres itératifs (cachés par défaut) ──
        
        self._iter_param_card.pack(fill="x", **p)
        iter_row = tk.Frame(self._iter_param_card.body, bg=C["card"])
        iter_row.pack(fill="x")

        for lbl, attr, default, width in [
            ("Tolérance", "e_tol", "1e-6", 10),
            ("Itér. max", "e_nmax", "100",  7),
        ]:
            col = tk.Frame(iter_row, bg=C["card"])
            col.pack(side="left", padx=(0, 12))
            tk.Label(col, text=lbl, font=F["small"],
                     bg=C["card"], fg=C["gray"]).pack(anchor="w", pady=(0, 3))
            e = mk_entry(col, default, width)
            e.pack(ipady=5)
            setattr(self, attr, e)

        # x0 section
        
        self._x0_frame.pack(fill="x", pady=(8, 0))
        tk.Label(self._x0_frame, text="Point de départ x₀ :",
                 font=F["small"], bg=C["card"], fg=C["gray"]).pack(side="left", padx=(0, 8))
        self._x0_entries = []
        self._rebuild_x0()
        self._iter_param_card.pack_forget()   # caché en mode direct

        # ── Bouton Calculer ──
        btn_row = tk.Frame(self._content, bg=C["bg"], padx=28, pady=4)
        btn_row.pack(fill="x")
        mk_btn(btn_row, "  Calculer  ", self._run2, color=C["accent"]).pack(side="left")
        mk_btn(btn_row, "  Réinitialiser  ", self._reset_matrix,
               secondary=True).pack(side="left", padx=10)

        # ── Stats ──
        sr = tk.Frame(self._content, bg=C["bg"])
        sr.pack(fill="x", **p)
        sr.columnconfigure((0, 1, 2, 3), weight=1)
        self.st_norm  = StatCard(sr, "‖Ax − b‖")
        self.st_norm.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.st_iter  = StatCard(sr, "Itérations")
        self.st_iter.grid(row=0, column=1, sticky="ew", padx=(0, 8))
        self.st_rho   = StatCard(sr, "Rayon spectral ρ")
        self.st_rho.grid(row=0, column=2, sticky="ew", padx=(0, 8))
        self.st_conv  = StatCard(sr, "Convergence")
        self.st_conv.grid(row=0, column=3, sticky="ew")

        # ── Résultat : vecteur solution ──
        sol_card = Card(self._content, "Vecteur solution x")
        sol_card.pack(fill="x", **p)
        self._sol_frame = tk.Frame(sol_card.body, bg=C["card"])
        self._sol_frame.pack(fill="x")
        tk.Label(self._sol_frame, text="—", font=F["mono"],
                 bg=C["card"], fg=C["gray"]).pack(anchor="w")

        # ── Graphe convergence ──
        gc = Card(self._content, "Convergence des erreurs")
        gc.pack(fill="x", **p)
        self._fig, self._ax = plt.subplots(figsize=(9, 2.8), facecolor=C["card"])
        self._ax.set_facecolor(C["bg"])
        self._style_ax(self._ax)
        self._canvas = FigureCanvasTkAgg(self._fig, gc.body)
        self._canvas.get_tk_widget().pack(fill="both")

        # ── Tableau d'itérations / étapes ──
        tc = Card(self._content, "Détail des étapes / itérations")
        tc.pack(fill="x", **p)

        sty = ttk.Style()
        sty.theme_use("clam")
        sty.configure("D.Treeview",
                       background=C["card"], foreground=C["white"],
                       fieldbackground=C["card"], rowheight=24, font=F["mono"])
        sty.configure("D.Treeview.Heading",
                       background=C["card2"], foreground=C["acc_light"],
                       font=("Segoe UI", 9, "bold"), relief="flat")
        sty.map("D.Treeview",
                background=[("selected", C["acc_bg"])],
                foreground=[("selected", C["white"])])

        self._tree_frame = tc.body
        self._tree = None   # sera créé dynamiquement selon le mode

        btns = tk.Frame(tc.body, bg=C["card"], pady=8)
        btns.pack(fill="x")
        mk_btn(btns, " Exporter CSV",  self._export_csv2,  secondary=True).pack(side="left", padx=(0, 8))
        mk_btn(btns, " Exporter JSON", self._export_json2, secondary=True).pack(side="left")

        tk.Frame(self._content, bg=C["bg"], height=20).pack()

    # ──────────────────────────────────────────
    def _style_ax(self, ax):
        ax.tick_params(colors=C["gray"], labelsize=8)
        for sp_ in ax.spines.values():
            sp_.set_edgecolor(C["border"])
        ax.set_xlabel("Itération", color=C["gray"], fontsize=9)
        ax.set_ylabel("Erreur (‖·‖∞)", color=C["gray"], fontsize=9)

    # ──────────────────────────────────────────
    def _sel_cat(self, name):
        self._cat_var.set(name)
        for n, f in self._cat_frames.items():
            is_sel = (n == name)
            bg = C["acc_bg"] if is_sel else C["card2"]
            brd = C["accent"] if is_sel else C["border"]
            f.configure(bg=bg, highlightbackground=brd)
            for w in f.winfo_children():
                w.configure(bg=bg)

        # Reconstruit les boutons algo
        for w in self._algo_row.winfo_children():
            w.destroy()
        self._algo_frames.clear()

        algos = self._direct_algos if name == "Directe" else self._iter_algos
        default = algos[0][0]
        self._algo_var.set(default)

        cols = len(algos)
        for c in range(cols):
            self._algo_row.columnconfigure(c, weight=1)

        for aname, desc, col in algos:
            f = tk.Frame(self._algo_row, bg=C["card2"],
                         highlightthickness=1,
                         highlightbackground=C["border"],
                         padx=14, pady=12, cursor="hand2")
            f.grid(row=0, column=col, padx=5, pady=5, sticky="ew")
            nl = tk.Label(f, text=aname, font=F["h3"], bg=C["card2"], fg=C["white"])
            nl.pack(anchor="w")
            dl = tk.Label(f, text=desc, font=F["small"], bg=C["card2"], fg=C["gray"])
            dl.pack(anchor="w")
            for w in [f, nl, dl]:
                w.bind("<Button-1>", lambda e, n=aname: self._sel_algo(n))
            self._algo_frames[aname] = f

        self._sel_algo(default)

        # Affiche/cache les paramètres itératifs
        if name == "Itérative":
            self._iter_param_card.pack(
                fill="x", padx=28, pady=8,
                before=self._mat_card if hasattr(self, '_mat_card') else None)
            # pack après matrice
            self._iter_param_card.pack(fill="x", padx=28, pady=8)
        else:
            self._iter_param_card.pack_forget()

    
    def _sel_algo(self, name):
        self._algo_var.set(name)
        for n, f in self._algo_frames.items():
            is_sel = (n == name)
            bg = C["acc_bg"] if is_sel else C["card2"]
            brd = C["accent"] if is_sel else C["border"]
            f.configure(bg=bg, highlightbackground=brd)
            for w in f.winfo_children():
                w.configure(bg=bg)
    # ──────────────────────────────────────────
    def _rebuild_matrix(self):
        self._n = self._n_var.get()
        for w in self._mat_body.winfo_children():
            w.destroy()

        header = tk.Frame(self._mat_body, bg=C["card"])
        header.pack(fill="x", pady=(0, 6))
        # colonnes A
        for j in range(self._n):
            tk.Label(header, text=f"  x{j+1}",
                     font=F["small"], bg=C["card"], fg=C["acc_light"],
                     width=8, anchor="center").grid(row=0, column=j, padx=2)
        tk.Label(header, text="  |  b",
                 font=F["small"], bg=C["card"], fg=C["teal"],
                 width=10, anchor="center").grid(row=0, column=self._n, padx=2)

        self._matrix_entries = []
        self._b_entries = []
        for i in range(self._n):
            row_f = tk.Frame(self._mat_body, bg=C["card"])
            row_f.pack(fill="x", pady=2)
            row_ents = []
            for j in range(self._n):
                default = "1" if i == j else "0"
                e = mk_entry(row_f, default, 8)
                e.grid(row=0, column=j, padx=2)
                row_ents.append(e)
            self._matrix_entries.append(row_ents)
            # séparateur
            tk.Label(row_f, text="│", font=F["mono"],
                     bg=C["card"], fg=C["muted"]).grid(row=0, column=self._n, padx=4)
            be = mk_entry(row_f, "0", 8)
            be.grid(row=0, column=self._n + 1, padx=2)
            self._b_entries.append(be)

        self._rebuild_x0()

    # ──────────────────────────────────────────
    def _rebuild_x0(self):
        for w in self._x0_frame.winfo_children():
            if isinstance(w, tk.Entry):
                w.destroy()
        self._x0_entries = []
        for i in range(self._n):
            e = mk_entry(self._x0_frame, "0", 6)
            e.pack(side="left", padx=3)
            self._x0_entries.append(e)

    # ──────────────────────────────────────────
    def _reset_matrix(self):
        for i in range(self._n):
            for j in range(self._n):
                self._matrix_entries[i][j].delete(0, "end")
                self._matrix_entries[i][j].insert(0, "1" if i == j else "0")
            self._b_entries[i].delete(0, "end")
            self._b_entries[i].insert(0, "0")

    # ──────────────────────────────────────────
    def _read_matrix(self):
        n = self._n
        A = np.zeros((n, n))
        b = np.zeros(n)
        for i in range(n):
            for j in range(n):
                A[i, j] = float(self._matrix_entries[i][j].get())
            b[i] = float(self._b_entries[i].get())
        return A, b

    # ──────────────────────────────────────────
    def _run2(self):
        algo = self._algo_var.get()
        cat  = self._cat_var.get()

        try:
            A, b = self._read_matrix()
        except ValueError as e:
            messagebox.showerror("Erreur de saisie", f"Valeur invalide dans la matrice:\n{e}")
            return

        x_sol = None
        steps_or_iters = []
        errors = []
        rho_val = None
        converged = True
        L_mat = U_mat = None

        try:
            if algo == "Gauss (pivot partiel)":
                x_sol, steps_or_iters = gauss_pivot_partiel(A, b)

            elif algo == "Gauss (pivot total)":
                x_sol, steps_or_iters = gauss_pivot_total(A, b)

            elif algo == "Décomposition LU":
                x_sol, L_mat, U_mat, steps_or_iters = decomp_lu(A, b)

            elif algo == "Cholesky":
                # Vérifie symétrie
                if not np.allclose(A, A.T):
                    if messagebox.askyesno("Cholesky",
                            "La matrice n'est pas symétrique.\n"
                            "Utiliser A' = (A + Aᵀ)/2 ?"):
                        A = (A + A.T) / 2
                    else:
                        return
                x_sol, L_mat, steps_or_iters = decomp_cholesky(A, b)

            elif algo == "Jacobi":
                tol  = float(self.e_tol.get())
                nmax = int(self.e_nmax.get())
                x0   = np.array([float(e.get()) for e in self._x0_entries])
                converged, x_sol, iters_list, errors, rho_val = jacobi(A, b, x0, tol, nmax)
                steps_or_iters = iters_list
                if not converged:
                    messagebox.showwarning("Jacobi",
                        f"Pas de convergence en {nmax} itérations.\n"
                        f"Rayon spectral ρ ≈ {rho_val:.4f} (> 1 → diverge).")

            elif algo == "Gauss-Seidel":
                tol  = float(self.e_tol.get())
                nmax = int(self.e_nmax.get())
                x0   = np.array([float(e.get()) for e in self._x0_entries])
                converged, x_sol, iters_list, errors, rho_val = gauss_seidel(A, b, x0, tol, nmax)
                steps_or_iters = iters_list
                if not converged:
                    messagebox.showwarning("Gauss-Seidel",
                        f"Pas de convergence en {nmax} itérations.\n"
                        f"Rayon spectral ρ ≈ {rho_val:.4f} (> 1 → diverge).")

        except ValueError as e:
            messagebox.showerror("Erreur", str(e))
            return

        # ── Stats ──
        residual = np.linalg.norm(A @ x_sol - b)
        self.st_norm.set(f"{residual:.2e}")

        if cat == "Itérative":
            self.st_iter.set(str(len(steps_or_iters)))
            self.st_rho.set(f"{rho_val:.4f}" if rho_val is not None else "—")
            self.st_conv.set("✓ Oui" if converged else "✗ Non")
        else:
            self.st_iter.set(f"{len(steps_or_iters)} ops")
            self.st_rho.set("—")
            self.st_conv.set("✓ Direct")

        # ── Vecteur solution ──
        for w in self._sol_frame.winfo_children():
            w.destroy()
        for i, xi in enumerate(x_sol):
            row = tk.Frame(self._sol_frame, bg=C["card"])
            row.pack(anchor="w")
            tk.Label(row, text=f"  x{i+1} =", font=F["mono"],
                     bg=C["card"], fg=C["gray"], width=6).pack(side="left")
            color = C["teal"] if abs(xi) > 1e-10 else C["muted"]
            tk.Label(row, text=f"{xi:.10f}", font=F["mono"],
                     bg=C["card"], fg=color).pack(side="left", padx=6)

        # ── LU / Cholesky : affiche les matrices ──
        if L_mat is not None:
            self._show_matrix_popup(algo, L_mat, U_mat)

        # ── Graphe ──
        self._ax.cla()
        self._style_ax(self._ax)
        if cat == "Itérative" and errors:
            self._ax.semilogy(range(len(errors)), errors,
                              color=C["acc_light"], lw=2, marker="o",
                              markersize=3, label="Erreur ‖·‖∞")
            self._ax.axhline(float(self.e_tol.get()),
                             color=C["teal"], lw=1, ls="--", alpha=0.7,
                             label=f"tol = {self.e_tol.get()}")
            self._ax.legend(facecolor=C["card"], edgecolor=C["border"],
                            labelcolor=C["white"], fontsize=8)
            self._fig.tight_layout(pad=0.8)
        else:
            # Méthodes directes : bar chart des résidus par composante
            residuals_comp = np.abs(A @ x_sol - b)
            bars = self._ax.bar(range(len(residuals_comp)), residuals_comp,
                                color=C["accent"], alpha=0.8)
            self._ax.set_xlabel("Composante", color=C["gray"], fontsize=9)
            self._ax.set_ylabel("|Ax−b|ᵢ", color=C["gray"], fontsize=9)
            self._ax.set_xticks(range(len(residuals_comp)))
            self._ax.set_xticklabels([f"r{i+1}" for i in range(len(residuals_comp))],
                                      color=C["gray"], fontsize=8)
        self._canvas.draw()

        # ── Tableau ──
        self._build_tree(cat, steps_or_iters)

        # ── Pour export ──
        self._result_data = {
            "algorithme": algo,
            "solution": x_sol.tolist(),
            "residuel": float(residual),
            "converged": converged,
            "details": [],
        }
        if cat == "Itérative":
            for k, (xk, err) in enumerate(zip(steps_or_iters, errors)):
                self._result_data["details"].append({
                    "iter": k, "x": xk.tolist(), "err": float(err)})
        else:
            for k, s in enumerate(steps_or_iters):
                self._result_data["details"].append({"etape": k, "op": s})

    # ──────────────────────────────────────────
    def _build_tree(self, cat, data):
        if self._tree:
            self._tree.destroy()
            self._tree = None

        if cat == "Itérative":
            n = self._n
            cols = ("k",) + tuple(f"x{i+1}" for i in range(n)) + ("err",)
            tree = ttk.Treeview(self._tree_frame, columns=cols,
                                show="headings", height=6, style="D.Treeview")
            tree.heading("k", text="k")
            tree.column("k", width=45, anchor="center")
            for i in range(n):
                tree.heading(f"x{i+1}", text=f"x{i+1}")
                tree.column(f"x{i+1}", width=130, anchor="center")
            tree.heading("err", text="Erreur ‖·‖∞")
            tree.column("err", width=120, anchor="center")

            errors = []
            for k, xk in enumerate(data):
                if k > 0:
                    err = np.linalg.norm(xk - data[k - 1], np.inf)
                else:
                    err = float('inf')
                errors.append(err)
                vals = (k,) + tuple(f"{v:.8f}" for v in xk) + (f"{err:.2e}",)
                tree.insert("", "end", values=vals)

        else:
            cols = ("n", "operation")
            tree = ttk.Treeview(self._tree_frame, columns=cols,
                                show="headings", height=6, style="D.Treeview")
            tree.heading("n", text="#")
            tree.column("n", width=45, anchor="center")
            tree.heading("operation", text="Opération effectuée")
            tree.column("operation", width=600, anchor="w")
            for k, s in enumerate(data):
                tree.insert("", "end", values=(k + 1, s))

        tree.pack(fill="x")
        self._tree = tree

    # ──────────────────────────────────────────
    def _show_matrix_popup(self, algo, L, U):
        """Affiche L (et U si LU) dans une fenêtre modale."""
        top = tk.Toplevel(self)
        top.title(f"Matrices — {algo}")
        top.configure(bg=C["bg"])
        top.geometry("520x360")

        tk.Label(top, text=f"Résultat de la décomposition — {algo}",
                 font=F["h2"], bg=C["bg"], fg=C["white"],
                 pady=12).pack(anchor="w", padx=20)
        tk.Frame(top, bg=C["border"], height=1).pack(fill="x")

        frame = tk.Frame(top, bg=C["bg"], padx=20, pady=14)
        frame.pack(fill="both", expand=True)

        def show_mat(parent, mat, label, color):
            tk.Label(parent, text=label, font=F["h3"],
                     bg=C["bg"], fg=color).pack(anchor="w", pady=(8, 2))
            for row in mat:
                tk.Label(parent,
                         text="  ".join(f"{v:10.4f}" for v in row),
                         font=F["mono"], bg=C["bg"], fg=C["white"]).pack(anchor="w")

        show_mat(frame, L, "Matrice L" if U is None else "L (triangulaire inférieure)", C["teal"])
        if U is not None:
            show_mat(frame, U, "U (triangulaire supérieure)", C["acc_light"])

        mk_btn(top, "Fermer", top.destroy, secondary=True).pack(pady=10)

    # ──────────────────────────────────────────
    def _export_csv2(self):
        if not self._result_data:
            messagebox.showinfo("Info", "Aucun résultat.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="resultats_axe2.csv")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Algorithme", self._result_data["algorithme"]])
            w.writerow(["Résidu ‖Ax−b‖", f"{self._result_data['residuel']:.6e}"])
            w.writerow([])
            # Solution
            w.writerow(["Solution"])
            for i, xi in enumerate(self._result_data["solution"]):
                w.writerow([f"x{i+1}", f"{xi:.12f}"])
            w.writerow([])
            # Détails
            if self._cat_var.get() == "Itérative":
                w.writerow(["Itération"] +
                           [f"x{i+1}" for i in range(len(self._result_data["solution"]))] +
                           ["Erreur"])
                for d in self._result_data["details"]:
                    w.writerow([d["iter"]] +
                               [f"{v:.12f}" for v in d["x"]] +
                               [f"{d['err']:.6e}"])
            else:
                w.writerow(["#", "Opération"])
                for d in self._result_data["details"]:
                    w.writerow([d["etape"] + 1, d["op"]])
        messagebox.showinfo("Exporté", f"Fichier sauvegardé :\n{path}")

    def _export_json2(self):
        if not self._result_data:
            messagebox.showinfo("Info", "Aucun résultat.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile="resultats_axe2.json")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._result_data, f, indent=2, ensure_ascii=False)
        messagebox.showinfo("Exporté", f"JSON sauvegardé :\n{path}")



        

# ══════════════════════════════════════════════
#  AXE 3
# ══════════════════════════════════════════════

class Axe3Frame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=C["bg"])
        self._exp_data = {}
        self._build()

    def _build(self):
        top = tk.Frame(self, bg=C["bg"], padx=28, pady=18)
        top.pack(fill="x")
        tk.Label(top, text="Axe 3 — Interpolation & Approximation",
                 font=F["title"], bg=C["bg"], fg=C["white"]).pack(side="left")
        tk.Label(top, text=" Etudiant 3 ", font=F["small"],
                 bg="#2a1f0a", fg=C["amber"], padx=8, pady=4).pack(side="right", anchor="n")
        tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

        outer = tk.Canvas(self, bg=C["bg"], bd=0, highlightthickness=0)
        sb = ttk.Scrollbar(self, orient="vertical", command=outer.yview)
        outer.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        outer.pack(fill="both", expand=True)
        content = tk.Frame(outer, bg=C["bg"])
        win = outer.create_window((0,0), window=content, anchor="nw")
        outer.bind("<Configure>", lambda e: outer.itemconfig(win, width=e.width))
        content.bind("<Configure>", lambda e: outer.configure(
            scrollregion=outer.bbox("all")))

        p = dict(padx=28, pady=8)

        # Methode
        mc = Card(content, "Methode")
        mc.pack(fill="x", **p)
        grid = tk.Frame(mc.body, bg=C["card"])
        grid.pack(fill="x")
        grid.columnconfigure((0,1), weight=1)

        self.method_var = tk.StringVar(value="Lagrange")
        self._meth_frames = {}
        methods = [
            ("Lagrange",          "Interpolation polynomiale", 0, 0),
            ("Newton diff. div.", "Interpolation Newton",      0, 1),
            ("Moindres carres",   "Approximation discrete",    1, 0),
            ("Descente gradient", "Regression lineaire",       1, 1),
        ]
        for name, desc, r, c in methods:
            f = tk.Frame(grid, bg=C["card2"],
                         highlightthickness=1, highlightbackground=C["border"],
                         padx=14, pady=12, cursor="hand2")
            f.grid(row=r, column=c, padx=5, pady=5, sticky="ew")
            nl = tk.Label(f, text=name, font=F["h3"], bg=C["card2"], fg=C["white"])
            nl.pack(anchor="w")
            dl = tk.Label(f, text=desc, font=F["small"], bg=C["card2"], fg=C["gray"])
            dl.pack(anchor="w")
            for w in [f, nl, dl]:
                w.bind("<Button-1>", lambda e, n=name: self._sel_meth(n))
            self._meth_frames[name] = f
        

        # Recommandation
        rec = tk.Frame(content, bg=C["teal_bg"],
                       highlightthickness=1, highlightbackground=C["teal"],
                       padx=16, pady=12)
        rec.pack(fill="x", **p)
        tk.Label(rec, text="Recommandation", font=F["h3"],
                 bg=C["teal_bg"], fg=C["teal"]).pack(anchor="w")
        self._rec3 = tk.Label(rec, text="", font=F["body"],
                              bg=C["teal_bg"], fg="#b0e8d5",
                              wraplength=900, justify="left")
        self._rec3.pack(anchor="w", pady=(4,0))
        self._sel_meth("Lagrange")

        # Donnees
        dc = Card(content, "Donnees")
        dc.pack(fill="x", **p)
        row = tk.Frame(dc.body, bg=C["card"])
        row.pack(fill="x")
        for lbl, attr, default, w in [
            ("x (virgule)", "e_xp", "0,1,2,3,4", 18),
            ("y (virgule)", "e_yp", "1,4,9,16,25", 18),
            ("x a evaluer", "e_xi", "2.5", 7),
            ("Degre",       "e_deg", "2", 4),
        ]:
            col = tk.Frame(row, bg=C["card"])
            col.pack(side="left", padx=(0,10))
            tk.Label(col, text=lbl, font=F["small"],
                     bg=C["card"], fg=C["gray"]).pack(anchor="w", pady=(0,3))
            e = mk_entry(col, default, w)
            e.pack(ipady=5)
            setattr(self, attr, e)
        col_b = tk.Frame(row, bg=C["card"])
        col_b.pack(side="left")
        tk.Label(col_b, text=" ", font=F["small"], bg=C["card"], fg=C["gray"]).pack(pady=(0,3))
        mk_btn(col_b, "Calculer \u2197", self._run3, C["amber"]).pack(ipady=4)

        # Stats
        sr = tk.Frame(content, bg=C["bg"])
        sr.pack(fill="x", **p)
        sr.columnconfigure((0,1,2), weight=1)
        self.st3_v = StatCard(sr, "Valeur interpolee")
        self.st3_v.grid(row=0, column=0, sticky="ew", padx=(0,8))
        self.st3_m = StatCard(sr, "Methode")
        self.st3_m.grid(row=0, column=1, sticky="ew", padx=(0,8))
        self.st3_p = StatCard(sr, "Nb points")
        self.st3_p.grid(row=0, column=2, sticky="ew")

        # Graphe
        gc = Card(content, "Visualisation")
        gc.pack(fill="x", **p)
        self.fig3, self.ax3 = plt.subplots(figsize=(9, 3.2), facecolor=C["card"])
        self.ax3.set_facecolor(C["bg"])
        for sp_ in self.ax3.spines.values(): sp_.set_edgecolor(C["border"])
        self.ax3.tick_params(colors=C["gray"], labelsize=8)
        self.canvas3 = FigureCanvasTkAgg(self.fig3, gc.body)
        self.canvas3.get_tk_widget().pack(fill="both")

        # Export/import
        ec = Card(content, "Export / Import")
        ec.pack(fill="x", **p)
        btns = tk.Frame(ec.body, bg=C["card"])
        btns.pack(fill="x")
        mk_btn(btns, "\u2b07 Exporter CSV",  self._export3, secondary=True).pack(side="left", padx=(0,8))
        mk_btn(btns, "\u2b06 Importer points CSV", self._import3, secondary=True).pack(side="left")

        tk.Frame(content, bg=C["bg"], height=20).pack()

    def _sel_meth(self, name):
        self.method_var.set(name)
        for n, f in self._meth_frames.items():
            is_sel = (n == name)
            bg = C["acc_bg"] if is_sel else C["card2"]
            brd = C["accent"] if is_sel else C["border"]
            f.configure(bg=bg, highlightbackground=brd)
            for w in f.winfo_children(): w.configure(bg=bg)
        recs = {
            "Lagrange":          "Lagrange est simple mais peut osciller (phenomene de Runge) avec beaucoup de points.",
            "Newton diff. div.": "Newton par differences divisees est plus stable numeriquement.",
            "Moindres carres":   "Moindres carres minimise la somme des carres des residus. Ideal pour donnees bruitees.",
            "Descente gradient": "Descente de gradient optimise iterativement la droite de regression.",
        }
        self._rec3.configure(text=recs.get(name, ""))

    def _run3(self):
        try:
            xp = np.array([float(v) for v in self.e_xp.get().split(",")])
            yp = np.array([float(v) for v in self.e_yp.get().split(",")])
            xi = float(self.e_xi.get())
            deg = int(self.e_deg.get())
        except Exception as e:
            messagebox.showerror("Erreur", str(e)); return

        method = self.method_var.get()
        self.ax3.cla()
        self.ax3.set_facecolor(C["bg"])
        for sp_ in self.ax3.spines.values(): sp_.set_edgecolor(C["border"])
        self.ax3.tick_params(colors=C["gray"], labelsize=8)

        x_d = np.linspace(min(xp)-0.5, max(xp)+0.5, 400)
        val = None

        if method == "Lagrange":
            y_d = [lagrange_eval(xp, yp, xi_) for xi_ in x_d]
            val = lagrange_eval(xp, yp, xi)
            self.ax3.plot(x_d, y_d, color=C["acc_light"], lw=2, label="Lagrange")
        elif method == "Newton diff. div.":
            y_d = [newton_interp_eval(xp, yp, xi_) for xi_ in x_d]
            val = newton_interp_eval(xp, yp, xi)
            self.ax3.plot(x_d, y_d, color=C["acc_light"], lw=2, label="Newton")
        elif method == "Moindres carres":
            poly = mc_discret(xp, yp, deg)
            y_d = poly(x_d); val = poly(xi)
            self.ax3.plot(x_d, y_d, color=C["acc_light"], lw=2, label=f"MC deg {deg}")
        elif method == "Descente gradient":
            m, b = descente_gradient_lin(xp, yp)
            y_d = m*x_d+b; val = m*xi+b
            self.ax3.plot(x_d, y_d, color=C["acc_light"], lw=2, label=f"y={m:.3f}x+{b:.3f}")

        self.ax3.scatter(xp, yp, color=C["amber"], zorder=5, s=55, label="Points")
        if val is not None:
            self.ax3.scatter([xi],[val], color=C["teal"], zorder=6, s=80,
                             label=f"f({xi})={val:.4f}")
        self.ax3.set_xlabel("x", color=C["gray"], fontsize=9)
        self.ax3.set_ylabel("y", color=C["gray"], fontsize=9)
        self.ax3.legend(facecolor=C["card"], edgecolor=C["border"],
                        labelcolor=C["white"], fontsize=8)
        self.fig3.tight_layout(pad=0.8)
        self.canvas3.draw()

        self.st3_v.set(f"{val:.6f}" if val is not None else "—")
        self.st3_m.set(method.split()[0])
        self.st3_p.set(str(len(xp)))
        self._exp_data = {"xp": list(xp), "yp": list(yp),
                          "xi": xi, "val": val, "method": method}

    def _export3(self):
        if not self._exp_data: messagebox.showinfo("Info","Aucun resultat."); return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV","*.csv")],
            initialfile="resultats_axe3.csv")
        if not path: return
        d = self._exp_data
        with open(path,'w',newline='',encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(["Methode", d["method"]])
            w.writerow(["x evalue", d["xi"]])
            w.writerow(["Valeur",   d["val"]])
            w.writerow([])
            w.writerow(["x_i","y_i"])
            for xi, yi in zip(d["xp"], d["yp"]):
                w.writerow([xi, yi])
        messagebox.showinfo("Exporte", f"Sauvegarde :\n{path}")

    def _import3(self):
        path = filedialog.askopenfilename(filetypes=[("CSV","*.csv"),("Tous","*.*")])
        if not path: return
        try:
            xp, yp = [], []
            with open(path,newline='',encoding='utf-8') as f:
                rows = list(csv.reader(f))
            reading = False
            for row in rows:
                if row and row[0] == "x_i": reading = True; continue
                if reading and len(row) >= 2:
                    try: xp.append(float(row[0])); yp.append(float(row[1]))
                    except: pass
            if not xp: messagebox.showerror("Erreur","Aucune donnee trouvee."); return
            self.e_xp.delete(0,"end"); self.e_xp.insert(0,",".join(str(v) for v in xp))
            self.e_yp.delete(0,"end"); self.e_yp.insert(0,",".join(str(v) for v in yp))
            messagebox.showinfo("Importe",f"{len(xp)} points charges.")
        except Exception as e:
            messagebox.showerror("Erreur import", str(e))


# ══════════════════════════════════════════════
#  APP PRINCIPALE
# ══════════════════════════════════════════════

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Analyse Numerique — USTHB 3INGSoft")
        self.geometry("1280x800")
        self.minsize(1000, 650)
        self.configure(bg=C["bg"])
        self._build()

    def _build(self):
        sidebar = tk.Frame(self, bg=C["panel"], width=230)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        logo = tk.Frame(sidebar, bg=C["panel"], padx=18, pady=22)
        logo.pack(fill="x")
        tk.Label(logo, text="Analyse\nnumerique", font=("Segoe UI",13,"bold"),
                 bg=C["panel"], fg=C["white"], justify="left").pack(anchor="w")
       
        tk.Frame(sidebar, bg=C["border"], height=1).pack(fill="x")

        nav = tk.Frame(sidebar, bg=C["panel"], pady=14)
        nav.pack(fill="x")
        tk.Label(nav, text="AXES", font=("Segoe UI",8,"bold"),
                 bg=C["panel"], fg=C["muted"], padx=18).pack(anchor="w", pady=(0,4))

        self._nav_btns = {}
        nav_items = [
            ("axe1", "  Fonctions non lineaires", C["accent"]),
            ("axe2", "  Systemes lineaires",       C["teal"]),
            ("axe3", "  Interpolation & Approx.",  C["amber"]),
        ]
        dots = {"axe1": C["accent"], "axe2": C["teal"], "axe3": C["amber"]}
        for key, text, color in nav_items:
            row = tk.Frame(nav, bg=C["panel"], cursor="hand2")
            row.pack(fill="x")
            dot = tk.Label(row, text="●", font=("Segoe UI",8),
                           bg=C["panel"], fg=color, padx=(18), pady=10)
            dot.pack(side="left")
            lbl = tk.Label(row, text=text, font=F["body"],
                           bg=C["panel"], fg=C["gray"], pady=10, anchor="w")
            lbl.pack(side="left", fill="x", expand=True)
            for w in [row, dot, lbl]:
                w.bind("<Button-1>", lambda e, k=key: self._switch(k))
                w.bind("<Enter>", lambda e, r=row, d=dot, l=lbl: [
                    r.configure(bg=C["hover"]),
                    d.configure(bg=C["hover"]),
                    l.configure(bg=C["hover"], fg=C["white"])])
                w.bind("<Leave>", lambda e, k=key, r=row, d=dot, l=lbl: self._nav_leave(k,r,d,l))
            self._nav_btns[key] = (row, dot, lbl, color)

        tk.Frame(sidebar, bg=C["border"], height=1).pack(fill="x")

        nav2 = tk.Frame(sidebar, bg=C["panel"], pady=14)
        nav2.pack(fill="x")
        tk.Label(nav2, text="OUTILS", font=("Segoe UI",8,"bold"),
                 bg=C["panel"], fg=C["muted"], padx=18).pack(anchor="w", pady=(0,4))
        for txt in ["d  Derivees & continuite", "=  Comparaison algos"]:
            l = tk.Label(nav2, text=txt, font=F["body"],
                         bg=C["panel"], fg=C["gray"], padx=18, pady=8, anchor="w", cursor="hand2")
            l.pack(fill="x")
            l.bind("<Enter>", lambda e, b=l: b.configure(bg=C["hover"], fg=C["white"]))
            l.bind("<Leave>", lambda e, b=l: b.configure(bg=C["panel"], fg=C["gray"]))

        self._main = tk.Frame(self, bg=C["bg"])
        self._main.pack(side="left", fill="both", expand=True)
        self._frames = {
            "axe1": Axe1Frame(self._main),
            "axe2": Axe2Frame(self._main),
            "axe3": Axe3Frame(self._main),
        }
        self._current = None
        self._switch("axe1")

    def _switch(self, key):
        if self._current: self._frames[self._current].pack_forget()
        self._frames[key].pack(fill="both", expand=True)
        self._current = key
        for k, (row, dot, lbl, color) in self._nav_btns.items():
            if k == key:
                row.configure(bg=C["hover"])
                dot.configure(bg=C["hover"])
                lbl.configure(bg=C["hover"], fg=C["white"])
            else:
                row.configure(bg=C["panel"])
                dot.configure(bg=C["panel"])
                lbl.configure(bg=C["panel"], fg=C["gray"])

    def _nav_leave(self, key, row, dot, lbl):
        if self._current == key:
            row.configure(bg=C["hover"]); dot.configure(bg=C["hover"])
            lbl.configure(bg=C["hover"], fg=C["white"])
        else:
            row.configure(bg=C["panel"]); dot.configure(bg=C["panel"])
            lbl.configure(bg=C["panel"], fg=C["gray"])


if __name__ == "__main__":
    app = App()
    app.mainloop()