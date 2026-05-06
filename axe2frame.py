import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import csv, json

from interfacePartagee import C, F, Card, StatCard, mk_entry, mk_btn, configure_treeview_style


# ══════════════════════════════════════════════
#  ALGORITHMES NUMÉRIQUES
# ══════════════════════════════════════════════

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
    n = len(b)
    Ab = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])
    steps = []
    for k in range(n):
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
    n = len(b)
    Ab = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])
    col_order = list(range(n))
    steps = []
    for k in range(n):
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
    x = np.zeros(n)
    for i, ci in enumerate(col_order):
        x[ci] = x_perm[i]
    return x, steps


def decomp_lu(A, b):
    n = len(b)
    L = np.eye(n)
    U = A.astype(float).copy()
    steps = []
    for k in range(n):
        if abs(U[k, k]) < 1e-14:
            raise ValueError("Pivot nul — essayez Gauss avec pivot.")
        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i] -= factor * U[k]
            steps.append(f"m_{i+1}{k+1} = {factor:.4f}")
    y = _forward_sub(L, b.astype(float))
    x = _back_sub(U, y)
    return x, L, U, steps


def decomp_cholesky(A, b):
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
            B = -R / D[:, None]
            rho = max(abs(np.linalg.eigvals(B)))
            return True, x, iters, errors, rho
    B = -R / D[:, None]
    rho = max(abs(np.linalg.eigvals(B)))
    return False, x, iters, errors, rho


def gauss_seidel(A, b, x0, tol, nmax):
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
#  FRAME PRINCIPALE
# ══════════════════════════════════════════════

class Axe2Frame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=C["bg"])
        self._result_data = []
        self._n = 3
        self._matrix_entries = []
        self._b_entries = []
        self._x0_entries = []
        
        self._build()

    # ──────────────────────────────────────────
    def _build(self):
        # ── Top bar ──
        top = tk.Frame(self, bg=C["bg"], padx=28, pady=18)
        top.pack(fill="x")
        tk.Label(top, text="Axe 2 — Résolution de systèmes linéaires",
                 font=F["title"], bg=C["bg"], fg=C["white"]).pack(side="left")
        tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

        # ── Scrollable canvas ──
        outer = tk.Canvas(self, bg=C["bg"], bd=0, highlightthickness=0)
        self._content = tk.Frame(outer, bg=C["bg"])
        sb = ttk.Scrollbar(self, orient="vertical", command=outer.yview)
        outer.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        outer.pack(fill="both", expand=True)
        win = outer.create_window((0, 0), window=self._content, anchor="nw")
        outer.bind("<Configure>", lambda e: outer.itemconfig(win, width=e.width))
        self._content.bind("<Configure>",
                           lambda e: outer.configure(scrollregion=outer.bbox("all")))

        p = dict(padx=28, pady=8)

        # ── Catégorie ──
        cat_card = Card(self._content, "Catégorie de méthode")
        cat_card.pack(fill="x", **p)
        cat_row = tk.Frame(cat_card.body, bg=C["card"])
        cat_row.pack(fill="x")
        cat_row.columnconfigure((0, 1), weight=1)

        self._cat_var = tk.StringVar(value="Directe") #choisir la methode
        self._cat_frames = {}
        for name, desc, c in [
            ("Directe",   "Gauss, LU, Cholesky",  0),
            ("Itérative", "Jacobi, Gauss-Seidel",  1),
        ]:
            f = tk.Frame(cat_row, bg=C["card2"],
                         highlightthickness=1, highlightbackground=C["border"],
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
            ("Gauss (pivot partiel)", "Stabilité numérique standard",  0),
            ("Gauss (pivot total)",   "Stabilité maximale",            1),
            ("Décomposition LU",      "Réutilisable pour plusieurs b", 2),
            ("Cholesky",              "Pour matrices SDP uniquement",   3),
        ]
        self._iter_algos = [
            ("Jacobi",       "Convergence si diag. dominante", 0),
            ("Gauss-Seidel", "Convergence plus rapide",        1),
        ]
        self._sel_cat("Directe")

        # ── Recommandation ──
        rec = tk.Frame(self._content, bg=C["teal_bg"],
                       highlightthickness=1, highlightbackground=C["teal"],
                       padx=16, pady=12)
        rec.pack(fill="x", **p)
        tk.Label(rec, text="Recommandation", font=F["h3"],
                 bg=C["teal_bg"], fg=C["teal"]).pack(anchor="w")
        self._rec = tk.Label(rec, text="Choisissez un algorithme et saisissez la matrice.",
                             font=F["body"], bg=C["teal_bg"], fg="#b0e8d5",
                             wraplength=900, justify="left")
        self._rec.pack(anchor="w", pady=(4, 0))

        # ── Taille ──
        sz_card = Card(self._content, "Taille du système")
        sz_card.pack(fill="x", **p)
        sz_row = tk.Frame(sz_card.body, bg=C["card"])
        sz_row.pack(anchor="w")
        tk.Label(sz_row, text="n =", font=F["small"],
                 bg=C["card"], fg=C["gray"]).pack(side="left")
        self._n_var = tk.IntVar(value=3)
        for n in [2, 3, 4, 5]:
            tk.Radiobutton(sz_row, text=str(n), variable=self._n_var, value=n,
                           font=F["body"], bg=C["card"], fg=C["white"],
                           activebackground=C["card"], activeforeground=C["acc_light"],
                           selectcolor=C["acc_bg"], indicatoron=0,
                           relief="flat", bd=0, highlightthickness=1,
                           highlightbackground=C["border"],
                           cursor="hand2", padx=12, pady=4,
                           command=self._rebuild_matrix).pack(side="left", padx=4)

        # ── Matrice ──
        self._mat_card = Card(self._content, "Matrice A et vecteur b")
        self._mat_card.pack(fill="x", **p)
        self._mat_body = self._mat_card.body
        self._rebuild_matrix()

        # ── Paramètres itératifs ──
        self._iter_param_card = Card(self._content, "Paramètres itératifs")
        iter_row = tk.Frame(self._iter_param_card.body, bg=C["card"])
        iter_row.pack(fill="x")
        for lbl, attr, default, width in [
            ("Tolérance", "e_tol",  "1e-6", 10),
            
        ]:
            col = tk.Frame(iter_row, bg=C["card"])
            col.pack(side="left", padx=(0, 12))
            tk.Label(col, text=lbl, font=F["small"],
                     bg=C["card"], fg=C["gray"]).pack(anchor="w", pady=(0, 3))
            e = mk_entry(col, default, width)
            e.pack(ipady=5)
            setattr(self, attr, e)

        # ── Choix de la norme ──
        norm_col = tk.Frame(iter_row, bg=C["card"])
        norm_col.pack(side="left", padx=(12, 0))
        tk.Label(norm_col, text="Norme erreur", font=F["small"],
                 bg=C["card"], fg=C["gray"]).pack(anchor="w", pady=(0, 3))
        norm_btn_row = tk.Frame(norm_col, bg=C["card"])
        norm_btn_row.pack(anchor="w")
        self._norm_var = tk.IntVar(value=2)
        self._norm_frames = {}
        for val, label, tip in [
            (1, "‖·‖₁", "Somme |xᵢ|"),
            (2, "‖·‖₂", "Euclidienne"),
            (0, "‖·‖∞", "Max |xᵢ|"),
        ]:
            f = tk.Frame(norm_btn_row, bg=C["card2"],
                         highlightthickness=1, highlightbackground=C["border"],
                         padx=9, pady=5, cursor="hand2")
            f.pack(side="left", padx=(0, 4))
            nl = tk.Label(f, text=label, font=F["mono"], bg=C["card2"], fg=C["white"])
            nl.pack()
            tl = tk.Label(f, text=tip, font=("Segoe UI", 7), bg=C["card2"], fg=C["gray"])
            tl.pack()
            for w in [f, nl, tl]:
                w.bind("<Button-1>", lambda e, v=val: self._sel_norm(v))
            self._norm_frames[val] = f
        self._sel_norm(2)
 
        self._x0_frame = tk.Frame(self._iter_param_card.body, bg=C["card"])
        self._x0_frame.pack(fill="x", pady=(8, 0))
        tk.Label(self._x0_frame, text="Point de départ x₀ :",
                 font=F["small"], bg=C["card"], fg=C["gray"]).pack(side="left", padx=(0, 8))
        self._x0_entries = []
        self._rebuild_x0()
        self._iter_param_card.pack_forget()
 

        # ── Boutons ──
        btn_row = tk.Frame(self._content, bg=C["bg"], padx=28, pady=4)
        btn_row.pack(fill="x")
        mk_btn(btn_row, "  Calculer  ", self._run2, color=C["accent"]).pack(side="left")
        mk_btn(btn_row, "  Réinitialiser  ", self._reset_matrix,
               secondary=True).pack(side="left", padx=10)

        # ── Stats ──
        sr = tk.Frame(self._content, bg=C["bg"])
        sr.pack(fill="x", **p)
        sr.columnconfigure((0, 1, 2, 3), weight=1)
        self.st_norm = StatCard(sr, "‖Ax − b‖")
        self.st_norm.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.st_iter = StatCard(sr, "Itérations")
        self.st_iter.grid(row=0, column=1, sticky="ew", padx=(0, 8))
        self.st_rho  = StatCard(sr, "Rayon spectral ρ")
        self.st_rho.grid(row=0, column=2, sticky="ew", padx=(0, 8))
        self.st_conv = StatCard(sr, "Convergence")
        self.st_conv.grid(row=0, column=3, sticky="ew")

        # ── Solution ──
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

        # ── Tableau ──
        tc = Card(self._content, "Détail des étapes / itérations")
        tc.pack(fill="x", **p)
        configure_treeview_style()
        self._tree_frame = tc.body
        self._tree = None

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

    def _sel_cat(self, name): #changer le style du card categorie selectionné et selectionner le 1 er algo apr defaut
        self._cat_var.set(name)
        for n, f in self._cat_frames.items():
            is_sel = (n == name)
            bg  = C["acc_bg"] if is_sel else C["card2"]
            brd = C["accent"] if is_sel else C["border"]
            f.configure(bg=bg, highlightbackground=brd)
            for w in f.winfo_children():
                w.configure(bg=bg)

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
                         highlightthickness=1, highlightbackground=C["border"],
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

        if not hasattr(self, '_iter_param_card'):
            return   
        if name == "Itérative":
            self._iter_param_card.pack(fill="x", padx=28, pady=8)
        else:
            self._iter_param_card.pack_forget()

    def _sel_algo(self, name): #changer le style du algo selectionné
        self._algo_var.set(name)
        for n, f in self._algo_frames.items():
            is_sel = (n == name)
            bg  = C["acc_bg"] if is_sel else C["card2"]
            brd = C["accent"] if is_sel else C["border"]
            f.configure(bg=bg, highlightbackground=brd)
            for w in f.winfo_children():
                w.configure(bg=bg)  
    def _sel_norm(self, name): 
        self._norm_var.set(name)
        
    def _rebuild_matrix(self):
        self._n = self._n_var.get()
        for w in self._mat_body.winfo_children():
            w.destroy() #supp l ancienne matrice

        header = tk.Frame(self._mat_body, bg=C["card"])
        header.pack(fill="x", pady=(0, 6))
        for j in range(self._n):
            tk.Label(header, text=f"  x{j+1}", font=F["small"],
                     bg=C["card"], fg=C["acc_light"],
                     width=8, anchor="center").grid(row=0, column=j, padx=2)
        tk.Label(header, text="  |  b", font=F["small"],
                 bg=C["card"], fg=C["teal"],
                 width=10, anchor="center").grid(row=0, column=self._n, padx=2)

        self._matrix_entries = []
        self._b_entries = []
        for i in range(self._n): #creer les lignes de la matrice
            row_f = tk.Frame(self._mat_body, bg=C["card"])
            row_f.pack(fill="x", pady=2)
            row_ents = []
            for j in range(self._n): #creer l entree pour chaque case
                e = mk_entry(row_f, "1" if i == j else "0", 8) #mat identite par defaut
                e.grid(row=0, column=j, padx=2)
                row_ents.append(e)
            self._matrix_entries.append(row_ents)
            tk.Label(row_f, text="│", font=F["mono"],
                     bg=C["card"], fg=C["muted"]).grid(row=0, column=self._n, padx=4)
            be = mk_entry(row_f, "0", 8)
            be.grid(row=0, column=self._n + 1, padx=2)
            self._b_entries.append(be)

        self._rebuild_x0()

    def _rebuild_x0(self):
        if not hasattr(self, '_x0_frame'):
            return   
        for w in self._x0_frame.winfo_children():
            if isinstance(w, tk.Entry): #prends seulement enteries du frame x0
                w.destroy()
        self._x0_entries = []
        for i in range(self._n):
            e = mk_entry(self._x0_frame, "0", 6)
            e.pack(side="left", padx=3)
            self._x0_entries.append(e)

    def _reset_matrix(self): #initialise la matrice a identite
        for i in range(self._n):
            for j in range(self._n):
                self._matrix_entries[i][j].delete(0, "end")
                self._matrix_entries[i][j].insert(0, "1" if i == j else "0")
            self._b_entries[i].delete(0, "end")
            self._b_entries[i].insert(0, "0")

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
                converged, x_sol, steps_or_iters, errors, rho_val = \
                    jacobi(A, b, x0, tol, nmax)
                if not converged:
                    messagebox.showwarning("Jacobi",
                        f"Pas de convergence en {nmax} itérations.\n"
                        f"Rayon spectral ρ ≈ {rho_val:.4f} (> 1 → diverge).")
            elif algo == "Gauss-Seidel":
                tol  = float(self.e_tol.get())
                nmax = int(self.e_nmax.get())
                x0   = np.array([float(e.get()) for e in self._x0_entries])
                converged, x_sol, steps_or_iters, errors, rho_val = \
                    gauss_seidel(A, b, x0, tol, nmax)
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
        else:
            residuals_comp = np.abs(A @ x_sol - b)
            self._ax.bar(range(len(residuals_comp)), residuals_comp,
                         color=C["accent"], alpha=0.8)
            self._ax.set_xlabel("Composante", color=C["gray"], fontsize=9)
            self._ax.set_ylabel("|Ax−b|ᵢ", color=C["gray"], fontsize=9)
            self._ax.set_xticks(range(len(residuals_comp)))
            self._ax.set_xticklabels([f"r{i+1}" for i in range(len(residuals_comp))],
                                      color=C["gray"], fontsize=8)
        self._fig.tight_layout(pad=0.8)
        self._canvas.draw()

        self._build_tree(cat, steps_or_iters, errors)

        # ── Export data ──
        self._result_data = {
            "algorithme": algo,
            "solution":   x_sol.tolist(),
            "residuel":   float(residual),
            "converged":  converged,
            "details":    [],
        }
        if cat == "Itérative":
            for k, (xk, err) in enumerate(zip(steps_or_iters, errors)):
                self._result_data["details"].append(
                    {"iter": k, "x": xk.tolist(), "err": float(err)})
        else:
            for k, s in enumerate(steps_or_iters):
                self._result_data["details"].append({"etape": k, "op": s})

    # ──────────────────────────────────────────
    def _build_tree(self, cat, data, errors=None):
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
            for k, xk in enumerate(data):
                err = errors[k] if errors else (
                    np.linalg.norm(xk - data[k - 1], np.inf) if k > 0 else float('inf'))
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
                tk.Label(parent, text="  ".join(f"{v:10.4f}" for v in row),
                         font=F["mono"], bg=C["bg"], fg=C["white"]).pack(anchor="w")

        show_mat(frame, L,
                 "Matrice L" if U is None else "L (triangulaire inférieure)", C["teal"])
        if U is not None:
            show_mat(frame, U, "U (triangulaire supérieure)", C["acc_light"])
        mk_btn(top, "Fermer", top.destroy, secondary=True).pack(pady=10)

    # ──────────────────────────────────────────
    def _export_csv2(self):
        if not self._result_data:
            messagebox.showinfo("Info", "Aucun résultat.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv")],
            initialfile="resultats_axe2.csv")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Algorithme", self._result_data["algorithme"]])
            w.writerow(["Résidu ‖Ax−b‖", f"{self._result_data['residuel']:.6e}"])
            w.writerow([])
            w.writerow(["Solution"])
            for i, xi in enumerate(self._result_data["solution"]):
                w.writerow([f"x{i+1}", f"{xi:.12f}"])
            w.writerow([])
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
            defaultextension=".json", filetypes=[("JSON", "*.json")],
            initialfile="resultats_axe2.json")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._result_data, f, indent=2, ensure_ascii=False)
        messagebox.showinfo("Exporté", f"JSON sauvegardé :\n{path}")