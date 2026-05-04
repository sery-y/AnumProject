import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sympy as sp
import csv, os, json
from axe1 import dichotomie, point_fixe, newton_

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
            ("f(x) =", "e_fx", "x**3 - 2*x - 5", 20),
            ("x0",     "e_x0", "2.0", 7),
            ("a", "e_a", "1.0", 5),
            ("b", "e_b", "3.0", 5),
            ("Tolerance", "e_tol", "1e-7", 7),
        ]
        for lbl, attr, default, w in fields:
            col = tk.Frame(row, bg=C["card"])
            col.pack(side="left", padx=(0,10))
            tk.Label(col, text=lbl, font=F["small"],
                     bg=C["card"], fg=C["gray"]).pack(anchor="w", pady=(0,3))
            e = mk_entry(col, default, w)
            e.pack(ipady=5)
            setattr(self, attr, e) #stocke les entrees

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
        self._update_rec()

    def _update_rec(self):
        recs = {
            "Newton": "Newton est conseille si f est derivable et si une bonne valeur initiale est disponible.",
            "Dichotomie":     "Dichotomie est garantie de converger si f(a)*f(b) < 0. Lente mais fiable.",
            "Point fixe":     "Point fixe converge si |g'(x)| < 1 au voisinage du point fixe.",
            
        }
        self._rec.configure(text=recs.get(self.algo_var.get(), ""))

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
                       "Point fixe non applicable.")
                  messagebox.showwarning("Point fixe", msg)
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
        top = tk.Frame(self, bg=C["bg"], padx=28, pady=18)
        top.pack(fill="x")
        tk.Label(top, text="Axe 2 — Resolution des systemes lineaires",
                 font=F["title"], bg=C["bg"], fg=C["white"]).pack(side="left")
        tk.Label(top, text=" Etudiant 2 ", font=F["small"],
                 bg="#0d2820", fg=C["teal"], padx=8, pady=4).pack(side="right", anchor="n")
        tk.Frame(self, bg=C["border"], height=1).pack(fill="x")
        tk.Label(self, text="\n\n  Module en cours de developpement — Etudiant 2",
                 font=F["h2"], bg=C["bg"], fg=C["muted"]).pack(pady=60)


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
        tk.Label(logo, text="USTHB — 3INGSoft", font=F["small"],
                 bg=C["panel"], fg=C["muted"]).pack(anchor="w")

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
        for txt in ["d  Derivees & continuite", "=  Comparaison algos", "^  Rapport & export"]:
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