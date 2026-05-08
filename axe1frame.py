"""
axe1_frame.py — Axe 1 : Résolution de fonctions non linéaires.
Dépendances : shared.py, axe1.py (dichotomie, point_fixe, newton_, …)
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sympy as sp
import csv, json

from interfacePartagee import C, F, Card, StatCard, mk_entry, mk_btn, configure_treeview_style
from axe1 import (dichotomie, point_fixe, newton_,
                  point_fixe_avec_relaxation, recommander_methode, point_fixe_avec_phi, trace_courbe,afficher_tableau_plt)


class Axe1Frame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=C["bg"])
        self._iters = []
        self._build()

    # ──────────────────────────────────────────
    def _build(self):
        self._initialized    = False
        self._dicho_success  = False
        self._dicho_sol      = None
        self._pf_success     = False
        self._pf_sol         = None
        self._pf_phi         = None
        self._newton_success = False
        self._newton_sol     = None

        # ── Top bar ──
        top = tk.Frame(self, bg=C["bg"], padx=28, pady=18)
        top.pack(fill="x")
        tk.Label(top, text="Axe 1 — Résolution de fonctions non linéaires",
                 font=F["title"], bg=C["bg"], fg=C["white"]).pack(side="left")
        tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

        # ── Scrollable canvas ──
        outer = tk.Canvas(self, bg=C["bg"], bd=0, highlightthickness=0)
        sb = ttk.Scrollbar(self, orient="vertical", command=outer.yview)
        outer.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        outer.pack(fill="both", expand=True)
        content = tk.Frame(outer, bg=C["bg"])
        win = outer.create_window((0, 0), window=content, anchor="nw")
        outer.bind("<Configure>", lambda e: outer.itemconfig(win, width=e.width))
        content.bind("<Configure>",
                     lambda e: outer.configure(scrollregion=outer.bbox("all")))

        p = dict(padx=28, pady=8)

        # ── Choisir un algorithme ──
        ac = Card(content, "Choisir un algorithme")
        ac.pack(fill="x", **p)
        grid = tk.Frame(ac.body, bg=C["card"])
        grid.pack(fill="x")
        grid.columnconfigure((0, 1), weight=1)

        self.algo_var = tk.StringVar(value="Newton")
        self._algo_frames = {}
        algos = [
            ("Newton",      "Convergence quadratique", 0, 0),
            ("Dichotomie",  "Convergence linéaire",    0, 1),
            ("Point fixe",  "Simple, conditionnel",    1, 0),
        ]
        for name, desc, r, c in algos:
            f = tk.Frame(grid, bg=C["card2"],
                         highlightthickness=1, highlightbackground=C["border"],
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
        self._rec.pack(anchor="w", pady=(4, 0))

        self._sel_algo("Newton")

        # ── Paramètres ──
        pc = Card(content, "Paramètres")
        pc.pack(fill="x", **p)
        row = tk.Frame(pc.body, bg=C["card"])
        row.pack(fill="x")

        fields = [
            ("f(x) =",    "e_fx",  "", 20),
            ("x0",        "e_x0",  "", 7),
            ("a",         "e_a",   "", 5),
            ("b",         "e_b",   "", 5),
            ("Tolérance", "e_tol", "", 7),
        ]
        for lbl, attr, default, w in fields:
            col = tk.Frame(row, bg=C["card"])
            col.pack(side="left", padx=(0, 10))
            tk.Label(col, text=lbl, font=F["small"],
                     bg=C["card"], fg=C["gray"]).pack(anchor="w", pady=(0, 3))
            e = mk_entry(col, default, w)
            e.pack(ipady=5)
            setattr(self, attr, e)
        
        # Champ phi (caché par défaut)
        self._phi_col = tk.Frame(row, bg=C["card"])
        tk.Label(self._phi_col, text="φ(x) =", font=F["small"],
         bg=C["card"], fg=C["gray"]).pack(anchor="w", pady=(0, 3))
        self.e_phi = mk_entry(self._phi_col, "", 15)
        self.e_phi.pack(ipady=5)
        self._phi_col.pack_forget()

        for attr in ["e_fx", "e_x0", "e_a", "e_b", "e_tol"]:
            getattr(self, attr).bind("<KeyRelease>", self._on_input_change)

        col_b = tk.Frame(row, bg=C["card"])
        col_b.pack(side="left")
        tk.Label(col_b, text=" ", font=F["small"], bg=C["card"], fg=C["gray"]).pack(pady=(0, 3))
        mk_btn(col_b, "Calculer", self._run).pack(ipady=4)

        # ── Stats ──
        sr = tk.Frame(content, bg=C["bg"])
        sr.pack(fill="x", **p)
        sr.columnconfigure((0, 1, 2), weight=1)
        self.st_r = StatCard(sr, "Racine approchée")
        self.st_r.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.st_i = StatCard(sr, "Itérations")
        self.st_i.grid(row=0, column=1, sticky="ew", padx=(0, 8))
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
        tc = Card(content, "Tableau des itérations")
        tc.pack(fill="x", **p)
        configure_treeview_style()

        cols = ("n", "xn", "fxn", "err")
        self.tree = ttk.Treeview(tc.body, columns=cols, show="headings",
                                  height=5, style="D.Treeview")
        for col, w, title in [("n", 55, "n"), ("xn", 210, "x_n"),
                               ("fxn", 210, "f(x_n)"), ("err", 160, "Erreur")]:
            self.tree.heading(col, text=title)
            self.tree.column(col, width=w, anchor="center")
        self.tree.pack(fill="x")

        # ── Boutons supplémentaires ──
        extra_card = Card(content, "Outils en plus")
        extra_card.pack(fill="x", **p)
        btn_row = tk.Frame(extra_card.body, bg=C["card"])
        btn_row.pack(fill="x")
        
        mk_btn(btn_row, "Tracer courbe", self._tracer_courbe, secondary=True).pack(side="left", padx=(0, 8))
        mk_btn(btn_row, "Tracer tableau", self._tracer_tableau, secondary=True).pack(side="left", padx=(0, 8))
        


        btns = tk.Frame(tc.body, bg=C["card"], pady=8)
        btns.pack(fill="x")
        mk_btn(btns, " Exporter CSV",  self._export_csv,  secondary=True).pack(side="left", padx=(0, 8))
        mk_btn(btns, " Exporter JSON", self._export_json, secondary=True).pack(side="left")

        tk.Frame(content, bg=C["bg"], height=20).pack()
        self._update_rec()

    # ──────────────────────────────────────────
    def _tracer_courbe(self):
      if not self._iters:
        messagebox.showinfo("Info", "Aucun résultat à tracer.")
        return
      f_str   = self.e_fx.get().strip().replace('^', '**')
      a       = float(self.e_a.get())
      b       = float(self.e_b.get())
      iters   = [it[1] for it in self._iters]
      sol     = self.st_r.get() if hasattr(self.st_r, 'get') else iters[-1]
      titre   = f"Courbe — {self.algo_var.get()}"
      plt.close('all')
      trace_courbe(f_str, iters, float(sol), a, b, titre)

    def _tracer_tableau(self):
      if not self._iters:
        messagebox.showinfo("Info", "Aucun résultat à afficher.")
        return
      methode = self.algo_var.get()
      iters   = [it[1] for it in self._iters]
      erreurs = [it[3] for it in self._iters]
      plt.close('all')
      afficher_tableau_plt(methode, iters, erreurs)

    def _style_ax(self, ax):
        ax.tick_params(colors=C["gray"], labelsize=8)
        for sp_ in ax.spines.values():
            sp_.set_edgecolor(C["border"])
        ax.set_xlabel("x", color=C["gray"], fontsize=9)
        ax.set_ylabel("f(x)", color=C["gray"], fontsize=9)

    def _sel_algo(self, name):
        self.algo_var.set(name)
        for n, f in self._algo_frames.items():
            is_sel = (n == name)
            bg  = C["acc_bg"] if is_sel else C["card2"]
            brd = C["accent"] if is_sel else C["border"]
            f.configure(bg=bg, highlightbackground=brd)
            for w in f.winfo_children():
                w.configure(bg=bg)
        if hasattr(self, '_phi_col'):
          if name == "Point fixe":
            self._phi_col.pack(side="left", padx=(0, 10))
          else:
            self._phi_col.pack_forget()

    def _update_rec(self):
        if not self._initialized:
            self._rec.configure(
                text="Veuillez entrer des paramètres valides pour obtenir une recommandation.")
            return
        if not any([self._dicho_success, self._pf_success, self._newton_success]):
            self._rec.configure(text="Aucune méthode applicable sur cet intervalle.")
            return
        _, message = recommander_methode(
            self._dicho_success,  self._dicho_sol,
            self._pf_success,     self._pf_sol,    self._pf_phi,
            self._newton_success, self._newton_sol)
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

            ok_d, sol_d, _, _ = dichotomie(f_str, a, b, tol, nmax)
            self._dicho_success = ok_d
            self._dicho_sol     = sol_d

            phi_str = self.e_phi.get().strip().replace('^', '**')
            if phi_str:
              ok_pf, phi, sol_pf, _, _, _ = point_fixe_avec_phi(
              f_str, phi_str, a, b, x0, tol, nmax)
            else:
              ok_pf, phi, sol_pf, _, _, _ = point_fixe(f_str, a, b, x0, tol, nmax)

            self._pf_success = ok_pf
            self._pf_sol     = sol_pf
            self._pf_phi     = phi

            

            ok_n, sol_n, _, _, *_ = newton_(f_str, a, b, x0, tol, nmax)
            self._newton_success = ok_n
            self._newton_sol     = sol_n
        except Exception:
            self._dicho_success  = False
            self._dicho_sol      = None
            self._pf_success     = False
            self._pf_sol         = None
            self._pf_phi         = None
            self._newton_success = False
            self._newton_sol     = None
        self._update_rec()

    # ──────────────────────────────────────────
    def _run(self):
        algo  = self.algo_var.get()
        f_str = self.e_fx.get().strip().replace('^', '**')

        try:
            tol  = float(self.e_tol.get())
            a    = float(self.e_a.get())
            b    = float(self.e_b.get())
            x0   = float(self.e_x0.get())
            nmax = 200
        except ValueError as e:
            messagebox.showerror("Erreur", f"Paramètre invalide : {e}")
            return

        root_val    = None
        iters_table = []

        try:
            x_sym = sp.Symbol('x')
            f_sym = sp.sympify(f_str)
            f_num = sp.lambdify(x_sym, f_sym, 'numpy')

            if algo == "Dichotomie":
                ok, sol, iters, erreurs = dichotomie(f_str, a, b, tol, nmax)
                if not ok:
                    messagebox.showwarning("Dichotomie",
                        "f(a) et f(b) ont le même signe — pas de racine garantie.")
                    return
                root_val = sol
                for i, (xn, err) in enumerate(zip(iters, erreurs)):
                    iters_table.append((i, xn, float(f_num(xn)), float(err)))

            elif algo == "Newton":
                ok, sol, iters, erreurs, _, _, _ = newton_(f_str, a, b, x0, tol, nmax)
                if not ok:
                    messagebox.showwarning("Newton",
                        "Conditions non vérifiées — Newton non applicable.")
                    return
                root_val = sol
                for i, (xn, err) in enumerate(zip(iters, erreurs)):
                    iters_table.append((i, xn, float(f_num(xn)), float(err)))

            elif algo == "Point fixe":
                phi_str = self.e_phi.get().strip().replace('^', '**')

                if phi_str:
                  ok, phi, sol, iters, erreurs, rapport = point_fixe_avec_phi(
                  f_str, phi_str, a, b, x0, tol, nmax)
                  if not ok:
                    messagebox.showwarning("Point fixe", "Méthode du point fixe non applicable avec cette phi.")
                    return
                else:
                  ok, phi, sol, iters, erreurs, rapport = point_fixe(
                    f_str, a, b, x0, tol, nmax)
                  if not ok:
                    msg = ("Aucune φ(x) stable et contractante trouvée sur [a,b].\n"
                           "Point fixe non applicable.\n"
                           "Voulez-vous essayer avec relaxation / Newton ?")
                    if messagebox.askyesno("Point fixe", msg):
                        ok, phi, sol, iters, erreurs, rapport = \
                            point_fixe_avec_relaxation(f_str, a, b, x0, tol, nmax)
                        if not ok:
                            messagebox.showerror(
                                "Échec",
                                "Même avec relaxation/Newton, aucune convergence trouvée.")
                    return
                # pour afficher phi quand je la genere
                if not phi_str and ok and phi is not None:
                  self.e_phi.delete(0, "end")
                  self.e_phi.insert(0, str(phi))

                root_val = sol
                for i, (xn, err) in enumerate(zip(iters, erreurs)):
                    iters_table.append((i, xn, float(f_num(xn)), float(err)))

        except Exception as e:
            messagebox.showerror("Erreur", str(e))
            return

        # ── Affichage ──
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
                it[0], f"{it[1]:.10f}", f"{it[2]:.6e}", f"{it[3]:.2e}"))
        self.tree.configure(height=len(iters_table))

        # ── Graphe ──
        self.ax.cla()
        self._style_ax(self.ax)
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
                self.ax.scatter([root_val], [0], color=C["teal"], s=60, zorder=6)
                self.ax.text(root_val + 0.05, max(ys) * 0.12,
                             f"x = {root_val:.2f}", color=C["teal"], fontsize=8)
            if len(iters_table) > 1:
                xi_pts = [it[1] for it in iters_table]
                yi_pts = np.clip([it[2] for it in iters_table], -50, 50)
                self.ax.scatter(xi_pts, yi_pts,
                                color=C["accent"], s=18, zorder=5, alpha=0.7)
            self.ax.legend(facecolor=C["card"], edgecolor=C["border"],
                           labelcolor=C["white"], fontsize=8)
        except Exception:
            pass
        self.fig.tight_layout(pad=0.8)
        self.canvas.draw()

    # ──────────────────────────────────────────
    def _export_csv(self):
        if not self._iters:
            messagebox.showinfo("Info", "Aucun résultat.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv")],
            initialfile="resultats_axe1.csv")
        if not path:
            return
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(["Algorithme", self.algo_var.get()])
            w.writerow(["f(x)", self.e_fx.get()])
            w.writerow([])
            w.writerow(["Iteration", "x_n", "f(x_n)", "Erreur"])
            for it in self._iters:
                w.writerow([it[0], f"{it[1]:.12f}", f"{it[2]:.12e}", f"{it[3]:.12e}"])
        messagebox.showinfo("Exporté", f"Sauvegardé :\n{path}")

    def _export_json(self):
        if not self._iters:
            messagebox.showinfo("Info", "Aucun résultat.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON", "*.json")],
            initialfile="resultats_axe1.json")
        if not path:
            return
        data = {
            "algorithme": self.algo_var.get(),
            "f(x)": self.e_fx.get(),
            "iterations": [{"n": it[0], "x": it[1], "fx": it[2], "err": it[3]}
                           for it in self._iters],
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        messagebox.showinfo("Exporté", f"JSON sauvegardé :\n{path}")