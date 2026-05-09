
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import sympy as sp
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from interfacePartagee import C, F, Card, mk_entry, mk_btn, configure_treeview_style
from axe1 import comparer_methodes as comparer_nonlin
from iteratives import comparer_methodes as comparer_lineaire





class ComparaisonFrame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=C["bg"])
        self._build()

    def _build(self):
        top = tk.Frame(self, bg=C["bg"], padx=28, pady=18)
        top.pack(fill="x")
        tk.Label(top, text="Comparaison des algorithmes",
                 font=F["title"], bg=C["bg"], fg=C["white"]).pack(side="left")
        tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

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

        # ── Type ──
        type_card = Card(self._content, "Type de comparaison")
        type_card.pack(fill="x", **p)
        type_row = tk.Frame(type_card.body, bg=C["card"])
        type_row.pack(fill="x")
        type_row.columnconfigure((0, 1), weight=1)

        self._type_var = tk.StringVar(value="nonlin")
        self._type_frames = {}
        for name, label, desc, col in [
            ("nonlin",   "Fonctions non linéaires", "Newton vs Dichotomie vs Point fixe", 0),
            ("lineaire", "Systèmes linéaires",       "Jacobi vs Gauss-Seidel",            1),
        ]:
            f = tk.Frame(type_row, bg=C["card2"],
                         highlightthickness=1, highlightbackground=C["border"],
                         padx=14, pady=12, cursor="hand2")
            f.grid(row=0, column=col, padx=5, pady=5, sticky="ew")
            nl = tk.Label(f, text=label, font=F["h3"], bg=C["card2"], fg=C["white"])
            nl.pack(anchor="w")
            dl = tk.Label(f, text=desc, font=F["small"], bg=C["card2"], fg=C["gray"])
            dl.pack(anchor="w")
            for w in [f, nl, dl]:
                w.bind("<Button-1>", lambda e, n=name: self._sel_type(n))
            self._type_frames[name] = f

        # ── Params non linéaires ──
        self._nl_card = Card(self._content, "Paramètres — Fonctions non linéaires")
        nl_row = tk.Frame(self._nl_card.body, bg=C["card"])
        nl_row.pack(fill="x")
        for lbl, attr, default, w in [
            ("f(x) =", "e_nl_fx",  "x**3 - x - 2", 22),
            ("a",      "e_nl_a",   "1",              5),
            ("b",      "e_nl_b",   "2",              5),
            
            ("Tol",    "e_nl_tol", "1e-6",           8),
        ]:
            col = tk.Frame(nl_row, bg=C["card"])
            col.pack(side="left", padx=(0, 10))
            tk.Label(col, text=lbl, font=F["small"],
                     bg=C["card"], fg=C["gray"]).pack(anchor="w", pady=(0, 3))
            e = mk_entry(col, default, w)
            e.pack(ipady=5)
            setattr(self, attr, e)

        # ── Params linéaires ──
        self._lin_card = Card(self._content, "Paramètres — Systèmes linéaires")
        lin_body = self._lin_card.body

        sz_row = tk.Frame(lin_body, bg=C["card"])
        sz_row.pack(fill="x", pady=(0, 8))
        tk.Label(sz_row, text="n =", font=F["small"],
                 bg=C["card"], fg=C["gray"]).pack(side="left")
        self._lin_n_var = tk.IntVar(value=3)
        for n in [2, 3, 4]:
            tk.Radiobutton(sz_row, text=str(n), variable=self._lin_n_var, value=n,
                           font=F["body"], bg=C["card"], fg=C["white"],
                           activebackground=C["card"], activeforeground=C["acc_light"],
                           selectcolor=C["acc_bg"], indicatoron=0,
                           relief="flat", bd=0, highlightthickness=1,
                           highlightbackground=C["border"],
                           cursor="hand2", padx=10, pady=3,
                           command=self._rebuild_lin_matrix).pack(side="left", padx=4)

        self._lin_mat_frame = tk.Frame(lin_body, bg=C["card"])
        self._lin_mat_frame.pack(fill="x")
        self._lin_matrix_entries = []
        self._lin_b_entries = []
        self._rebuild_lin_matrix()

        tol_row = tk.Frame(lin_body, bg=C["card"])
        tol_row.pack(fill="x", pady=(8, 0))
        for lbl, attr, default, w in [
            ("Tol",      "e_lin_tol",  "1e-6", 10),
            ("Iter max", "e_lin_nmax", "100",    6),
        ]:
            col = tk.Frame(tol_row, bg=C["card"])
            col.pack(side="left", padx=(0, 12))
            tk.Label(col, text=lbl, font=F["small"],
                     bg=C["card"], fg=C["gray"]).pack(anchor="w", pady=(0, 3))
            e = mk_entry(col, default, w)
            e.pack(ipady=5)
            setattr(self, attr, e)
        # ── Norme ──
        norm_card = Card(self._content, "Norme")
         
        norm_card.pack(fill="x", **p)
        norm_row = tk.Frame(norm_card.body, bg=C["card"])
        norm_row.pack(fill="x")
        self._norm_var = tk.IntVar(value=2)
        self._norm_frames = {}
        for val, label, tip in [
          (1, "‖·‖₁", "Somme |xᵢ|"),
          (2, "‖·‖₂", "Euclidienne"),
          (0, "‖·‖∞", "Max |xᵢ|"),
          (3, "‖·‖p", ""),
]:
           f = tk.Frame(norm_row, bg=C["card2"],
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

           self._norm_card = norm_card
           self._norm_card.pack_forget()

        self._p_custom_col = tk.Frame(norm_card.body, bg=C["card"])
        tk.Label(self._p_custom_col, text="p =", font=F["small"],
         bg=C["card"], fg=C["gray"]).pack(side="left", padx=(0, 4))
        self._p_custom_entry = mk_entry(self._p_custom_col, "3", 4)
        self._p_custom_entry.pack(side="left", ipady=4)
        self._p_custom_col.pack_forget()
        self._sel_norm(2)

        # ── Bouton ──
        btn_row = tk.Frame(self._content, bg=C["bg"], padx=28, pady=4)
        btn_row.pack(fill="x")
        mk_btn(btn_row, "  Comparer  ", self._run, color=C["accent"]).pack(side="left")

        

       

    def _style_ax(self, ax):
        ax.tick_params(colors=C["gray"], labelsize=8)
        for s in ax.spines.values():
            s.set_edgecolor(C["border"])


    def _sel_norm(self, val):
      self._norm_var.set(val)
      for v, f in self._norm_frames.items():
        is_sel = (v == val)
        bg  = C["acc_bg"] if is_sel else C["card2"]
        brd = C["accent"] if is_sel else C["border"]
        f.configure(bg=bg, highlightbackground=brd)
        for w in f.winfo_children():
            w.configure(bg=bg)
      if hasattr(self, '_p_custom_col'):
        if val == 3:
            self._p_custom_col.pack(anchor="w", pady=(4, 0))
        else:
            self._p_custom_col.pack_forget()


    def _sel_type(self, name):
      self._type_var.set(name)
      for n, f in self._type_frames.items():
        is_sel = (n == name)
        bg  = C["acc_bg"] if is_sel else C["card2"]
        brd = C["accent"] if is_sel else C["border"]
        f.configure(bg=bg, highlightbackground=brd)
        for w in f.winfo_children():
            w.configure(bg=bg)
      if name == "nonlin":
        self._nl_card.pack(fill="x", padx=28, pady=8)
        self._lin_card.pack_forget()
        self._norm_card.pack_forget()
      else:
        self._lin_card.pack(fill="x", padx=28, pady=8)
        self._nl_card.pack_forget()
        self._norm_card.pack(fill="x", padx=28, pady=8)

    def _rebuild_lin_matrix(self):
        n = self._lin_n_var.get()
        for w in self._lin_mat_frame.winfo_children():
            w.destroy()
        self._lin_matrix_entries = []
        self._lin_b_entries = []

        hdr = tk.Frame(self._lin_mat_frame, bg=C["card"])
        hdr.pack(fill="x", pady=(0, 4))
        for j in range(n):
            tk.Label(hdr, text=f"  x{j+1}", font=F["small"],
                     bg=C["card"], fg=C["acc_light"],
                     width=7, anchor="center").grid(row=0, column=j, padx=2)
        tk.Label(hdr, text="  | b", font=F["small"],
                 bg=C["card"], fg=C["teal"],
                 width=8, anchor="center").grid(row=0, column=n, padx=2)

        default_A = [[10 if i == j else (1 if abs(i-j) == 1 else 0)
                      for j in range(n)] for i in range(n)]
        default_b = [sum(default_A[i]) for i in range(n)]

        for i in range(n):
            rf = tk.Frame(self._lin_mat_frame, bg=C["card"])
            rf.pack(fill="x", pady=2)
            row_ents = []
            for j in range(n):
                e = mk_entry(rf, str(default_A[i][j]), 7)
                e.grid(row=0, column=j, padx=2)
                row_ents.append(e)
            self._lin_matrix_entries.append(row_ents)
            tk.Label(rf, text="│", font=F["mono"],
                     bg=C["card"], fg=C["muted"]).grid(row=0, column=n, padx=4)
            be = mk_entry(rf, str(default_b[i]), 7)
            be.grid(row=0, column=n+1, padx=2)
            self._lin_b_entries.append(be)

    def _read_lin_matrix(self):
        n = self._lin_n_var.get()
        A = np.zeros((n, n))
        b = np.zeros(n)
        for i in range(n):
            for j in range(n):
                A[i, j] = float(self._lin_matrix_entries[i][j].get())
            b[i] = float(self._lin_b_entries[i].get())
        return A, b

    def _run(self):
        if self._type_var.get() == "nonlin":
            self._run_nonlin()
        else:
            self._run_lineaire()

    def _run_nonlin(self):
      f_str = self.e_nl_fx.get().strip().replace('^', '**')
      try:
        a   = float(self.e_nl_a.get())
        b   = float(self.e_nl_b.get())
        tol = float(self.e_nl_tol.get())
      except ValueError as e:
        messagebox.showerror("Erreur", str(e))
        return
      comparer_nonlin(f_str, a, b, tol)
    
    def _run_lineaire(self):
      try:
        A, b = self._read_lin_matrix()
        tol  = float(self.e_lin_tol.get())
        nmax = int(self.e_lin_nmax.get())
      except ValueError as e:
        messagebox.showerror("Erreur", str(e))
        return
      norm_type = self._norm_var.get()
      p_val = None
      if norm_type == 3:
        try:
            p_val = int(self._p_custom_entry.get())
        except ValueError:
            p_val = 3
      x0 = np.zeros(len(b))
      comparer_lineaire(A, b, x0, nmax, tol, norm_type, p_val)