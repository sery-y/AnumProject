import tkinter as tk
from tkinter import ttk

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
            tk.Label(hdr, text=title.upper(), font=("Segoe UI", 8, "bold"),
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

    def set(self, v):
        self._val.configure(text=v)


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


def configure_treeview_style():
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