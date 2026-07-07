"""
Immune Disease Risk Detection System — Desktop GUI
===================================================
Tabs:
  1. 🩺  Patient Assessment   – autoimmune labs & symptoms → risk
  2. 🧬  Immunodeficiency     – immunodef-specific form → risk
  3. 📊  Dashboard            – 70/30 test metrics, charts, 4-level risk
  4. ⚙️  Train Models         – train / retrain with progress
  5. 📄  PDF Ingestion        – scanned or digital → extract → retrain
  6. 🗃️  Dataset Viewer       – browse / filter patients
  7. ℹ️  Model Info           – raw JSON metrics
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading, json, math, sys
from pathlib import Path

BASE_DIR   = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / 'outputs'
MODEL_DIR  = BASE_DIR / 'models'
DATA_DIR   = BASE_DIR / 'data'
sys.path.insert(0, str(BASE_DIR))

# ── Palette  (Light Mode) ────────────────────────────────────
C = {
    'bg':      '#f0f4f8',
    'surface': '#ffffff',
    'card':    '#ffffff',
    'card2':   '#f1f5f9',
    'accent':  '#4f46e5',
    'accent2': '#6366f1',
    'green':   '#16a34a',
    'yellow':  '#d97706',
    'red':     '#dc2626',
    'blue':    '#2563eb',
    'purple':  '#7c3aed',
    'teal':    '#0d9488',
    'orange':  '#ea580c',
    'text':    '#0f172a',
    'sub':     '#64748b',
    'border':  '#cbd5e1',
    'hover':   '#e0e7ff',
    'input':   '#f8fafc',
}

RISK_COLORS = {
    'No Risk':       '#3fb950',
    'Low Risk':      '#58a6ff',
    'Moderate Risk': '#e3b341',
    'Severe Risk':   '#f85149',
    'Normal':        '#3fb950',
    'At Risk':       '#f85149',
    'Autoimmune':    '#bc8cff',
    'Immunodeficiency': '#58a6ff',
}

FH  = ('Segoe UI', 19, 'bold')
FT  = ('Segoe UI', 12, 'bold')
FL  = ('Segoe UI', 10)
FS  = ('Segoe UI', 9)
FM  = ('Consolas', 9)
FBN = ('Segoe UI', 10, 'bold')

def rc(label): return RISK_COLORS.get(str(label), C['text'])


# ══════════════════════════════════════════════════════════════
# ROUNDED-CORNER CANVAS WIDGETS
# ══════════════════════════════════════════════════════════════
def rounded_rect(canvas, x1, y1, x2, y2, r=10, **kwargs):
    """Draw a filled rounded rectangle on a Canvas."""
    pts = [
        x1+r, y1,  x2-r, y1,
        x2,   y1,  x2,   y1+r,
        x2,   y2-r,x2,   y2,
        x2-r, y2,  x1+r, y2,
        x1,   y2,  x1,   y2-r,
        x1,   y1+r,x1,   y1,
        x1+r, y1,
    ]
    return canvas.create_polygon(pts, smooth=True, **kwargs)


class RoundedFrame(tk.Canvas):
    """
    A Canvas that draws itself as a rounded rectangle,
    then acts as a container for child widgets.
    """
    def __init__(self, parent, radius=12, bg=None, border_color=None,
                 border_width=1, **kwargs):
        self._bg     = bg or C['card']
        self._border = border_color or C['border']
        self._bw     = border_width
        self._r      = radius
        super().__init__(parent, bg=C['bg'],
                         highlightthickness=0, **kwargs)
        self.bind('<Configure>', self._redraw)
        self._inner = tk.Frame(self, bg=self._bg)
        self._inner.place(relx=0, rely=0, relwidth=1, relheight=1)

    def _redraw(self, event=None):
        self.delete('all')
        w, h = self.winfo_width(), self.winfo_height()
        if w < 4 or h < 4: return
        # Shadow
        rounded_rect(self, 3, 3, w-1, h-1, self._r,
                     fill='#94a3b8', outline='', stipple='gray25')
        # Border
        if self._bw:
            rounded_rect(self, 1, 1, w-2, h-2, self._r,
                         fill=self._border, outline='')
        # Fill
        rounded_rect(self, self._bw+1, self._bw+1,
                     w-self._bw-2, h-self._bw-2,
                     max(self._r-1, 2),
                     fill=self._bg, outline='')
        self._inner.lift()

    def interior(self): return self._inner


def rcard(parent, height=None, **kwargs):
    """Shortcut: create a RoundedFrame card."""
    kw = {'radius': 12, 'bg': C['card'], **kwargs}
    if height:
        kw['height'] = height
    f = RoundedFrame(parent, **kw)
    return f


# ── Plain helpers ─────────────────────────────────────────────
def lbl(parent, text, font=FL, fg=None, bg=None, **kw):
    return tk.Label(parent, text=text, font=font,
                    fg=fg or C['text'], bg=bg or C['card'], **kw)

def sep(parent, color=None, bg=None):
    return tk.Frame(parent, height=1, bg=color or C['border'])

def ent(parent, var=None, width=14, **kw):
    e = tk.Entry(parent, textvariable=var, width=width,
                 bg=C['input'], fg=C['text'],
                 insertbackground=C['text'],
                 relief='flat', font=FL,
                 highlightthickness=1,
                 highlightbackground=C['border'],
                 highlightcolor=C['accent'], **kw)
    return e

def rbtn(parent, text, cmd, color=None, **kw):
    """Rounded-look button via a Canvas."""
    bg = color or C['accent']
    c  = tk.Canvas(parent, width=kw.pop('width', 180),
                   height=kw.pop('height', 34),
                   bg=C['bg'], highlightthickness=0, cursor='hand2')
    def draw(w=180, h=34, col=bg):
        c.delete('all')
        rounded_rect(c, 0, 0, w, h, 8, fill=col, outline='')
        c.create_text(w//2, h//2, text=text,
                      fill='white', font=FBN, anchor='center')
    c.bind('<Configure>', lambda e: draw(e.width, e.height))
    c.bind('<Button-1>', lambda e: cmd())
    c.bind('<Enter>',  lambda e: draw(c.winfo_width(), c.winfo_height(),
                                       C['accent2'] if color in (None, C['accent']) else color))
    c.bind('<Leave>',  lambda e: draw(c.winfo_width(), c.winfo_height(), bg))
    draw()
    return c


# ══════════════════════════════════════════════════════════════
# CANVAS CHARTS (all using rounded_rect for bars)
# ══════════════════════════════════════════════════════════════
class BarChart(tk.Canvas):
    def __init__(self, parent, data, title='', color=None,
                 width=380, height=240, **kw):
        super().__init__(parent, width=width, height=height,
                         bg=C['card'], highlightthickness=0, **kw)
        self._draw(data, title, color or C['accent'], width, height)

    def _draw(self, data, title, color, W, H):
        if not data: return
        pad_l, pad_r, pad_t, pad_b = 140, 20, 28, 10
        items  = list(data.items())[:14]
        mx     = max(v for _, v in items) or 1
        bh     = max(8, (H - pad_t - pad_b) // max(len(items),1) - 4)
        step   = (H - pad_t - pad_b) / max(len(items),1)

        if title:
            self.create_text(W//2, 14, text=title,
                             fill=C['sub'], font=('Segoe UI',9,'bold'),
                             anchor='center')
        for i, (name, val) in enumerate(items):
            y = pad_t + i*step + step/2
            bar_w = max(int((val/mx)*(W-pad_l-pad_r)), 2)
            self.create_text(pad_l-6, y, text=str(name)[:20],
                             fill=C['text'], font=('Segoe UI',8), anchor='e')
            rounded_rect(self, pad_l, y-bh/2, pad_l+bar_w, y+bh/2, 4,
                         fill=color, outline='')
            self.create_text(pad_l+bar_w+5, y, text=f'{val:,}',
                             fill=C['sub'], font=('Segoe UI',8), anchor='w')


class ConfusionMatrix(tk.Canvas):
    def __init__(self, parent, cm, labels, width=280, height=220, **kw):
        super().__init__(parent, width=width, height=height,
                         bg=C['card'], highlightthickness=0, **kw)
        self._draw(cm, labels, width, height)

    def _draw(self, cm, labels, W, H):
        n = len(cm)
        if n == 0: return
        pad = 55
        cw  = (W-pad)//n
        ch  = (H-pad)//n
        mx  = max(max(row) for row in cm) or 1

        self.create_text(W//2, 12, text='Confusion Matrix',
                         fill=C['sub'], font=('Segoe UI',9,'bold'))
        for i, lab in enumerate(labels):
            x = pad + i*cw + cw//2
            self.create_text(x, 28, text=lab[:10],
                             fill=C['sub'], font=('Segoe UI',8), angle=30)
            self.create_text(24, pad + i*ch + ch//2, text=lab[:10],
                             fill=C['sub'], font=('Segoe UI',8))
        for i in range(n):
            for j in range(n):
                val   = cm[i][j]
                color = C['green'] if i == j else C['red']
                x0, y0 = pad + j*cw + 2, pad + i*ch + 2
                x1, y1 = x0+cw-4, y0+ch-4
                rounded_rect(self, x0, y0, x1, y1, 4,
                             fill=color, outline='', stipple='')
                self.create_text((x0+x1)//2, (y0+y1)//2, text=str(val),
                                 fill='white', font=('Segoe UI',10,'bold'))


class ROCCurve(tk.Canvas):
    def __init__(self, parent, fpr, tpr, auc,
                 width=280, height=230, **kw):
        super().__init__(parent, width=width, height=height,
                         bg=C['card'], highlightthickness=0, **kw)
        self._draw(fpr, tpr, auc, width, height)

    def _draw(self, fpr, tpr, auc, W, H):
        pl, pr, pt, pb = 44, 12, 22, 32
        iw, ih = W-pl-pr, H-pt-pb
        self.create_text(W//2, 11,
                         text=f'ROC Curve  (AUC = {auc:.4f})',
                         fill=C['sub'], font=('Segoe UI',9,'bold'))
        self.create_line(pl, pt, pl, pt+ih, fill=C['border'], width=1)
        self.create_line(pl, pt+ih, pl+iw, pt+ih, fill=C['border'], width=1)
        self.create_text(pl+iw//2, H-8, text='False Positive Rate',
                         fill=C['sub'], font=('Segoe UI',8))
        self.create_text(12, pt+ih//2, text='TPR',
                         fill=C['sub'], font=('Segoe UI',8), angle=90)
        self.create_line(pl, pt+ih, pl+iw, pt, fill=C['border'],
                         dash=(4,4), width=1)
        if fpr and tpr:
            pts = []
            for f, t in zip(fpr, tpr):
                pts.extend([pl+int(f*iw), pt+int((1-t)*ih)])
            if len(pts) >= 4:
                self.create_line(pts, fill=C['accent'], width=2.5, smooth=True)


class PieChart(tk.Canvas):
    def __init__(self, parent, data, title='', width=240, height=220, **kw):
        super().__init__(parent, width=width, height=height,
                         bg=C['card'], highlightthickness=0, **kw)
        self._draw(data, title, width, height)

    def _draw(self, data, title, W, H):
        if not data: return
        COLORS = [C['accent'],C['green'],C['yellow'],C['red'],
                  C['purple'],C['blue'],C['teal'],C['orange']]
        total  = sum(data.values()) or 1
        cx, cy = W//2, (H-30)//2 + 12
        r      = min(cx, cy) - 28
        start  = -90.0
        if title:
            self.create_text(W//2, 10, text=title,
                             fill=C['sub'], font=('Segoe UI',9,'bold'))
        for i, (lab, val) in enumerate(data.items()):
            ext   = (val/total)*360
            col   = COLORS[i % len(COLORS)]
            self.create_arc(cx-r, cy-r, cx+r, cy+r,
                            start=start, extent=ext,
                            fill=col, outline=C['card'], width=2)
            mid = math.radians(start + ext/2)
            lx  = cx + (r+16)*math.cos(mid)
            ly  = cy + (r+16)*math.sin(mid)
            pct = int(val/total*100)
            if pct > 4:
                self.create_text(lx, ly, text=f'{pct}%',
                                 fill='white', font=('Segoe UI',8,'bold'))
            start += ext
        # Legend
        ls = H - 18*min(len(data),3) - 4
        for i, (lab, val) in enumerate(list(data.items())[:6]):
            col = COLORS[i%len(COLORS)]
            row, col2 = divmod(i, 2)
            lx  = col2*(W//2) + 6
            ly  = ls + row*16
            rounded_rect(self, lx, ly, lx+10, ly+10, 3, fill=col, outline='')
            self.create_text(lx+14, ly+5, anchor='w',
                             text=f'{lab[:16]} ({val})',
                             fill=C['sub'], font=('Segoe UI',7))


# ══════════════════════════════════════════════════════════════
# SCROLLABLE FRAME helper
# ══════════════════════════════════════════════════════════════
def make_scrollable(parent):
    """Return (canvas, inner_frame). Pack canvas & scrollbar yourself."""
    canvas = tk.Canvas(parent, bg=C['bg'], highlightthickness=0)
    sb     = ttk.Scrollbar(parent, orient='vertical', command=canvas.yview)
    inner  = tk.Frame(canvas, bg=C['bg'])
    inner.bind('<Configure>',
               lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
    canvas.create_window((0,0), window=inner, anchor='nw')
    canvas.configure(yscrollcommand=sb.set)
    canvas.pack(side='left', fill='both', expand=True)
    sb.pack(side='right', fill='y')
    canvas.bind_all('<MouseWheel>',
        lambda e: canvas.yview_scroll(int(-1*(e.delta/120)),'units'))
    return canvas, inner


# ══════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════
class ImmuneRiskApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Immune Disease Risk Detection System')
        self.geometry('1320x880')
        self.configure(bg=C['bg'])
        self.resizable(True, True)
        self.minsize(1050, 700)
        self._build_ui()
        self._check_models()

    # ── Layout ───────────────────────────────────────────────
    def _build_ui(self):
        self.sidebar = tk.Frame(self, bg=C['surface'], width=215)
        self.sidebar.pack(side='left', fill='y')
        self.sidebar.pack_propagate(False)
        self.main = tk.Frame(self, bg=C['bg'])
        self.main.pack(side='left', fill='both', expand=True)
        self._build_sidebar()
        self._build_pages()
        self._show_page('assess')

    def _build_sidebar(self):
        # Logo
        lf = tk.Frame(self.sidebar, bg=C['surface'], pady=20)
        lf.pack(fill='x')
        tk.Label(lf, text='🧬', font=('Segoe UI',32),
                 bg=C['surface'], fg=C['accent']).pack()
        tk.Label(lf, text='ImmuneRisk', font=('Segoe UI',13,'bold'),
                 bg=C['surface'], fg=C['text']).pack()
        tk.Label(lf, text='Detection System', font=FS,
                 bg=C['surface'], fg=C['sub']).pack()

        sep(self.sidebar).pack(fill='x', padx=12, pady=6)

        self.nav_btns = {}
        nav = [
            ('assess',  '🩺  Patient Assessment'),
            ('immuno',  '🧬  Immunodeficiency'),
            ('dash',    '📊  Dashboard'),
            ('train',   '⚙️   Train Models'),
            ('pdf',     '📄  PDF Ingestion'),
            ('dataset', '🗃️   Dataset Viewer'),
            ('errors',  '🔍  Misclassifications'),
            ('info',    'ℹ️   Model Info'),
        ]
        for pid, text in nav:
            b = tk.Button(self.sidebar, text=text,
                          command=lambda p=pid: self._show_page(p),
                          bg=C['surface'], fg=C['text'],
                          activebackground=C['hover'],
                          activeforeground=C['text'],
                          relief='flat', anchor='w', font=FL,
                          padx=18, pady=11, bd=0, cursor='hand2')
            b.pack(fill='x')
            self.nav_btns[pid] = b

        self.status_var = tk.StringVar(value='● Ready')
        tk.Label(self.sidebar, textvariable=self.status_var,
                 bg=C['surface'], fg=C['green'],
                 font=FS, pady=10).pack(side='bottom', fill='x')

    def _build_pages(self):
        self.pages = {}
        for pid in ('assess','immuno','dash','train','pdf','dataset','errors','info'):
            f = tk.Frame(self.main, bg=C['bg'])
            f.place(relx=0, rely=0, relwidth=1, relheight=1)
            self.pages[pid] = f
        self._build_assess_page(self.pages['assess'])
        self._build_immuno_page(self.pages['immuno'])
        self._build_dashboard_page(self.pages['dash'])
        self._build_train_page(self.pages['train'])
        self._build_pdf_page(self.pages['pdf'])
        self._build_dataset_page(self.pages['dataset'])
        self._build_errors_page(self.pages['errors'])
        self._build_info_page(self.pages['info'])

    def _show_page(self, pid):
        for p, b in self.nav_btns.items():
            b.config(bg=C['hover'] if p == pid else C['surface'],
                     fg=C['accent'] if p == pid else C['text'])
        self.pages[pid].lift()
        if pid == 'dash': self._load_dashboard()

    # ══════════════════════════════════════════════════════════
    # SHARED RESULT DISPLAY
    # ══════════════════════════════════════════════════════════
    def _display_result(self, result, container):
        for w in container.winfo_children(): w.destroy()

        if 'error' in result:
            lbl(container, f'⚠️  {result["error"]}', FS,
                fg=C['yellow'], bg=C['card']).pack(pady=10)
            return

        lbl(container, '📋  Assessment Results', FT,
            bg=C['card']).pack(anchor='w', pady=(4,10))

        # Stat tiles
        g = tk.Frame(container, bg=C['card'])
        g.pack(fill='x', pady=4)

        def tile(par, title, val, col_idx, span=1):
            f = tk.Frame(par, bg=C['card2'], padx=14, pady=10)
            f.grid(row=0, column=col_idx, columnspan=span,
                   padx=5, pady=3, sticky='ew')
            par.columnconfigure(col_idx, weight=1)
            lbl(f, title, FS, fg=C['sub'], bg=C['card2']).pack()
            lbl(f, val, ('Segoe UI',14,'bold'),
                fg=rc(val), bg=C['card2']).pack(pady=2)

        tile(g, 'Risk Status',      result.get('risk_label','—'), 0)
        tile(g, 'Risk Score',       f"{result.get('risk_score',0)*100:.1f}%", 1)
        tile(g, 'Disease Category', result.get('disease_category','—'), 2)
        tile(g, 'Risk Level',       result.get('risk_level',
                                    result.get('risk_category','—')), 3)
        tile(g, 'Severity Score',   f"{result.get('severity_score',0)*100:.1f}%", 4)

        # Disease category probabilities
        probs = result.get('category_probabilities', {})
        if probs:
            lbl(container, 'Disease Category Probabilities',
                FS, fg=C['sub'], bg=C['card']).pack(anchor='w', pady=(12,4))
            for cat, prob in sorted(probs.items(), key=lambda x: -x[1]):
                r = tk.Frame(container, bg=C['card'])
                r.pack(fill='x', pady=1)
                lbl(r, cat, width=24, anchor='w', font=FS,
                    bg=C['card']).pack(side='left')
                bg_bar = tk.Frame(r, bg=C['card2'], height=14, width=220)
                bg_bar.pack(side='left', padx=4)
                bg_bar.pack_propagate(False)
                tk.Frame(bg_bar, bg=rc(cat), height=14,
                         width=int(220*prob)).pack(side='left')
                lbl(r, f'{prob*100:.1f}%', FS,
                    fg=rc(cat), bg=C['card']).pack(side='left', padx=4)

        # 4-level risk probabilities
        level_probs = result.get('level_probabilities', {})
        LEVEL_C = {'No Risk':C['green'],'Low Risk':C['blue'],
                   'Moderate Risk':C['yellow'],'Severe Risk':C['red']}
        if level_probs:
            lbl(container, 'Risk Level Probabilities',
                FS, fg=C['sub'], bg=C['card']).pack(anchor='w', pady=(12,4))
            for lvl in ['No Risk','Low Risk','Moderate Risk','Severe Risk']:
                prob = level_probs.get(lvl, 0.0)
                r    = tk.Frame(container, bg=C['card'])
                r.pack(fill='x', pady=2)
                col  = LEVEL_C.get(lvl, C['accent'])
                lbl(r, lvl, width=18, anchor='w', font=FS,
                    fg=col, bg=C['card']).pack(side='left')
                bg_bar = tk.Frame(r, bg=C['card2'], height=16, width=280)
                bg_bar.pack(side='left', padx=4)
                bg_bar.pack_propagate(False)
                if int(280*prob) > 0:
                    tk.Frame(bg_bar, bg=col, height=16,
                             width=int(280*prob)).pack(side='left')
                lbl(r, f'{prob*100:.1f}%', FS,
                    fg=col, bg=C['card']).pack(side='left', padx=6)

    # ══════════════════════════════════════════════════════════
    # ASSESS PAGE  (autoimmune — labs + symptoms)
    # ══════════════════════════════════════════════════════════
    def _build_assess_page(self, parent):
        hdr = tk.Frame(parent, bg=C['bg'], pady=14)
        hdr.pack(fill='x', padx=28)
        lbl(hdr, 'Patient Assessment', FH, bg=C['bg']).pack(anchor='w')
        lbl(hdr, 'Enter lab values and symptoms → instant risk prediction',
            FS, fg=C['sub'], bg=C['bg']).pack(anchor='w')

        _, sf_inner = make_scrollable(parent)
        self.assess_vars = {}

        def section(title, fields, par):
            c = tk.Frame(par, bg=C['card'], pady=2)
            c.pack(fill='x', padx=26, pady=(0,10))
            # Rounded header bar
            hf = tk.Frame(c, bg=C['card'])
            hf.pack(fill='x', padx=12, pady=(10,4))
            lbl(hf, title, FT, fg=C['accent'], bg=C['card']).pack(anchor='w')
            sep(c).pack(fill='x', padx=12)
            g = tk.Frame(c, bg=C['card'])
            g.pack(fill='x', padx=12, pady=8)
            for i, (name, default, tip) in enumerate(fields):
                row, col = divmod(i, 3)
                cell = tk.Frame(g, bg=C['card'])
                cell.grid(row=row, column=col, padx=8, pady=3, sticky='ew')
                g.columnconfigure(col, weight=1)
                lbl(cell, name, FS, fg=C['sub'], bg=C['card']).pack(anchor='w')
                var = tk.StringVar(value=str(default))
                self.assess_vars[name] = var
                ent(cell, var=var, width=13).pack(fill='x')
                if tip:
                    lbl(cell, tip, ('Segoe UI',7),
                        fg=C['border'], bg=C['card']).pack(anchor='w')

        section('👤  Demographics', [
            ('Age',45,'years'),
            ('Gender_numeric',1,'1=Male  0=Female'),
            ('Sickness_Duration_Months',6,'months'),
        ], sf_inner)

        section('🩸  Complete Blood Count', [
            ('RBC_Count',4.5,'M/μL'),('Hemoglobin',13.5,'g/dL'),
            ('Hematocrit',40.0,'%'),('MCV',90.0,'fL'),('MCH',30.0,'pg'),
            ('MCHC',33.0,'g/dL'),('RDW',13.0,'%'),
            ('Reticulocyte_Count',1.5,'%'),('WBC_Count',7000,'cells/μL'),
            ('Neutrophils',60.0,'%'),('Lymphocytes',30.0,'%'),
            ('Monocytes',6.0,'%'),('Eosinophils',3.0,'%'),
            ('Basophils',1.0,'%'),('PLT_Count',250000,'cells/μL'),
            ('MPV',9.0,'fL'),
        ], sf_inner)

        section('🔬  Inflammatory Markers', [
            ('ESR',15.0,'mm/hr'),('CRP',1.0,'mg/L'),
            ('ANA',0,'0=Neg  1=Pos'),('C3',1.0,'g/L'),
            ('C4',0.2,'g/L'),('MBL_Level',1.5,'μg/mL'),
            ('Esbach',1.0,'g/L'),
        ], sf_inner)

        sym_c = tk.Frame(sf_inner, bg=C['card'])
        sym_c.pack(fill='x', padx=26, pady=(0,10))
        lbl(sym_c, '🤒  Symptoms', FT, fg=C['accent'],
            bg=C['card']).pack(anchor='w', padx=12, pady=(10,4))
        sep(sym_c).pack(fill='x', padx=12)
        sg = tk.Frame(sym_c, bg=C['card'])
        sg.pack(fill='x', padx=12, pady=8)
        symptoms = [
            'Low-grade fever','Fatigue or chronic tiredness','Dizziness',
            'Weight loss','Rashes and skin lesions','Stiffness in the joints',
            'Brittle hair or hair loss','Dry eyes and/or mouth',
            "General 'unwell' feeling",'Joint pain',
        ]
        for i, s in enumerate(symptoms):
            r, c2 = divmod(i, 2)
            var = tk.IntVar(value=0)
            self.assess_vars[s] = var
            tk.Checkbutton(sg, text=s, variable=var,
                           bg=C['card'], fg=C['text'],
                           activebackground=C['card'],
                           selectcolor=C['input'],
                           font=FL, cursor='hand2').grid(
                row=r, column=c2, sticky='w', padx=8, pady=3)

        bf = tk.Frame(sf_inner, bg=C['bg'])
        bf.pack(fill='x', padx=26, pady=10)
        rbtn(bf, '  🔍  Run Risk Assessment  ',
             self._run_assessment, width=240, height=38).pack(pady=4)

        self._result_c = tk.Frame(sf_inner, bg=C['card'])
        self._result_c.pack(fill='x', padx=26, pady=(0,28))
        inner_r = tk.Frame(self._result_c, bg=C['card'])
        inner_r.pack(fill='x', padx=14, pady=10)
        lbl(inner_r, 'Results will appear here after assessment.',
            FS, fg=C['sub'], bg=C['card']).pack()
        self._assess_result_inner = inner_r

    def _run_assessment(self):
        def run():
            try:
                patient = {}
                for key, var in self.assess_vars.items():
                    v = var.get()
                    try:    patient[key] = float(v) if not isinstance(v, int) else v
                    except: patient[key] = v
                self.status_var.set('● Assessing…')
                from ml_model import predict_patient
                result = predict_patient(patient)
                self.after(0, lambda: self._display_result(
                    result, self._assess_result_inner))
                self.status_var.set('● Ready')
            except Exception as e:
                self.after(0, lambda: messagebox.showerror('Error', str(e)))
                self.status_var.set('● Error')
        threading.Thread(target=run, daemon=True).start()

    # ══════════════════════════════════════════════════════════
    # IMMUNODEFICIENCY PAGE
    # ══════════════════════════════════════════════════════════
    def _build_immuno_page(self, parent):
        hdr = tk.Frame(parent, bg=C['bg'], pady=14)
        hdr.pack(fill='x', padx=28)
        lbl(hdr, 'Immunodeficiency Assessment', FH, bg=C['bg']).pack(anchor='w')
        lbl(hdr, 'Enter clinical flags and demographics specific to immunodeficiency screening',
            FS, fg=C['sub'], bg=C['bg']).pack(anchor='w')

        _, sf = make_scrollable(parent)
        self.immuno_vars = {}

        # ── Demographics ─────────────────────────────────────
        dem = tk.Frame(sf, bg=C['card'])
        dem.pack(fill='x', padx=26, pady=(0,10))
        lbl(dem, '👤  Patient Demographics', FT, fg=C['accent'],
            bg=C['card']).pack(anchor='w', padx=12, pady=(10,4))
        sep(dem).pack(fill='x', padx=12)
        dg = tk.Frame(dem, bg=C['card'])
        dg.pack(fill='x', padx=12, pady=8)

        demo_fields = [
            ('Age', 30, 'years'),
            ('Gender_numeric', 1, '1=Male  0=Female'),
            ('Sickness_Duration_Months', 0, 'months (0 if unknown)'),
        ]
        for i, (name, default, tip) in enumerate(demo_fields):
            cell = tk.Frame(dg, bg=C['card'])
            cell.grid(row=0, column=i, padx=8, pady=3, sticky='ew')
            dg.columnconfigure(i, weight=1)
            lbl(cell, name, FS, fg=C['sub'], bg=C['card']).pack(anchor='w')
            var = tk.StringVar(value=str(default))
            self.immuno_vars[name] = var
            ent(cell, var=var, width=13).pack(fill='x')
            lbl(cell, tip, ('Segoe UI',7), fg=C['border'],
                bg=C['card']).pack(anchor='w')

        # ── Age range selector ───────────────────────────────
        ag = tk.Frame(sf, bg=C['card'])
        ag.pack(fill='x', padx=26, pady=(0,10))
        lbl(ag, '📅  Age Range', FT, fg=C['accent'],
            bg=C['card']).pack(anchor='w', padx=12, pady=(10,4))
        sep(ag).pack(fill='x', padx=12)
        agf = tk.Frame(ag, bg=C['card'])
        agf.pack(fill='x', padx=12, pady=8)
        lbl(agf, 'Select age range:', FS, fg=C['sub'],
            bg=C['card']).pack(side='left', padx=(0,8))
        age_ranges = ['2-5','6-11','12-17','18-24','25-34','35-44','45-54','55-64','65+']
        self._age_range_var = tk.StringVar(value='25-34')
        age_combo = ttk.Combobox(agf, textvariable=self._age_range_var,
                                  values=age_ranges, width=12, state='readonly')
        age_combo.pack(side='left')
        lbl(agf, '← Used when exact age unknown',
            FS, fg=C['border'], bg=C['card']).pack(side='left', padx=8)

        # ── Clinical flags ───────────────────────────────────
        cf_card = tk.Frame(sf, bg=C['card'])
        cf_card.pack(fill='x', padx=26, pady=(0,10))
        lbl(cf_card, '🏥  Clinical Flags', FT, fg=C['accent'],
            bg=C['card']).pack(anchor='w', padx=12, pady=(10,4))
        sep(cf_card).pack(fill='x', padx=12)
        cfg = tk.Frame(cf_card, bg=C['card'])
        cfg.pack(fill='x', padx=12, pady=8)

        flags = [
            ('Had_Transplant',    'Had Stem Cell / Organ Transplant',
             'Prior transplant indicates severe immunodeficiency'),
            ('Ig_Therapy_Ever',   'Received Immunoglobulin (Ig) Therapy',
             'Ig therapy = established antibody deficiency'),
            ('Family_HX_PI',      'Family History of Primary Immunodeficiency',
             'Genetic / hereditary risk factor'),
        ]
        for i, (key, label, tip) in enumerate(flags):
            var = tk.IntVar(value=0)
            self.immuno_vars[key] = var
            cell = tk.Frame(cfg, bg=C['card'])
            cell.grid(row=i, column=0, sticky='w', padx=8, pady=4)
            tk.Checkbutton(cell, text=label, variable=var,
                           bg=C['card'], fg=C['text'],
                           activebackground=C['card'],
                           selectcolor=C['input'],
                           font=FL, cursor='hand2').pack(side='left')
            lbl(cell, f'  ({tip})', FS,
                fg=C['sub'], bg=C['card']).pack(side='left')

        # ── Diagnosis selector ───────────────────────────────
        diag_card = tk.Frame(sf, bg=C['card'])
        diag_card.pack(fill='x', padx=26, pady=(0,10))
        lbl(diag_card, '🔬  Known / Suspected Diagnosis', FT, fg=C['accent'],
            bg=C['card']).pack(anchor='w', padx=12, pady=(10,4))
        sep(diag_card).pack(fill='x', padx=12)
        diagf = tk.Frame(diag_card, bg=C['card'])
        diagf.pack(fill='x', padx=12, pady=8)

        lbl(diagf, 'Category:', FS, fg=C['sub'],
            bg=C['card']).pack(side='left', padx=(0,6))
        categories = [
            'Unknown / Screening',
            'Predominantly Antibody Deficiencies',
            'Combined immunodeficiencies with associated or syndromic features',
            'Immunodeficiencies affecting cellular and humoral immunity',
            'Congenital defects of phagocyte number or function',
            'Diseases of Immune Dysregulation',
            'Complement Deficiencies',
            'Defects in Intrinsic and Innate Immunity',
            'Other',
        ]
        self._immuno_cat_var = tk.StringVar(value='Unknown / Screening')
        cat_combo = ttk.Combobox(diagf, textvariable=self._immuno_cat_var,
                                  values=categories, width=50, state='readonly')
        cat_combo.pack(side='left', padx=4)

        # Specific diagnosis
        diagf2 = tk.Frame(diag_card, bg=C['card'])
        diagf2.pack(fill='x', padx=12, pady=(0,6))
        lbl(diagf2, 'Specific diagnosis (if known):', FS,
            fg=C['sub'], bg=C['card']).pack(side='left', padx=(0,6))
        self._specific_diag_var = tk.StringVar(value='')
        ent(diagf2, var=self._specific_diag_var, width=44).pack(side='left')

        specific_diags = [
            'Common Variable Immune Deficiency (CVID)',
            'Chronic Granulomatous Disease',
            'Agammaglobulinemia',
            'DiGeorge syndrome',
            'Severe Combined Immune Deficiency (SCID)',
            'Hypogammaglobulinemia',
            'Wiskott-Aldrich Syndrome',
            'Hyper IgM Syndrome',
            'IgA Deficiency',
            'Combined Immune Deficiency',
            'Ataxia telangiectasia',
            'Complement Deficiency',
            'Leukocyte adhesion deficiency',
        ]
        lbl(diagf2, '  Quick select:', FS, fg=C['border'],
            bg=C['card']).pack(side='left', padx=(10,4))
        quick = ttk.Combobox(diagf2, values=specific_diags, width=36)
        quick.pack(side='left')
        quick.bind('<<ComboboxSelected>>',
                   lambda e: self._specific_diag_var.set(quick.get()))

        # ── Available lab values (optional) ─────────────────
        lab_opt = tk.Frame(sf, bg=C['card'])
        lab_opt.pack(fill='x', padx=26, pady=(0,10))
        lbl(lab_opt, '🧪  Lab Values (if available)', FT, fg=C['accent'],
            bg=C['card']).pack(anchor='w', padx=12, pady=(10,4))
        sep(lab_opt).pack(fill='x', padx=12)
        lbl(lab_opt,
            'These are optional — leave blank if not available. '
            'The model will use clinical flags for prediction.',
            FS, fg=C['sub'], bg=C['card']).pack(anchor='w', padx=12, pady=4)
        lg = tk.Frame(lab_opt, bg=C['card'])
        lg.pack(fill='x', padx=12, pady=4)
        opt_labs = [
            ('WBC_Count','','cells/μL'),('Lymphocytes','','%'),
            ('Neutrophils','','%'),('IgG level','','mg/dL (optional)'),
            ('ESR','','mm/hr'),('CRP','','mg/L'),
        ]
        for i, (name, default, tip) in enumerate(opt_labs):
            row, col = divmod(i, 3)
            cell = tk.Frame(lg, bg=C['card'])
            cell.grid(row=row, column=col, padx=8, pady=3, sticky='ew')
            lg.columnconfigure(col, weight=1)
            lbl(cell, name, FS, fg=C['sub'], bg=C['card']).pack(anchor='w')
            var = tk.StringVar(value=str(default))
            self.immuno_vars[name] = var
            ent(cell, var=var, width=13).pack(fill='x')
            lbl(cell, tip, ('Segoe UI',7), fg=C['border'],
                bg=C['card']).pack(anchor='w')

        bf = tk.Frame(sf, bg=C['bg'])
        bf.pack(fill='x', padx=26, pady=10)
        rbtn(bf, '  🔍  Run Immunodeficiency Assessment  ',
             self._run_immuno_assessment, width=290, height=38).pack(pady=4)

        self._immuno_result_inner = tk.Frame(sf, bg=C['card'])
        self._immuno_result_inner.pack(fill='x', padx=26, pady=(0,28))
        ri = tk.Frame(self._immuno_result_inner, bg=C['card'])
        ri.pack(fill='x', padx=14, pady=10)
        lbl(ri, 'Results will appear here.',
            FS, fg=C['sub'], bg=C['card']).pack()
        self._immuno_result_container = ri

    def _run_immuno_assessment(self):
        def run():
            try:
                patient = {}
                # Age from range if exact not given
                age_range_map = {
                    '2-5':3,'6-11':8,'12-17':14,'18-24':21,
                    '25-34':29,'35-44':39,'45-54':49,'55-64':59,'65+':70
                }
                for key, var in self.immuno_vars.items():
                    v = var.get()
                    try:    patient[key] = float(v) if v != '' and not isinstance(v,int) else (v if isinstance(v,int) else 0)
                    except: patient[key] = 0

                if patient.get('Age', 0) == 0:
                    patient['Age'] = age_range_map.get(
                        self._age_range_var.get(), 30)

                # Set diagnosis for category classification
                specific = self._specific_diag_var.get().strip()
                patient['Diagnosis'] = specific if specific else 'Common Variable Immune Deficiency (CVID)'

                # Boost severity for Ig therapy (clinical proxy)
                if patient.get('Ig_Therapy_Ever', 0):
                    patient['Antibody_Score'] = patient.get('Antibody_Score', 0) + 2

                self.status_var.set('● Assessing…')
                from ml_model import predict_patient
                result = predict_patient(patient)
                # Override category to Immunodeficiency
                result['disease_category'] = 'Immunodeficiency'
                self.after(0, lambda: self._display_result(
                    result, self._immuno_result_container))
                self.status_var.set('● Ready')
            except Exception as e:
                self.after(0, lambda: messagebox.showerror('Error', str(e)))
                self.status_var.set('● Error')
        threading.Thread(target=run, daemon=True).start()

    # ══════════════════════════════════════════════════════════
    # DASHBOARD PAGE
    # ══════════════════════════════════════════════════════════
    def _build_dashboard_page(self, parent):
        hdr = tk.Frame(parent, bg=C['bg'], pady=14)
        hdr.pack(fill='x', padx=28)
        lbl(hdr, 'Test Results Dashboard', FH, bg=C['bg']).pack(anchor='w')
        lbl(hdr, '70 / 30 stratified train-test split — all metrics on held-out test data',
            FS, fg=C['sub'], bg=C['bg']).pack(anchor='w')
        rbtn(hdr, '  🔄  Refresh  ', self._load_dashboard,
             color=C['card2'], width=120, height=32).pack(anchor='e')

        _, self._dash_inner = make_scrollable(parent)

    def _load_dashboard(self):
        for w in self._dash_inner.winfo_children(): w.destroy()

        path = BASE_DIR / 'outputs' / 'test_results.json'
        if not path.exists():
            lbl(self._dash_inner,
                'No test results found.\nGo to Train Models and run training first.',
                FL, fg=C['yellow'], bg=C['bg']).pack(pady=40)
            return

        with open(path) as f: data = json.load(f)
        rm  = data.get('risk_model', {})
        cm  = data.get('category_model', {})
        sm  = data.get('strat_model', {})
        ds  = data.get('dataset', {})

        PAD = 26

        def section(title):
            c = tk.Frame(self._dash_inner, bg=C['card'])
            c.pack(fill='x', padx=PAD, pady=(0,12))
            lbl(c, title, FT, fg=C['accent'],
                bg=C['card']).pack(anchor='w', padx=14, pady=(10,4))
            sep(c).pack(fill='x', padx=14)
            body = tk.Frame(c, bg=C['card'])
            body.pack(fill='x', padx=14, pady=8)
            return body

        def tiles(parent, stats):
            g = tk.Frame(parent, bg=C['card'])
            g.pack(fill='x', pady=4)
            for i, (lab, val, col) in enumerate(stats):
                f = tk.Frame(g, bg=C['card2'], padx=12, pady=8)
                f.grid(row=0, column=i, padx=5, sticky='ew')
                g.columnconfigure(i, weight=1)
                lbl(f, lab, FS, fg=C['sub'], bg=C['card2']).pack()
                lbl(f, str(val), ('Segoe UI',15,'bold'),
                    fg=col, bg=C['card2']).pack(pady=2)

        trained = data.get('trained_at','—')[:19].replace('T',' ')

        # ── Overview ─────────────────────────────────────────
        ov = section('📌  Overview')
        tiles(ov, [
            ('Total Samples',    f"{ds.get('total_records',0):,}", C['text']),
            ('Train / Test',     data.get('train_test_split','70/30'), C['accent']),
            ('Disease Classes',  ds.get('diagnosis_count','—'), C['text']),
            ('Avg Age',          ds.get('avg_age','—'), C['text']),
            ('Trained At',       trained, C['sub']),
        ])

        # ── Model 1 ───────────────────────────────────────────
        r1 = section('🔮  Model 1: Risk Detection  (Random Forest, 70/30)')
        tiles(r1, [
            ('Train',     f"{rm.get('train_size',0):,}",  C['text']),
            ('Test',      f"{rm.get('test_size',0):,}",   C['text']),
            ('Accuracy',  f"{rm.get('accuracy',0)*100:.2f}%", C['green']),
            ('Precision', f"{rm.get('precision',0)*100:.2f}%",C['accent']),
            ('Recall',    f"{rm.get('recall',0)*100:.2f}%",   C['accent']),
            ('F1',        f"{rm.get('f1',0)*100:.2f}%",       C['yellow']),
            ('ROC-AUC',   f"{rm.get('roc_auc',0):.4f}",       C['green']),
            ('CV AUC',    f"{rm.get('cv_auc_mean',0):.4f}±{rm.get('cv_auc_std',0):.4f}", C['sub']),
        ])
        viz1 = tk.Frame(r1, bg=C['card'])
        viz1.pack(fill='x', pady=6)
        cm1d = rm.get('confusion_matrix',[[0,0],[0,0]])
        ConfusionMatrix(viz1, cm1d, rm.get('label_names',['Normal','At Risk']),
                        width=290, height=210).pack(side='left', padx=8)
        roc  = rm.get('roc_curve',{})
        ROCCurve(viz1, roc.get('fpr',[]), roc.get('tpr',[]),
                 rm.get('roc_auc',0), width=290, height=210).pack(side='left', padx=8)
        fi   = rm.get('feature_importance',[])
        if fi:
            fi_d = {r['feature']: round(r['importance'],4) for r in fi[:12]}
            BarChart(viz1, fi_d, title='Top Feature Importances',
                     color=C['accent'], width=360, height=210).pack(side='left', padx=8)

        # ── Model 2 ───────────────────────────────────────────
        r2 = section('🏷️  Model 2: Disease Category  (Gradient Boosting, 70/30)')
        tiles(r2, [
            ('Train',       f"{cm.get('train_size',0):,}", C['text']),
            ('Test',        f"{cm.get('test_size',0):,}",  C['text']),
            ('Accuracy',    f"{cm.get('accuracy',0)*100:.2f}%",    C['green']),
            ('F1 Weighted', f"{cm.get('f1_weighted',0)*100:.2f}%", C['accent']),
        ])
        viz2 = tk.Frame(r2, bg=C['card'])
        viz2.pack(fill='x', pady=6)
        cm2d = cm.get('confusion_matrix',[])
        cm2l = cm.get('label_names',[])
        if cm2d and cm2l:
            ConfusionMatrix(viz2, cm2d, cm2l,
                            width=320, height=210).pack(side='left', padx=8)
        td = cm.get('test_distribution',{})
        if td:
            PieChart(viz2, td, title='Test Set Distribution',
                     width=260, height=210).pack(side='left', padx=8)
        rep = cm.get('classification_report',{})
        if rep:
            tf = tk.Frame(r2, bg=C['card2'])
            tf.pack(fill='x', pady=4)
            for ci, h in enumerate(['Class','Precision','Recall','F1','Support']):
                lbl(tf, h, FS, fg=C['accent'],
                    bg=C['card2']).grid(row=0, column=ci, padx=10, pady=4, sticky='w')
            ri = 1
            for cls, vals in rep.items():
                if cls in ('accuracy','macro avg','weighted avg'): continue
                if isinstance(vals, dict):
                    for ci, v in enumerate([cls,
                        f"{vals.get('precision',0):.3f}",
                        f"{vals.get('recall',0):.3f}",
                        f"{vals.get('f1-score',0):.3f}",
                        str(int(vals.get('support',0)))]):
                        lbl(tf, v, FS, fg=C['text'],
                            bg=C['card2']).grid(row=ri, column=ci, padx=10, pady=2, sticky='w')
                    ri += 1

        # ── Model 3 — 4-level stratification ─────────────────
        r3 = section('🎯  Model 3: 4-Level Risk Stratification  (Gradient Boosting, 70/30)')
        tiles(r3, [
            ('Train',      f"{sm.get('train_size',0):,}", C['text']),
            ('Test',       f"{sm.get('test_size',0):,}",  C['text']),
            ('Accuracy',   f"{sm.get('accuracy',0)*100:.2f}%",    C['green']),
            ('F1 Weighted',f"{sm.get('f1_weighted',0)*100:.2f}%", C['accent']),
        ])

        LEVEL_C = {'No Risk':C['green'],'Low Risk':C['blue'],
                   'Moderate Risk':C['yellow'],'Severe Risk':C['red']}
        LEVEL_ORDER = ['No Risk','Low Risk','Moderate Risk','Severe Risk']

        risk_dist = sm.get('risk_distribution',{})
        ordered   = {k: risk_dist[k] for k in LEVEL_ORDER if k in risk_dist}

        viz3 = tk.Frame(r3, bg=C['card'])
        viz3.pack(fill='x', pady=6)

        if ordered:
            PieChart(viz3, ordered, title='Risk Distribution',
                     width=260, height=220).pack(side='left', padx=8)
            BarChart(viz3, ordered, title='4-Level Risk Counts',
                     color=C['yellow'], width=340, height=220).pack(side='left', padx=8)

        # Confusion matrix for 4-level
        cm3d = sm.get('confusion_matrix',[])
        if cm3d:
            ConfusionMatrix(viz3, cm3d, LEVEL_ORDER,
                            width=300, height=220).pack(side='left', padx=8)

        # Risk level count tiles
        if ordered:
            lbl(r3, 'Full dataset breakdown:', FS, fg=C['sub'],
                bg=C['card']).pack(anchor='w', pady=(8,2))
            tiles(r3, [(lvl, f"{ordered.get(lvl,0):,}", LEVEL_C.get(lvl,C['text']))
                       for lvl in LEVEL_ORDER if lvl in ordered])

        # ── Dataset overview ──────────────────────────────────
        r4 = section('📁  Dataset Overview')
        viz4 = tk.Frame(r4, bg=C['card'])
        viz4.pack(fill='x', pady=6)

        cat_dist = ds.get('category_dist',{})
        if cat_dist:
            PieChart(viz4, cat_dist, title='Disease Categories',
                     width=260, height=220).pack(side='left', padx=8)

        top30 = ds.get('diagnosis_top30',{})
        if top30:
            BarChart(viz4, top30, title='Top 30 Diagnoses',
                     color=C['purple'], width=400, height=300).pack(side='left', padx=8)

        age_h = ds.get('age_histogram',{})
        if age_h:
            counts = age_h.get('counts',[])
            bins   = age_h.get('bins',[])
            if counts:
                age_data = {f"{int(bins[i])}-{int(bins[i+1])}": c
                            for i, c in enumerate(counts) if i < len(bins)-1}
                BarChart(viz4, age_data, title='Age Distribution',
                         color=C['blue'], width=300, height=220).pack(side='left', padx=8)

        tk.Frame(self._dash_inner, height=24, bg=C['bg']).pack()

    # ══════════════════════════════════════════════════════════
    # TRAIN PAGE
    # ══════════════════════════════════════════════════════════
    def _build_train_page(self, parent):
        hdr = tk.Frame(parent, bg=C['bg'], pady=14)
        hdr.pack(fill='x', padx=28)
        lbl(hdr, 'Train Models', FH, bg=C['bg']).pack(anchor='w')
        lbl(hdr, 'Train / retrain all 3 ML models with 70/30 split',
            FS, fg=C['sub'], bg=C['bg']).pack(anchor='w')

        ds_c = tk.Frame(parent, bg=C['card'])
        ds_c.pack(fill='x', padx=28, pady=8)
        lbl(ds_c, '📁  Dataset Status', FT, fg=C['accent'],
            bg=C['card']).pack(anchor='w', padx=14, pady=(10,4))
        sep(ds_c).pack(fill='x', padx=14)
        self._ds_status_var = tk.StringVar(value='Checking…')
        tk.Label(ds_c, textvariable=self._ds_status_var,
                 font=FL, bg=C['card'], fg=C['text'],
                 justify='left').pack(anchor='w', padx=14, pady=6)
        self._refresh_ds_status()

        opt_c = tk.Frame(parent, bg=C['card'])
        opt_c.pack(fill='x', padx=28, pady=6)
        lbl(opt_c, '⚙️   Options', FT, fg=C['accent'],
            bg=C['card']).pack(anchor='w', padx=14, pady=(10,4))
        sep(opt_c).pack(fill='x', padx=14)
        og = tk.Frame(opt_c, bg=C['card'])
        og.pack(fill='x', padx=14, pady=8)
        lbl(og, 'Custom dataset (optional):', FS, fg=C['sub'],
            bg=C['card']).grid(row=0, column=0, sticky='w', pady=4)
        self._custom_ds = tk.StringVar()
        ent(og, var=self._custom_ds, width=44).grid(
            row=0, column=1, padx=6, sticky='ew')
        rbtn(og, 'Browse', self._browse_ds,
             color=C['card2'], width=80, height=28).grid(
            row=0, column=2, padx=4)
        og.columnconfigure(1, weight=1)

        log_c = tk.Frame(parent, bg=C['card'])
        log_c.pack(fill='x', padx=28, pady=6)
        self._train_log = scrolledtext.ScrolledText(
            log_c, height=9, bg=C['input'], fg=C['text'],
            font=FM, relief='flat', state='disabled',
            insertbackground=C['text'])
        self._train_log.pack(fill='x', padx=14, pady=8)
        self._train_prog = ttk.Progressbar(log_c, mode='determinate', length=500)
        self._train_prog.pack(padx=14, pady=(0,4))
        self._train_status = tk.StringVar(value='Ready to train')
        tk.Label(log_c, textvariable=self._train_status,
                 font=FS, bg=C['card'], fg=C['sub']).pack(pady=(0,6))

        rbtn(parent, '  🚀  Start Training  ',
             self._start_training, width=200, height=38).pack(pady=10)

    def _refresh_ds_status(self):
        try:
            import pandas as pd
            lines = []
            for name, path in [
                ('Autoimmune+Normal', BASE_DIR/'data'/'combined_with_normal_2000.csv'),
                ('Combined All',      BASE_DIR/'data'/'combined_all.csv'),
                ('Immunodeficiency',  BASE_DIR/'data'/'immunodeficiency_dataset.csv'),
            ]:
                if path.exists():
                    n = len(pd.read_csv(path, low_memory=False))
                    lines.append(f'✅  {name}: {n:,} records')
                else:
                    lines.append(f'⚠️   {name}: not found')
            self._ds_status_var.set('\n'.join(lines))
        except Exception as e:
            self._ds_status_var.set(str(e))

    def _browse_ds(self):
        p = filedialog.askopenfilename(
            title='Select CSV', filetypes=[('CSV','*.csv'),('All','*.*')])
        if p: self._custom_ds.set(p)

    def _start_training(self):
        def run():
            try:
                self._log_train('Starting 70/30 training pipeline…\n')
                self._train_prog['value'] = 0
                self._train_status.set('Initialising…')
                from ml_model import train_all
                def pcb(step, total, msg):
                    pct = int(step/total*100)
                    self._train_prog['value'] = pct
                    self._train_status.set(msg)
                    self._log_train(f'[{pct}%] {msg}\n')
                    self.update_idletasks()
                meta = train_all(
                    data_path=self._custom_ds.get().strip() or None,
                    progress_callback=pcb)
                self._train_prog['value'] = 100
                self._train_status.set('✅ Done!')
                rm = meta['risk_model']
                sm = meta['strat_model']
                self._log_train(
                    f'\n✅ Training complete!\n'
                    f'  Risk AUC  : {rm.get("roc_auc")}\n'
                    f'  Cat Acc   : {meta["category_model"].get("accuracy")}\n'
                    f'  Strat Acc : {sm.get("accuracy")}\n'
                    f'  Risk dist : {sm.get("risk_distribution")}\n'
                    f'  → Open Dashboard tab to view full results\n')
                self.status_var.set('● Models ready')
                messagebox.showinfo('Training Complete',
                    f'All models trained!\n'
                    f'Risk AUC: {rm.get("roc_auc")}\n'
                    f'Open the Dashboard tab for full results.')
            except Exception as e:
                self._log_train(f'ERROR: {e}\n')
                self._train_status.set(f'Error: {e}')
                messagebox.showerror('Training Failed', str(e))
        threading.Thread(target=run, daemon=True).start()

    def _log_train(self, msg):
        self._train_log.configure(state='normal')
        self._train_log.insert(tk.END, msg)
        self._train_log.see(tk.END)
        self._train_log.configure(state='disabled')

    # ══════════════════════════════════════════════════════════
    # PDF PAGE
    # ══════════════════════════════════════════════════════════
    def _build_pdf_page(self, parent):
        hdr = tk.Frame(parent, bg=C['bg'], pady=14)
        hdr.pack(fill='x', padx=28)
        lbl(hdr, 'PDF Ingestion', FH, bg=C['bg']).pack(anchor='w')
        lbl(hdr,
            'Import clinical reports — digital or scanned (OCR) → extract → dataset → retrain',
            FS, fg=C['sub'], bg=C['bg']).pack(anchor='w')

        drop = tk.Frame(parent, bg=C['card'])
        drop.pack(fill='x', padx=28, pady=8)
        dz = tk.Frame(drop, bg=C['card2'], pady=22)
        dz.pack(fill='x', padx=14, pady=10)
        lbl(dz, '📄', ('Segoe UI',32), bg=C['card2'],
            fg=C['accent']).pack()
        lbl(dz, 'Select PDF file or folder', FT,
            bg=C['card2']).pack(pady=4)
        lbl(dz, 'Auto-detects scanned PDFs and runs OCR automatically',
            FS, fg=C['sub'], bg=C['card2']).pack()
        br = tk.Frame(dz, bg=C['card2'])
        br.pack(pady=10)
        rbtn(br, '  📄  Select PDF  ', self._sel_pdf_file,
             width=160, height=34).pack(side='left', padx=6)
        rbtn(br, '  📁  Select Folder  ', self._sel_pdf_folder,
             color=C['purple'], width=170, height=34).pack(side='left', padx=6)

        opt = tk.Frame(parent, bg=C['card'])
        opt.pack(fill='x', padx=28, pady=4)
        oi = tk.Frame(opt, bg=C['card'])
        oi.pack(fill='x', padx=14, pady=8)
        self._pdf_retrain = tk.BooleanVar(value=True)
        tk.Checkbutton(oi, text='Retrain models after ingestion',
                       variable=self._pdf_retrain,
                       bg=C['card'], fg=C['text'],
                       activebackground=C['card'],
                       selectcolor=C['input'], font=FL).pack(anchor='w')
        self._pdf_sel_var = tk.StringVar(value='No file selected')
        tk.Label(opt, textvariable=self._pdf_sel_var,
                 font=FS, bg=C['card'], fg=C['sub']).pack(
                     anchor='w', padx=14, pady=(0,6))

        log_c = tk.Frame(parent, bg=C['card'])
        log_c.pack(fill='both', expand=True, padx=28, pady=6)
        lbl(log_c, 'Ingestion Log', FT, fg=C['accent'],
            bg=C['card']).pack(anchor='w', padx=14, pady=(10,4))
        sep(log_c).pack(fill='x', padx=14)
        self._pdf_log = scrolledtext.ScrolledText(
            log_c, height=12, bg=C['input'], fg=C['text'],
            font=FM, relief='flat', state='disabled')
        self._pdf_log.pack(fill='both', expand=True, padx=14, pady=8)
        self._pdf_prog = ttk.Progressbar(log_c, mode='indeterminate')
        self._pdf_prog.pack(fill='x', padx=14, pady=(0,6))

    def _sel_pdf_file(self):
        p = filedialog.askopenfilename(
            title='Select Clinical PDF',
            filetypes=[('PDF','*.pdf'),('All','*.*')])
        if p:
            self._pdf_sel_var.set(f'📄 {Path(p).name}')
            self._run_pdf(p, False)

    def _sel_pdf_folder(self):
        p = filedialog.askdirectory(title='Select folder with PDFs')
        if p:
            n = len(list(Path(p).glob('*.pdf')))
            self._pdf_sel_var.set(f'📁 {Path(p).name} ({n} PDFs)')
            self._run_pdf(p, True)

    def _run_pdf(self, path, is_folder):
        def run():
            self._pdf_prog.start()
            try:
                from pdf_ingestion import ingest_pdf, ingest_folder
                retrain = self._pdf_retrain.get()
                def pcb(msg): self._log_pdf(f'  → {msg}\n'); self.update_idletasks()
                if is_folder:
                    self._log_pdf(f'📁 Folder: {path}\n')
                    results = ingest_folder(path, retrain_after_all=retrain)
                    ok = sum(1 for r in results if r['status'] in ('success','partial'))
                    self._log_pdf(f'✅ {ok}/{len(results)} PDFs ingested.\n')
                else:
                    self._log_pdf(f'📄 {Path(path).name}\n')
                    r = ingest_pdf(path, retrain=retrain, progress_callback=pcb)
                    m = r.get('extraction_method','unknown').upper()
                    self._log_pdf(
                        f'Extraction : {"🔍" if m=="OCR" else "📝"} {m}\n'
                        f'Status     : {r["status"]}\n'
                        f'Patient    : {r["patient_data"].get("Patient_ID")}\n'
                        f'Diagnosis  : {r["patient_data"].get("Diagnosis")}\n'
                        f'Fields     : {len([v for v in r["patient_data"].values() if v not in (None,"")])}\n'
                        f'Updated    : {r["dataset_updated"]}\n'
                        f'Retrained  : {r["retrained"]}\n'
                    )
                    for e in r.get('errors',[]): self._log_pdf(f'⚠️  {e}\n')
                    self._log_pdf('─'*50+'\n')
            except Exception as e:
                self._log_pdf(f'ERROR: {e}\n')
            finally:
                self._pdf_prog.stop()
        threading.Thread(target=run, daemon=True).start()

    def _log_pdf(self, msg):
        self._pdf_log.configure(state='normal')
        self._pdf_log.insert(tk.END, msg)
        self._pdf_log.see(tk.END)
        self._pdf_log.configure(state='disabled')

    # ══════════════════════════════════════════════════════════
    # DATASET PAGE
    # ══════════════════════════════════════════════════════════
    def _build_dataset_page(self, parent):
        hdr = tk.Frame(parent, bg=C['bg'], pady=14)
        hdr.pack(fill='x', padx=28)
        lbl(hdr, 'Dataset Viewer', FH, bg=C['bg']).pack(anchor='w')

        ctrl = tk.Frame(parent, bg=C['card'])
        ctrl.pack(fill='x', padx=28, pady=6)
        ci = tk.Frame(ctrl, bg=C['card'])
        ci.pack(fill='x', padx=14, pady=8)

        self._ds_src = tk.StringVar(value='combined')
        lbl(ci, 'Dataset:', FS, fg=C['sub'],
            bg=C['card']).pack(side='left')
        for val, text in [('combined','Combined All'),
                           ('autoimmune','Autoimmune+Normal'),
                           ('immunodef','Immunodeficiency')]:
            tk.Radiobutton(ci, text=text, variable=self._ds_src,
                           value=val, command=self._load_ds_view,
                           bg=C['card'], fg=C['text'],
                           activebackground=C['card'],
                           selectcolor=C['input'],
                           font=FL).pack(side='left', padx=8)
        lbl(ci, 'Filter:', FS, fg=C['sub'],
            bg=C['card']).pack(side='left', padx=(16,4))
        self._ds_filter = tk.StringVar()
        ent(ci, var=self._ds_filter, width=20).pack(side='left')
        rbtn(ci, 'Apply', self._load_ds_view,
             color=C['card2'], width=70, height=26).pack(side='left', padx=5)
        self._ds_count = tk.StringVar()
        tk.Label(ci, textvariable=self._ds_count, font=FS,
                 bg=C['card'], fg=C['sub']).pack(side='right', padx=8)

        tf = tk.Frame(parent, bg=C['bg'])
        tf.pack(fill='both', expand=True, padx=28, pady=6)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('D.Treeview',
                         background=C['surface'], foreground=C['text'],
                         rowheight=24, fieldbackground=C['surface'], font=FS)
        style.configure('D.Treeview.Heading',
                         background=C['card2'], foreground=C['accent'],
                         font=('Segoe UI',9,'bold'),
                         relief='flat')
        style.map('D.Treeview',
                   background=[('selected', C['accent'])],
                   foreground=[('selected', '#ffffff')])

        cols = ('Patient_ID','Age','Gender','Diagnosis',
                'WBC_Count','Lymphocytes','ESR','CRP','ANA')
        self._ds_tree = ttk.Treeview(tf, columns=cols,
                                      show='headings', style='D.Treeview')
        for c in cols:
            self._ds_tree.heading(c, text=c)
            self._ds_tree.column(c, width=120, anchor='center')
        vsb = ttk.Scrollbar(tf, orient='vertical',
                             command=self._ds_tree.yview)
        hsb = ttk.Scrollbar(tf, orient='horizontal',
                             command=self._ds_tree.xview)
        self._ds_tree.configure(yscrollcommand=vsb.set,
                                 xscrollcommand=hsb.set)
        self._ds_tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        tf.grid_rowconfigure(0, weight=1)
        tf.grid_columnconfigure(0, weight=1)
        self._load_ds_view()

    def _load_ds_view(self):
        import pandas as pd
        for r in self._ds_tree.get_children(): self._ds_tree.delete(r)
        src  = self._ds_src.get()
        paths = {
            'combined':   BASE_DIR/'data'/'combined_all.csv',
            'autoimmune': BASE_DIR/'data'/'combined_with_normal_2000.csv',
            'immunodef':  BASE_DIR/'data'/'immunodeficiency_dataset.csv',
        }
        path = paths.get(src)
        if not path or not path.exists():
            self._ds_count.set('Not found'); return
        df = pd.read_csv(path, nrows=500, low_memory=False)
        fl = self._ds_filter.get().strip()
        if fl:
            df = df[df['Diagnosis'].str.contains(fl, case=False, na=False)]
        self._ds_count.set(f'{len(df)} records')
        cols = ('Patient_ID','Age','Gender','Diagnosis',
                'WBC_Count','Lymphocytes','ESR','CRP','ANA')
        for _, row in df.head(200).iterrows():
            self._ds_tree.insert('', 'end',
                values=tuple(str(row.get(c,'—')) for c in cols))


    # ══════════════════════════════════════════════════════════
    # MISCLASSIFICATIONS PAGE
    # ══════════════════════════════════════════════════════════
    def _build_errors_page(self, parent):
        hdr = tk.Frame(parent, bg=C['bg'], pady=14)
        hdr.pack(fill='x', padx=28)
        lbl(hdr, 'Misclassifications', FH, bg=C['bg']).pack(anchor='w')
        lbl(hdr,
            'Every patient the model got wrong on the 30% test set — what it predicted vs reality',
            FS, fg=C['sub'], bg=C['bg']).pack(anchor='w')

        # ── Model selector + summary tiles ───────────────────
        ctrl = tk.Frame(parent, bg=C['card'])
        ctrl.pack(fill='x', padx=28, pady=6)
        ci = tk.Frame(ctrl, bg=C['card'])
        ci.pack(fill='x', padx=14, pady=8)

        self._err_model_var = tk.StringVar(value='risk')
        lbl(ci, 'Model:', FS, fg=C['sub'], bg=C['card']).pack(side='left')
        for val, text in [
            ('risk',     'Model 1: Risk Detection'),
            ('category', 'Model 2: Disease Category'),
            ('level',    'Model 3: Risk Stratification'),
        ]:
            tk.Radiobutton(ci, text=text, variable=self._err_model_var,
                           value=val, command=self._load_errors,
                           bg=C['card'], fg=C['text'],
                           activebackground=C['card'],
                           selectcolor=C['input'],
                           font=FL).pack(side='left', padx=10)

        rbtn(ci, '  🔄  Refresh  ', self._load_errors,
             color=C['card2'], width=100, height=26).pack(side='right', padx=4)

        # Summary strip
        self._err_summary = tk.Frame(parent, bg=C['bg'])
        self._err_summary.pack(fill='x', padx=28, pady=(0,4))

        # Filter bar
        fbar = tk.Frame(parent, bg=C['card'])
        fbar.pack(fill='x', padx=28, pady=(0,6))
        fi = tk.Frame(fbar, bg=C['card'])
        fi.pack(fill='x', padx=14, pady=6)
        lbl(fi, 'Filter diagnosis / error type:', FS,
            fg=C['sub'], bg=C['card']).pack(side='left', padx=(0,6))
        self._err_filter = tk.StringVar()
        ent(fi, var=self._err_filter, width=28).pack(side='left')
        rbtn(fi, 'Apply', self._load_errors,
             color=C['card2'], width=70, height=26).pack(side='left', padx=5)
        rbtn(fi, 'Clear', lambda: (self._err_filter.set(''), self._load_errors()),
             color=C['card2'], width=60, height=26).pack(side='left', padx=3)
        self._err_count_var = tk.StringVar()
        tk.Label(fi, textvariable=self._err_count_var, font=FS,
                 bg=C['card'], fg=C['sub']).pack(side='right', padx=8)

        # Table
        tf = tk.Frame(parent, bg=C['bg'])
        tf.pack(fill='both', expand=True, padx=28, pady=(0,6))

        style = ttk.Style()
        style.configure('E.Treeview',
                         background=C['surface'], foreground=C['text'],
                         rowheight=26, fieldbackground=C['surface'], font=FS)
        style.configure('E.Treeview.Heading',
                         background=C['card2'], foreground=C['accent'],
                         font=('Segoe UI',9,'bold'), relief='flat')
        style.map('E.Treeview',
                   background=[('selected', C['accent'])],
                   foreground=[('selected','#ffffff')])

        self._err_tree_cols = []
        self._err_tree = ttk.Treeview(tf, show='headings', style='E.Treeview')
        vsb = ttk.Scrollbar(tf, orient='vertical', command=self._err_tree.yview)
        hsb = ttk.Scrollbar(tf, orient='horizontal', command=self._err_tree.xview)
        self._err_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self._err_tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        tf.grid_rowconfigure(0, weight=1)
        tf.grid_columnconfigure(0, weight=1)

        # Detail panel at bottom
        detail_c = tk.Frame(parent, bg=C['card'])
        detail_c.pack(fill='x', padx=28, pady=(0,8))
        lbl(detail_c, 'Selected Patient Detail', FT, fg=C['accent'],
            bg=C['card']).pack(anchor='w', padx=14, pady=(8,4))
        sep(detail_c).pack(fill='x', padx=14)
        self._err_detail = tk.Text(
            detail_c, height=4, bg=C['input'], fg=C['text'],
            font=FM, relief='flat', state='disabled',
            wrap='word')
        self._err_detail.pack(fill='x', padx=14, pady=6)
        self._err_tree.bind('<<TreeviewSelect>>', self._on_err_select)

        # Export button
        rbtn(parent, '  💾  Export to CSV  ', self._export_errors,
             color=C['green'], width=180, height=32).pack(pady=4)

        self._err_current_df = None
        self._load_errors()

    def _load_errors(self):
        import pandas as pd

        # Clear summary
        for w in self._err_summary.winfo_children(): w.destroy()

        model = self._err_model_var.get()
        files = {
            'risk':     OUTPUT_DIR / 'misclassified_risk.csv',
            'category': OUTPUT_DIR / 'misclassified_category.csv',
            'level':    OUTPUT_DIR / 'misclassified_risk_level.csv',
        }
        path = BASE_DIR / 'outputs' / files[model].name

        if not path.exists():
            self._err_count_var.set('No data — run Training first')
            # Clear table and show message
            for row in self._err_tree.get_children():
                self._err_tree.delete(row)
            self._err_tree['columns'] = ['message']
            self._err_tree.heading('message', text='Status')
            self._err_tree.column('message', width=500)
            self._err_tree.insert('', 'end',
                values=('No misclassification data found. Go to Train Models and run training first.',))
            self._err_current_df = None
            return

        df = pd.read_csv(path, low_memory=False)
        flt = self._err_filter.get().strip()
        if flt:
            mask = pd.Series([False]*len(df))
            for col in df.columns:
                if df[col].dtype == object:
                    mask = mask | df[col].str.contains(flt, case=False, na=False)
            df = df[mask]

        self._err_current_df = df
        self._err_count_var.set(f'{len(df)} misclassified patients')

        # ── Summary tiles ─────────────────────────────────────
        LEVEL_C = {'No Risk':C['green'],'Low Risk':C['blue'],
                   'Moderate Risk':C['yellow'],'Severe Risk':C['red']}
        total_test = 5574  # from training log

        g = tk.Frame(self._err_summary, bg=C['bg'])
        g.pack(fill='x', pady=4)

        def stile(par, title, val, col, idx):
            f = tk.Frame(par, bg=C['card'], padx=14, pady=8)
            f.grid(row=0, column=idx, padx=5, sticky='ew')
            par.columnconfigure(idx, weight=1)
            lbl(f, title, FS, fg=C['sub'], bg=C['card']).pack()
            lbl(f, str(val), ('Segoe UI',14,'bold'),
                fg=col, bg=C['card']).pack(pady=2)

        stile(g, 'Total Wrong',
              f"{len(df):,}", C['red'], 0)
        stile(g, 'Error Rate',
              f"{len(df)/total_test*100:.2f}%", C['yellow'], 1)

        if model == 'risk':
            fn = len(df[df.get('error_type','').str.contains('Negative', na=False)])
            fp = len(df[df.get('error_type','').str.contains('Positive', na=False)])
            stile(g, 'Missed Disease (FN)', fn, C['red'],    2)
            stile(g, 'False Alarm (FP)',    fp, C['yellow'], 3)
            stile(g, 'Avg Confidence',
                  f"{df['confidence'].mean()*100:.1f}%" if 'confidence' in df.columns else '—',
                  C['sub'], 4)

        elif model == 'category':
            if 'error_type' in df.columns:
                top = df['error_type'].value_counts().head(1)
                stile(g, 'Most Common Error',
                      top.index[0][:30] if len(top) else '—', C['sub'], 2)
            stile(g, 'Avg Confidence',
                  f"{df['confidence'].mean()*100:.1f}%" if 'confidence' in df.columns else '—',
                  C['sub'], 3)

        elif model == 'level':
            under = len(df[df.get('error_direction','').str.contains('Under', na=False)])
            over  = len(df[df.get('error_direction','').str.contains('Over',  na=False)])
            stile(g, 'Under-estimated Risk', under, C['red'],    2)
            stile(g, 'Over-estimated Risk',  over,  C['yellow'], 3)
            stile(g, 'Avg Levels Off',
                  f"{df['levels_off'].mean():.2f}" if 'levels_off' in df.columns else '—',
                  C['sub'], 4)

        # ── Table ─────────────────────────────────────────────
        # Choose columns based on model
        col_sets = {
            'risk':     ['Patient_ID','Age','Gender','Diagnosis',
                         'actual_text','predicted_text',
                         'confidence','error_type',
                         'Symptom_Burden','Antibody_Score','ESR','CRP'],
            'category': ['Patient_ID','Age','Gender','Diagnosis',
                         'actual_cat','predicted_cat',
                         'confidence','error_type',
                         'Symptom_Burden','Antibody_Score'],
            'level':    ['Patient_ID','Age','Gender','Diagnosis',
                         'actual_level','predicted_level',
                         'levels_off','error_direction','error_type',
                         'Symptom_Burden','Antibody_Score','severity_score'],
        }
        cols = [c for c in col_sets[model] if c in df.columns]

        # Rebuild treeview columns
        self._err_tree['columns'] = cols
        self._err_tree_cols = cols
        for col in cols:
            w = 160 if col in ('error_type','Diagnosis','error_direction') else 90
            self._err_tree.heading(col, text=col.replace('_',' ').title())
            self._err_tree.column(col, width=w, anchor='center', minwidth=60)

        for row in self._err_tree.get_children():
            self._err_tree.delete(row)

        # Colour-code rows by error type
        self._err_tree.tag_configure('fn',  background='#fee2e2')  # light red
        self._err_tree.tag_configure('fp',  background='#fef9c3')  # light yellow
        self._err_tree.tag_configure('under', background='#fee2e2')
        self._err_tree.tag_configure('over',  background='#fef9c3')
        self._err_tree.tag_configure('cat',   background='#ede9fe') # light purple

        for _, row in df.iterrows():
            vals = tuple(str(row.get(c,'—')) for c in cols)
            et   = str(row.get('error_type',''))
            ed   = str(row.get('error_direction',''))
            if 'Negative' in et:    tag = ('fn',)
            elif 'Positive' in et:  tag = ('fp',)
            elif 'Under' in ed:     tag = ('under',)
            elif 'Over' in ed:      tag = ('over',)
            else:                   tag = ('cat',)
            self._err_tree.insert('', 'end', values=vals, tags=tag)

    def _on_err_select(self, event):
        sel = self._err_tree.selection()
        if not sel or self._err_current_df is None: return
        idx   = self._err_tree.index(sel[0])
        row   = self._err_current_df.iloc[idx]
        model = self._err_model_var.get()

        lines = []
        lines.append(f"Patient: {row.get('Patient_ID','?')}  |  Age: {row.get('Age','?')}  |  Gender: {row.get('Gender','?')}  |  Diagnosis: {row.get('Diagnosis','?')}")

        if model == 'risk':
            lines.append(f"Expected: {row.get('actual_text','?')}  →  Model predicted: {row.get('predicted_text','?')}  (confidence {float(row.get('confidence',0))*100:.1f}%)")
            lines.append(f"Error type: {row.get('error_type','?')}")
        elif model == 'category':
            lines.append(f"Expected category: {row.get('actual_cat','?')}  →  Model predicted: {row.get('predicted_cat','?')}  (confidence {float(row.get('confidence',0))*100:.1f}%)")
        elif model == 'level':
            lines.append(f"Expected level: {row.get('actual_level','?')}  →  Model predicted: {row.get('predicted_level','?')}  ({row.get('error_direction','?')}, {row.get('levels_off','?')} level(s) off)")

        lines.append(f"Symptom burden: {row.get('Symptom_Burden','?')}  |  Antibody score: {row.get('Antibody_Score','?')}  |  ESR: {row.get('ESR','?')}  |  CRP: {row.get('CRP','?')}  |  Severity score: {row.get('severity_score','?')}")

        self._err_detail.configure(state='normal')
        self._err_detail.delete('1.0', tk.END)
        self._err_detail.insert(tk.END, '\n'.join(lines))
        self._err_detail.configure(state='disabled')

    def _export_errors(self):
        if self._err_current_df is None or len(self._err_current_df) == 0:
            messagebox.showinfo('Nothing to export', 'No misclassification data loaded.')
            return
        path = filedialog.asksaveasfilename(
            defaultextension='.csv',
            filetypes=[('CSV','*.csv'),('All','*.*')],
            initialfile=f'misclassified_{self._err_model_var.get()}.csv')
        if path:
            self._err_current_df.to_csv(path, index=False)
            messagebox.showinfo('Exported', f'Saved to:\n{path}')


    # ══════════════════════════════════════════════════════════
    # INFO PAGE
    # ══════════════════════════════════════════════════════════
    def _build_info_page(self, parent):
        hdr = tk.Frame(parent, bg=C['bg'], pady=14)
        hdr.pack(fill='x', padx=28)
        lbl(hdr, 'Model Info', FH, bg=C['bg']).pack(anchor='w')
        self._info_text = scrolledtext.ScrolledText(
            parent, bg=C['input'], fg=C['text'],
            font=FM, relief='flat', state='disabled')
        self._info_text.pack(fill='both', expand=True, padx=28, pady=8)
        rbtn(parent, '  🔄  Refresh  ', self._load_info,
             color=C['card2'], width=120, height=32).pack(pady=6)
        self._load_info()

    def _load_info(self):
        mp = BASE_DIR/'models'/'model_metadata.json'
        self._info_text.configure(state='normal')
        self._info_text.delete('1.0', tk.END)
        if mp.exists():
            with open(mp) as f:
                self._info_text.insert(tk.END, json.dumps(json.load(f), indent=2))
        else:
            self._info_text.insert(tk.END,
                'No models found.\nGo to Train Models and run training first.')
        self._info_text.configure(state='disabled')

    # ── Utilities ─────────────────────────────────────────────
    def _check_models(self):
        mp = BASE_DIR/'models'/'model_metadata.json'
        self.status_var.set(
            '● Models ready' if mp.exists() else '⚠️  Models not trained')


if __name__ == '__main__':
    ImmuneRiskApp().mainloop()
