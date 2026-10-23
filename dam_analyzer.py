"""
DAM Fly Analyzer
Análisis de monitores DAM (Drosophila Activity Monitor) combinando:
  - estilo SleepMat: sueño (5 min de inactividad), episodios, latencia, día/noche, estadística
  - estilo Rtivity : perfiles de 24 h, series temporales, actogramas, periodograma chi-cuadrado

Uso:  streamlit run dam_analyzer.py
"""
import io
import re
import itertools
import datetime as dt

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

st.set_page_config(page_title="DAM Fly Analyzer", page_icon="🪰", layout="wide")

COLORS = px.colors.qualitative.Plotly
METRICS = {
    "Sueño total (min)": "sueno_min",
    "Nº de episodios de sueño": "n_bouts",
    "Duración media del episodio (min)": "long_bout",
    "Latencia al sueño (min, solo noche)": "latencia_min",
    "Actividad total (cuentas)": "actividad",
    "Actividad en vigilia (cuentas/min)": "act_vigilia",
}


# ───────────────────────── lectura de datos ─────────────────────────
@st.cache_data(show_spinner=False)
def read_monitor(raw: bytes, mon: int) -> pd.DataFrame:
    """Lee un archivo MonitorN.txt estándar de DAM (42 columnas, separado por tabs)."""
    df = pd.read_csv(io.BytesIO(raw), sep="\t", header=None, index_col=False)
    if df.shape[1] < 42:
        raise ValueError(f"Se esperaban 42 columnas y hay {df.shape[1]}.")
    stamp = df[1].astype(str) + " " + df[2].astype(str)
    ts = pd.to_datetime(stamp, format="%d %b %y %H:%M:%S", errors="coerce")
    if ts.isna().mean() > 0.5:
        ts = pd.to_datetime(stamp, dayfirst=True, errors="coerce")
    ch = df.iloc[:, 10:42].apply(pd.to_numeric, errors="coerce").fillna(0)
    ch.columns = [f"M{mon}_C{i:02d}" for i in range(1, 33)]
    ch.index = pd.DatetimeIndex(ts)
    ch = ch[~ch.index.isna()]
    return ch[~ch.index.duplicated()].sort_index()


@st.cache_data(show_spinner=False)
def demo_data() -> pd.DataFrame:
    """Datos sintéticos: 32 moscas, 7 días, bins de 1 min. Mitad control, mitad mutante."""
    rng = np.random.default_rng(1)
    idx = pd.date_range("2024-01-01 08:00", periods=7 * 1440, freq="1min")
    h = np.arange(len(idx)) / 60
    out = {}
    for i in range(32):
        mutant = i >= 16
        period = 24.8 if mutant else 24.0
        amp = rng.uniform(2, 4) * (1.4 if mutant else 1.0)
        phase = (h % period) / period
        day = phase < 0.5
        shape = 0.08 + 0.9 * day * (0.35 + 0.65 * np.cos(4 * np.pi * phase) ** 2)
        x = rng.poisson(amp * shape).astype(float)
        if i in (5, 21):  # moscas que mueren
            x[4 * 1440:] = 0
        out[f"M1_C{i + 1:02d}"] = x
    return pd.DataFrame(out, index=idx)


# ───────────────────────── procesamiento ─────────────────────────
def bin_minutes(idx) -> int:
    return max(1, int(round(pd.Series(idx).diff().median().total_seconds() / 60)))


def get_window(idx, lights_on, start_day, n_days):
    """Ventana que comienza en un ZT0 (encendido de luz)."""
    first = idx[0].normalize() + pd.Timedelta(hours=lights_on)
    if first < idx[0]:
        first += pd.Timedelta(days=1)
    start = first + pd.Timedelta(days=int(start_day))
    return start, start + pd.Timedelta(days=int(n_days))


def sleep_state(act: pd.DataFrame, bin_min: int, thr_min: int) -> pd.DataFrame:
    """Sueño = ≥ thr_min minutos consecutivos sin actividad (Hendricks / Pfeiffenberger)."""
    w = max(1, int(round(thr_min / bin_min)))
    inactive = (act == 0).astype(float)
    full = inactive.rolling(w, min_periods=w).min().fillna(0)
    return full[::-1].rolling(w, min_periods=1).max()[::-1]


def run_lengths(x: np.ndarray) -> np.ndarray:
    d = np.diff(np.r_[0, (x > 0).astype(int), 0])
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def compute_metrics(act, slp, start, lh, bin_min, geno) -> pd.DataFrame:
    hrs = np.asarray((act.index - start).total_seconds()) / 3600
    day_no = (hrs // 24).astype(int)
    is_day = (hrs % 24) < lh
    phases = {"Día": is_day, "Noche": ~is_day, "24 h": np.ones_like(is_day)}
    rows = []
    for fly in act.columns:
        a, s = act[fly].to_numpy(), slp[fly].to_numpy()
        for d in np.unique(day_no):
            for ph, pm in phases.items():
                m = pm & (day_no == d)
                if not m.any():
                    continue
                am, sm = a[m], s[m]
                ln = run_lengths(sm) * bin_min
                awake = sm == 0
                lat = np.nan
                if ph == "Noche":
                    hit = np.flatnonzero(sm > 0)
                    lat = hit[0] * bin_min if hit.size else np.nan
                rows.append(dict(
                    Fly=fly, Genotipo=geno[fly], Dia=d + 1, Fase=ph,
                    sueno_min=sm.sum() * bin_min,
                    n_bouts=len(ln),
                    long_bout=ln.mean() if len(ln) else np.nan,
                    latencia_min=lat,
                    actividad=am.sum(),
                    act_vigilia=am[awake].sum() / (awake.sum() * bin_min) if awake.any() else np.nan,
                ))
    return pd.DataFrame(rows)


def to_profile(df_bins: pd.DataFrame, start) -> pd.DataFrame:
    """Promedia bins de 30 min por hora ZT (0-24) para cada mosca."""
    hrs = np.asarray((df_bins.index - start).total_seconds()) / 3600
    zt = np.floor((hrs % 24) * 2) / 2
    return df_bins.groupby(zt).mean()


def hex_rgba(h, a):
    h = h.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def line_band(prof, geno, ylab, shades, xlab="ZT (h)"):
    fig = go.Figure()
    for i, g in enumerate(sorted(set(geno[f] for f in prof.columns))):
        flies = [f for f in prof.columns if geno[f] == g]
        m = prof[flies].mean(axis=1)
        e = prof[flies].sem(axis=1).fillna(0)
        c = COLORS[i % len(COLORS)]
        fig.add_trace(go.Scatter(x=m.index, y=m + e, mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=m.index, y=m - e, mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor=hex_rgba(c, 0.2),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=m.index, y=m, name=f"{g} (n={len(flies)})",
                                 line=dict(color=c, width=2)))
    for x0, x1 in shades:
        fig.add_vrect(x0=x0, x1=x1, fillcolor="grey", opacity=0.15, line_width=0, layer="below")
    fig.update_layout(xaxis_title=xlab, yaxis_title=ylab, height=420,
                      legend=dict(orientation="h", y=1.1), margin=dict(t=30))
    return fig


def actogram_fig(series: pd.Series, start, n_days, title):
    s = series.resample("30min", origin=start).sum()
    full = pd.date_range(start, periods=n_days * 48, freq="30min")
    v = s.reindex(full, fill_value=0).to_numpy().reshape(n_days, 48)
    nxt = np.vstack([v[1:], np.zeros((1, 48))])
    z = np.hstack([v, nxt])  # doble graficado (48 h)
    fig = go.Figure(go.Heatmap(z=z, x=np.arange(96) * 0.5, y=list(range(1, n_days + 1)),
                               colorscale="Greys", colorbar=dict(title="cuentas/30 min")))
    fig.update_yaxes(autorange="reversed", title="Día", dtick=1)
    fig.update_xaxes(title="ZT (h) – doble graficado", dtick=6)
    fig.update_layout(title=title, height=130 + 45 * n_days)
    return fig


def chisq_periodogram(x: np.ndarray, pbin_min: int, lo=16, hi=32) -> pd.DataFrame:
    """Periodograma chi-cuadrado de Sokolove-Bushell."""
    res = []
    for P in range(int(lo * 60 / pbin_min), int(hi * 60 / pbin_min) + 1):
        K = len(x) // P
        if K < 2:
            break
        m = x[: K * P].reshape(K, P)
        M = m.mean()
        den = ((m - M) ** 2).sum()
        q = (K * P) * K * ((m.mean(0) - M) ** 2).sum() / den if den > 0 else 0.0  # N·K·S/T
        res.append((P * pbin_min / 60, q, stats.chi2.ppf(0.95, P - 1)))
    return pd.DataFrame(res, columns=["periodo_h", "Qp", "umbral"])


def run_stats(df, col):
    groups = {g: v[col].dropna().to_numpy() for g, v in df.groupby("Genotipo")}
    groups = {g: v for g, v in groups.items() if len(v) > 0}
    if len(groups) < 2:
        return None, None
    glob = None
    if len(groups) > 2:
        try:
            H, p = stats.kruskal(*groups.values())
            glob = f"Kruskal-Wallis: H = {H:.2f}, p = {p:.4g}"
        except ValueError:
            glob = "Kruskal-Wallis no calculable (valores idénticos)."
    pairs = list(itertools.combinations(groups, 2))
    rows = []
    for a, b in pairs:
        try:
            p = stats.mannwhitneyu(groups[a], groups[b]).pvalue
        except ValueError:
            p = np.nan
        rows.append(dict(A=a, B=b, p=p, p_bonferroni=min(1.0, p * len(pairs))))
    return glob, pd.DataFrame(rows)


# ───────────────────────── interfaz ─────────────────────────
def parse_rules(text):
    """Reglas tipo '81:1-16=Control' o '17-32,40=Mutante' (monitor opcional). Las últimas ganan."""
    rules = []
    for part in re.split(r"[;\n]+", text):
        part = part.strip()
        if not part or "=" not in part:
            continue
        lhs, grp = part.rsplit("=", 1)
        mon = None
        if ":" in lhs:
            m_, lhs = lhs.split(":", 1)
            mon = int(m_.strip())
        chans = set()
        for tok in lhs.split(","):
            tok = tok.strip()
            if "-" in tok:
                a, b = tok.split("-", 1)
                chans.update(range(int(a), int(b) + 1))
            elif tok:
                chans.add(int(tok))
        rules.append((mon, chans, grp.strip()))
    return rules


def stats_table(perfly):
    rows = []
    for ph in ["Día", "Noche", "24 h"]:
        sub = perfly[perfly["Fase"] == ph]
        for lab, col in METRICS.items():
            if col == "latencia_min" and ph != "Noche":
                continue
            _, pw = run_stats(sub, col)
            if pw is None:
                continue
            for _, r in pw.iterrows():
                rows.append({"Fase": ph, "Métrica": lab, "Grupo A": r["A"], "Grupo B": r["B"],
                             "p": r["p"], "p (Bonferroni)": r["p_bonferroni"]})
    return pd.DataFrame(rows)


def daily_evolution(dm, col, label):
    d = dm[dm["Fase"].isin(["Día", "Noche"])]
    g = d.groupby(["Genotipo", "Fase", "Dia"])[col].agg(media="mean", sem="sem").reset_index()
    g["sem"] = g["sem"].fillna(0)
    fig = px.line(g, x="Dia", y="media", color="Genotipo", facet_col="Fase", error_y="sem",
                  markers=True, labels={"media": label, "Dia": "Día del experimento"})
    fig.update_xaxes(dtick=1)
    fig.update_layout(height=380)
    return fig


st.title("🪰 DAM Fly Analyzer")

with st.sidebar:
    st.header("Ajustes (opcionales)")
    lights_on = st.number_input("Luces encendidas a las (hora reloj, = ZT0)", 0.0, 23.5, 8.0, 0.5)
    lh = st.number_input("Horas de luz por día", 1.0, 23.0, 12.0, 0.5)
    sleep_thr = st.number_input("Minutos de inactividad = sueño", 1, 30, 5)
    dead_h = st.number_input("Mosca muerta si no se mueve en las últimas (h)", 6, 72, 24)
    drop_dead = st.checkbox("Excluir moscas muertas", True)
    pdays = st.number_input("Periodograma: usar solo los últimos N días (0 = toda la semana)", 0, 30, 0)
    st.divider()
    demo = st.checkbox("Usar datos de ejemplo")

files = st.file_uploader("1 · Carga tu(s) archivo(s) Monitor (.txt)", accept_multiple_files=True)

if demo:
    counts = demo_data()
elif files:
    frames, used = [], set()
    for i, f in enumerate(files):
        nums = re.findall(r"\d+", f.name)
        mon = int(nums[0]) if nums and int(nums[0]) not in used else i + 1
        used.add(mon)
        try:
            frames.append(read_monitor(f.getvalue(), mon))
        except Exception as e:
            st.error(f"No pude leer {f.name}: {e}")
            st.stop()
    counts = pd.concat(frames, axis=1, join="inner").sort_index()
else:
    st.info("Carga un archivo Monitor*.txt para empezar (o activa «Usar datos de ejemplo» en la barra lateral).")
    st.stop()

idx = counts.index
bin_min = bin_minutes(idx)
data_first, data_last = idx[0].date(), idx[-1].date()

# ── fechas de la semana
first = idx[0].normalize() + pd.Timedelta(hours=lights_on)
if first < idx[0]:
    first += pd.Timedelta(days=1)
d0 = min(first.date(), data_last)
avail0 = int(np.floor(((idx[-1] + pd.Timedelta(minutes=bin_min)) - first) / pd.Timedelta(days=1) + 1e-9))
use_all = st.checkbox("Usar todo el registro (todas las semanas)")
if use_all:
    nweeks = max(1, int(np.ceil(avail0 / 7)))

    def wk_range(w):
        a = first + pd.Timedelta(days=7 * (w - 1))
        b = first + pd.Timedelta(days=min(7 * w, avail0) - 1)
        return a.date(), b.date()

    opts = ["Todo el registro"]
    for w in range(1, nweeks + 1):
        a, b = wk_range(w)
        nd_ = (b - a).days + 1
        opts.append(f"Semana {w} ({a:%d/%m} – {b:%d/%m})" + ("" if nd_ == 7 else f" · {nd_} días"))
    pick = st.selectbox("Periodo a analizar (elige una semana para comparar grupos en esa misma semana)", opts)
    dates = (first.date(), wk_range(nweeks)[1]) if pick == opts[0] else wk_range(opts.index(pick))
else:
    dates = st.date_input(
        "2 · Elige tu semana (desde – hasta, ambos días incluidos)",
        value=(d0, min(d0 + dt.timedelta(days=6), data_last)),
        min_value=data_first, max_value=data_last,
        help="Cada día empieza en ZT0 (hora de encendido de luz de los ajustes).")
if len(dates) != 2:
    st.info("Selecciona la fecha final del rango.")
    st.stop()

start = pd.Timestamp(dates[0]) + pd.Timedelta(hours=lights_on)
if start < idx[0]:
    start += pd.Timedelta(days=1)
    st.warning(f"Los datos empiezan después de ZT0 del {dates[0]}; el análisis arranca en {start:%d-%m-%Y %H:%M}.")
requested = (dates[1] - dates[0]).days + 1
avail = int(np.floor(((idx[-1] + pd.Timedelta(minutes=bin_min)) - start) / pd.Timedelta(days=1) + 1e-9))
n_days = min(requested, avail)
if n_days < 1:
    st.error("El rango elegido no contiene ningún día completo de datos.")
    st.stop()
if n_days < requested:
    st.warning(f"Solo hay {n_days} día(s) completo(s) en ese rango (pediste {requested}); se analiza lo disponible.")
end = start + pd.Timedelta(days=n_days)

# ── genotipos (opcional)
with st.expander("Grupos / genotipos (opcional – por defecto todas las moscas son un solo grupo)"):
    st.caption("Asignación rápida por canales. Una regla por línea: `monitor:canales=Grupo`, "
               "por ejemplo `81:1-16=Control` y `81:17-32=Mutante`. El monitor es opcional "
               "(sin él, aplica a todos los monitores) y puedes mezclar rangos: `1-8,12=Grupo A`.")
    rules_txt = st.text_area("Reglas", value=("1:1-16=Control\n1:17-32=Mutante" if demo else ""),
                             key="rules_" + str(demo), height=100)
    try:
        rules = parse_rules(rules_txt)
    except Exception:
        st.error("No entendí las reglas. Usa el formato `81:1-16=Control`.")
        rules = []
    gdef = {}
    for f in counts.columns:
        mon_n = int(f[1:f.index("_")])
        ch_n = int(f.rsplit("C", 1)[1])
        gdef[f] = "G1"
        for mon_r, chans, g_ in rules:
            if (mon_r is None or mon_r == mon_n) and ch_n in chans:
                gdef[f] = g_
    meta = pd.DataFrame({"Fly": counts.columns, "Genotipo": [gdef[f] for f in counts.columns], "Incluir": True})
    st.caption("Puedes ajustar a mano en la tabla (o pegar desde Excel). Desmarca «Incluir» en canales sin mosca.")
    meta = st.data_editor(meta, key="geno_" + str(hash(tuple(counts.columns))) + str(demo) + str(hash(rules_txt)),
                          hide_index=True, disabled=["Fly"], height=250)
meta = meta[meta["Incluir"]]
if meta.empty:
    st.warning("No hay moscas seleccionadas.")
    st.stop()
counts = counts[list(meta["Fly"])]
geno = dict(zip(meta["Fly"], meta["Genotipo"].astype(str)))

# ── cálculo
slp_all = sleep_state(counts, bin_min, sleep_thr)
mask = (idx >= start) & (idx < end)
act, slp = counts[mask], slp_all[mask]

tail = min(len(act), max(1, int(dead_h * 60 / bin_min)))
alive = act.iloc[-tail:].sum() > 0
dead = list(alive[~alive].index)
if drop_dead and dead:
    keep = list(alive[alive].index)
    act, slp = act[keep], slp[keep]
if act.shape[1] == 0:
    st.warning("Todas las moscas fueron excluidas.")
    st.stop()

dm = compute_metrics(act, slp, start, lh, bin_min, geno)
perfly = dm.groupby(["Fly", "Genotipo", "Fase"], as_index=False)[list(METRICS.values())].mean()
shades_zt = [(lh, 24)]
shades_ts = [((start + pd.Timedelta(hours=d * 24 + lh)).to_pydatetime(),
              (start + pd.Timedelta(hours=(d + 1) * 24)).to_pydatetime()) for d in range(int(n_days))]
a30 = act.resample("30min", origin=start).sum()
s30 = slp.resample("30min", origin=start).sum() * bin_min
genos = sorted(set(geno[f] for f in act.columns))

# ───────── resultados (todo en una página)
st.divider()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Moscas analizadas", act.shape[1])
m2.metric("Días analizados", int(n_days))
m3.metric("Periodo", f"{start:%d/%m} → {(end - pd.Timedelta(minutes=1)):%d/%m}")
m4.metric("Excluidas (muertas)", len(dead) if drop_dead else 0,
          help=", ".join(dead) if dead else "ninguna")
cnt = pd.Series(geno).loc[list(act.columns)].value_counts()
if len(cnt) > 1:
    st.caption("Moscas por grupo: " + " · ".join(f"{g}: {n}" for g, n in cnt.items()))

st.header("Perfil promedio de 24 h")
st.caption("Promedio de los días analizados · banda = error estándar entre moscas · gris = oscuridad")
p1, p2 = st.columns(2)
p1.plotly_chart(line_band(to_profile(s30, start), geno, "Sueño (min / 30 min)", shades_zt))
p2.plotly_chart(line_band(to_profile(a30, start), geno, "Actividad (cuentas / 30 min)", shades_zt))

st.header("Serie temporal de la semana")
st.plotly_chart(line_band(s30, geno, "Sueño (min / 30 min)", shades_ts, xlab="Tiempo"))
st.plotly_chart(line_band(a30, geno, "Actividad (cuentas / 30 min)", shades_ts, xlab="Tiempo"))

st.header("Evolución día a día")
e1, e2 = st.columns(2)
e1.markdown("**Sueño (min)**")
e1.plotly_chart(daily_evolution(dm, "sueno_min", "Sueño (min)"))
e2.markdown("**Actividad (cuentas)**")
e2.plotly_chart(daily_evolution(dm, "actividad", "Actividad (cuentas)"))

if n_days > 7:
    st.header("Comparación semana a semana")
    st.caption("Semanas de 7 días contadas desde tu fecha inicial · punto = media de las moscas · barras = error estándar")
    dw = dm.copy()
    dw["wk"] = (dw["Dia"] - 1) // 7 + 1
    nd = dw.groupby("wk")["Dia"].nunique()
    wlabel = {w: f"Sem {w}" + ("" if nd[w] == 7 else f" ({nd[w]} d)") for w in nd.index}
    dw["Semana"] = dw["wk"].map(wlabel)
    worder = [wlabel[w] for w in sorted(wlabel)]
    pfw_all = dw.groupby(["Fly", "Genotipo", "Fase", "Semana"], as_index=False)[list(METRICS.values())].mean()
    pfw = pfw_all[pfw_all["Fase"].isin(["Día", "Noche"])]
    wcols = st.columns(2)
    for i, (label, col) in enumerate(METRICS.items()):
        g = pfw.groupby(["Genotipo", "Fase", "Semana"])[col].agg(media="mean", sem="sem").reset_index()
        g["sem"] = g["sem"].fillna(0)
        figw = px.line(g, x="Semana", y="media", color="Genotipo", facet_col="Fase", error_y="sem",
                       markers=True, labels={"media": label},
                       category_orders={"Semana": worder, "Fase": ["Día", "Noche"]})
        figw.update_layout(title=label, height=360)
        wcols[i % 2].plotly_chart(figw)
    wt = []
    for w in worder:
        t = stats_table(pfw_all[pfw_all["Semana"] == w])
        if not t.empty:
            t.insert(0, "Semana", w)
            wt.append(t)
    if wt:
        with st.expander("Estadística entre grupos, semana por semana (Mann-Whitney, Bonferroni)"):
            st.dataframe(pd.concat(wt).round(5))

st.header("Comparación por genotipo (promedio de la semana, cada punto = una mosca)")
cols = st.columns(2)
for i, (label, col) in enumerate(METRICS.items()):
    d = perfly if col != "latencia_min" else perfly[perfly["Fase"] == "Noche"]
    fig = px.box(d, x="Genotipo", y=col, color="Fase", points="all", hover_name="Fly",
                 category_orders={"Fase": ["Día", "Noche", "24 h"]}, labels={col: label})
    fig.update_traces(pointpos=0, jitter=0.5, marker=dict(size=5, opacity=0.8))
    fig.update_layout(title=label, height=380)
    cols[i % 2].plotly_chart(fig)

st.header("Estadística")
tbl = stats_table(perfly)
if tbl.empty:
    st.caption("Se necesitan al menos 2 genotipos para comparar (define genotipos en el panel de arriba).")
else:
    st.caption("Mann-Whitney U por pares, con corrección de Bonferroni dentro de cada métrica y fase.")
    st.dataframe(tbl.round(5))
summ = (perfly.groupby(["Fase", "Genotipo"])[list(METRICS.values())]
        .agg(["mean", "sem"]).round(2))
st.markdown("**Media y SEM por genotipo y fase**")
st.dataframe(summ)

st.header("Actogramas")
st.caption("Doble graficados (48 h), bins de 30 min · el eje X empieza en ZT0")
acols = st.columns(2)
for i, g in enumerate(genos):
    ser = act[[f for f in act.columns if geno[f] == g]].mean(axis=1)
    acols[i % 2].plotly_chart(actogram_fig(ser, start, int(n_days), f"{g} (promedio)"))
with st.expander("Actograma de una mosca individual"):
    fly = st.selectbox("Mosca", list(act.columns))
    st.plotly_chart(actogram_fig(act[fly], start, int(n_days), fly))

st.header("Periodograma chi-cuadrado")
pstart = max(start, end - pd.Timedelta(days=int(pdays))) if pdays else start
pdays_eff = int((end - pstart) / pd.Timedelta(days=1))
if pdays_eff < 2:
    st.info("El periodograma necesita al menos 2 días.")
else:
    pbin = max(bin_min, 10)
    pact = counts.loc[(idx >= pstart) & (idx < end), list(act.columns)]
    pact = pact.resample(f"{pbin}min", origin=pstart).sum()
    st.caption(f"Sokolove-Bushell, α = 0.05 · {pdays_eff} días desde {pstart:%d/%m}. "
               "Ideal para días en oscuridad constante (ajusta «últimos N días» en la barra lateral).")
    rows, curves, thr = [], {}, None
    for f in pact.columns:
        d = chisq_periodogram(pact[f].to_numpy(), pbin)
        if d.empty:
            continue
        thr = d.set_index("periodo_h")["umbral"]
        curves[f] = d.set_index("periodo_h")["Qp"]
        j = d["Qp"].idxmax()
        pw_ = d.loc[j, "Qp"] - d.loc[j, "umbral"]
        rows.append(dict(Fly=f, Genotipo=geno[f], periodo_h=d.loc[j, "periodo_h"],
                         potencia=pw_, ritmico=bool(pw_ > 0)))
    if rows:
        per = pd.DataFrame(rows)
        figp = go.Figure()
        for i, g in enumerate(sorted(per["Genotipo"].unique())):
            cur = pd.concat([curves[f] for f in per.loc[per["Genotipo"] == g, "Fly"]], axis=1).mean(axis=1)
            figp.add_trace(go.Scatter(x=cur.index, y=cur, name=g, line=dict(color=COLORS[i % len(COLORS)])))
        figp.add_trace(go.Scatter(x=thr.index, y=thr, name="p = 0.05", line=dict(color="red", dash="dash")))
        figp.update_layout(xaxis_title="Periodo (h)", yaxis_title="Qp (promedio)", height=420)
        st.plotly_chart(figp)
        r = per[per["ritmico"]].groupby("Genotipo").agg(
            n_ritmicas=("Fly", "count"), periodo_medio_h=("periodo_h", "mean"),
            periodo_sem=("periodo_h", "sem"), potencia_media=("potencia", "mean")).round(3)
        r["n_total"] = per.groupby("Genotipo")["Fly"].count()
        st.markdown("**Moscas rítmicas por genotipo**")
        st.dataframe(r)
        with st.expander("Periodo por mosca"):
            st.dataframe(per.round(3))
    else:
        per = pd.DataFrame()
        st.warning("Muy pocos datos para calcular el periodograma.")

st.header("Descargas")
d1, d2, d3, d4 = st.columns(4)
d1.download_button("Métricas por día (CSV)", dm.to_csv(index=False), "metricas_por_dia.csv")
d2.download_button("Métricas por mosca (CSV)", perfly.to_csv(index=False), "metricas_por_mosca.csv")
d3.download_button("Estadística (CSV)", tbl.to_csv(index=False), "estadistica.csv")
if pdays_eff >= 2 and rows:
    d4.download_button("Periodos (CSV)", per.to_csv(index=False), "periodos.csv")
