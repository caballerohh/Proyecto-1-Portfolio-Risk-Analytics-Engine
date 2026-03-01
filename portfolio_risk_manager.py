pip install reportlab

# LIBRARY
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis, norm, linregress, skew, jarque_bera
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# ============================
# 1. Parámetros del portafolio
# ============================
# Bloque 1: Portafolio y pesos
portfolio = {

    "SPY": 0.1597,
    "DIA": 0.1619,
    "BND": 0.1478,
    "JPM": 0.1318,
    "GS": 0.1317,
    "AMZN": 0.1317,
    "AAPL": 0.1311
}

#Fecha de compra (solo para estudio pre y post)
fecha_compra = "2025-08-20" 
fecha_compra_dt = pd.to_datetime(fecha_compra)

#Parámetros de fechas para el análisis y benchmark
benchmark = "SPY" #BENCHMARK

#Nuevas fechas de inicio y fin para el análisis
fecha_analisis_inicio = "2025-08-20" 
fecha_analisis_cierre = datetime.today().strftime("2025-11-25") 

fecha_analisis_inicio_dt = pd.to_datetime(fecha_analisis_inicio)
fecha_analisis_cierre_dt = pd.to_datetime(fecha_analisis_cierre)

data_download_start_date = fecha_analisis_inicio_dt - timedelta(days=252*1.5) # 
fecha_inicio_descarga = data_download_start_date.strftime("%Y-%m-%d")

fecha_ejecucion = datetime.now()
fecha_ejecucion_peru = fecha_ejecucion - timedelta(hours=4) #Cambiar hora a Perú

# Color de fondo para gráficos
background_color = "white"

# Tamaño de imagenes en PDF
IMG_WIDTH = 480
IMG_HEIGHT = 240


# ============================
# 2. Cosntruir el portafolio
# ============================
tickers = list(portfolio.keys()) + [benchmark]
# Descargar datos de YH finance
data_all = yf.download(tickers, start=fecha_inicio_descarga, end=fecha_analisis_cierre, auto_adjust=True)["Close"].dropna()

# Limpiar datos en dias cotizados (252 por año) 
if fecha_compra_dt not in data_all.index:
    precios_at_purchase_date = data_all.reindex([fecha_compra_dt], method='nearest')
    if precios_at_purchase_date.empty or pd.isna(precios_at_purchase_date.iloc[0][list(portfolio.keys())]).any():
         print(f"Error: Could not find a valid trading day price near {fecha_compra}.")
         raise ValueError(f"No data available near purchase date {fecha_compra}")

    # Seleccionar datos obtenidos validos sin N/A
    available_keys = [key for key in portfolio.keys() if key in precios_at_purchase_date.columns]
    precios_compra = precios_at_purchase_date.iloc[0][available_keys]
    closest_date = precios_at_purchase_date.index[0].strftime('%Y-%m-%d')
    print(f"Warning: Exact purchase date {fecha_compra} not found. Using prices from {closest_date}")
    fecha_compra_dt = precios_at_purchase_date.index[0] # Update 
else:
    precios_compra = data_all.loc[fecha_compra_dt, list(portfolio.keys())]


# Determinar el tiempo de análisis
data_for_analysis = data_all[(data_all.index >= fecha_analisis_inicio_dt) & (data_all.index <= fecha_analisis_cierre_dt)].copy()

if data_for_analysis.empty:
    raise ValueError(f"No data available for the analysis period from {fecha_analisis_inicio_dt.strftime('%Y-%m-%d')} to {fecha_analisis_cierre_dt.strftime('%Y-%m-%d')}. Please check the dates and data availability.")
data_for_rolling = data_all.copy()


# ============================
# 3. Retornos y métricas del portafolio
# ============================
# Calculando métricas puntuales del portafolio
precios = data_for_analysis[list(portfolio.keys())]
returns = precios.pct_change().dropna()
weights = np.array(list(portfolio.values()))
portfolio_returns = returns.dot(weights)

# Calcular retornos del portafolio
full_portfolio_returns = data_for_rolling[list(portfolio.keys())].pct_change().dropna().dot(weights)

# Calcular retornos multihorizontes históricos (dia final es el dia de ejecución del script)
returns_1d = portfolio_returns.iloc[-1]
returns_1w = portfolio_returns.rolling(5).apply(lambda x: (1+x).prod()-1).iloc[-1]
returns_1m = portfolio_returns.rolling(21).apply(lambda x: (1+x).prod()-1).iloc[-1]
returns_1y = full_portfolio_returns[full_portfolio_returns.index <= fecha_analisis_cierre_dt].rolling(252).apply(lambda x: (1+x).prod()-1).iloc[-1]

# YTD retorno
start_of_analysis_year = pd.Timestamp(f"{fecha_analisis_cierre_dt.year}-01-01")
start_of_year_returns = full_portfolio_returns[(full_portfolio_returns.index >= start_of_analysis_year) & (full_portfolio_returns.index <= fecha_analisis_cierre_dt)]
returns_ytd = (start_of_year_returns + 1).prod() - 1

# Calcular elretorno del periodo de análisis (fecha inicio -> Fecha final)
total_return = (portfolio_returns + 1).prod() - 1


# Calculo de métricas de VaR y CVaR históricos
VaR_95 = np.percentile(portfolio_returns, 5)
VaR_99 = np.percentile(portfolio_returns, 1)
CVaR_95 = portfolio_returns[portfolio_returns < VaR_95].mean()
CVaR_99 = portfolio_returns[portfolio_returns < VaR_99].mean()

# Rendimientos y drawdown
cumulative = (1 + portfolio_returns).cumprod()
rolling_max = cumulative.cummax()
drawdown = (cumulative - rolling_max) / rolling_max
max_dd = drawdown.min()

# Ratios financieros
rf = 0.0365 / 252 #Tasa de T-bills diario
sharpe = (portfolio_returns.mean() - rf) / portfolio_returns.std() * np.sqrt(252)
downside_returns = portfolio_returns[portfolio_returns < 0]
sortino = (portfolio_returns.mean() - rf) / downside_returns.std() * np.sqrt(252) if downside_returns.std() != 0 else np.nan
calmar = (portfolio_returns.mean() * 252) / abs(max_dd) if max_dd != 0 else np.nan

# Beta y Jensen Alpha vs benchmark
bench_returns = data_for_analysis[benchmark].pct_change().dropna()
aligned = pd.concat([portfolio_returns, bench_returns], axis=1, join="inner").dropna()
aligned.columns = ["Portafolio", benchmark]
if len(aligned) > 1:
    beta, alpha, r_val, p_val, std_err = linregress(aligned[benchmark], aligned["Portafolio"])
    jensen_alpha = alpha * 252
    r2 = r_val**2
    # Tracking error vs benchmark
    tracking_error = np.std(aligned["Portafolio"] - aligned[benchmark]) * np.sqrt(252)
else:
    beta = np.nan
    alpha = np.nan
    r_val = np.nan
    p_val = np.nan
    std_err = np.nan
    jensen_alpha = np.nan
    r2 = np.nan
    tracking_error = np.nan
    print("Warning: Insufficient data to calculate Beta, Alpha, R-squared, and T.E.")

# Hecho estilizados (distribucción de rendimientos)
skewness_val = skew(portfolio_returns)
kurtosis_val = kurtosis(portfolio_returns)
jb_stat, jb_p_val = jarque_bera(portfolio_returns)
sample_days = len(portfolio_returns)
mean_return = np.nanmean(portfolio_returns)
median_return = np.nanmedian(portfolio_returns)
std_dev_return = np.nanstd(portfolio_returns)


# ============================
# 4. Funciones para graficar
# ============================
plt.rcParams.update({"figure.autolayout": True})
sns.set_style("whitegrid")

def _apply_axes_bg(ax, bg_color):
    if bg_color != "transparent":
        ax.set_facecolor(bg_color)

def save_chart(filename, plot_func, figsize=(10,4)):
    if background_color == "transparent":
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure(figsize=figsize, facecolor=background_color)
    ax = fig.add_subplot(111)
    try:
        plot_func(ax)
    except TypeError:
        plot_func()

    # Remover lineas de fondo y color de fondo
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)

    if background_color == "transparent":
        fig.patch.set_alpha(0.0)
        plt.savefig(filename, dpi=150, transparent=True)
    else:
        plt.savefig(filename, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)



# ============================
# 5. Graficos
# ============================

# 5.1 Rendimiento de la cartera
port_precio = (data_for_analysis[list(portfolio.keys())] / data_for_analysis[list(portfolio.keys())].iloc[0]).dot(weights)

def plot_precio(ax):
    ax.plot(port_precio.index, port_precio.values, label="Portafolio",linestyle="--", linewidth=1.5)
    if fecha_compra_dt >= fecha_analisis_inicio_dt and fecha_compra_dt <= fecha_analisis_cierre_dt:
        ax.axvline(fecha_compra_dt, color="red", linestyle="--", label="Fecha Compra")
    ax.set_title("Graph 1: Evolución del valor del portafolio")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio")
    ax.legend()

save_chart("precio_hist.png", plot_precio, figsize=(10,5))

# 5.2 Retornos diarios
def plot_retornos(ax):
    ax.plot(portfolio_returns.index, portfolio_returns.values, alpha=0.9,linestyle="--", linewidth=1.5)
    ax.set_title("Graph 2: Rendimiento no acumulados del portafolio")
    ax.set_ylabel("Retorno")

save_chart("retornos_diarios.png", plot_retornos, figsize=(10,5))

# 5.3 Mapa de calor de retornos diarios
daily_returns_monthly = portfolio_returns.copy()
daily_returns_monthly.index = pd.MultiIndex.from_arrays([daily_returns_monthly.index.year, daily_returns_monthly.index.month, daily_returns_monthly.index.day], names=["year", "month", "day"])
heatmap_data_daily = daily_returns_monthly.unstack(level="day")

def plot_heatmap(ax):
    # Annotations removed
    sns.heatmap(heatmap_data_daily, annot=False, fmt=".2%", cmap="RdYlGn", center=0, ax=ax, cbar=False, annot_kws={"color": "black"})
    ax.set_title("Graph 3: Heatmap de retornos diarios (mes vs día)")
    ax.set_ylabel("Año, Mes")
    ax.set_xlabel("Día del Mes")
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)

save_chart("return_heatmap.png", plot_heatmap, figsize=(10,5))


# 5.4 Rendimiento acumulado
def plot_cum(ax):
    ax.plot(cumulative.index, cumulative.values, alpha=0.9,linestyle="--", linewidth=1.5)
    ax.set_title("Graph 4: Rendimiento acumulado del portafolio")
    ax.set_ylabel("Crecimiento acumulado")
save_chart("rend_acumulado.png", plot_cum, figsize=(10,5))

# 5.5 Volatilidad anualizada rolling (21 días)
vol = (full_portfolio_returns.rolling(21).std()*np.sqrt(252)).dropna()
vol_plot = vol[(vol.index >= fecha_analisis_inicio_dt) & (vol.index <= fecha_analisis_cierre_dt)] # Slice for plotting
def plot_vol(ax):
    ax.plot(vol_plot.index, vol_plot.values)
    ax.set_title("Graph 5: Volatilidad Rolling del portafolio (21 días)")
    ax.set_ylabel("Volatilidad anualizada")
save_chart("volatilidad.png", plot_vol, figsize=(10,5))

# 5.6 Distribución de retornos
def plot_dist(ax):
    data = portfolio_returns.values.flatten()
    ax.hist(data, bins=50, density=True)
    x = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 200)
    p = norm.pdf(x, np.nanmean(data), np.nanstd(data))
    ax.plot(x, p, "r", linewidth=2)
    ax.set_title("Graph 6: Distribución de retornos del Portafolio")
save_chart("asimetria.png", plot_dist, figsize=(10,5))

# 5.7 Curtosis rolling (21 días).
kurt = full_portfolio_returns.rolling(21).apply(lambda x: kurtosis(x)).dropna()
kurt_plot = kurt[(kurt.index >= fecha_analisis_inicio_dt) & (kurt.index <= fecha_analisis_cierre_dt)] # Slice for plotting
def plot_curtosis(ax):
    ax.plot(kurt_plot.index, kurt_plot.values)
    ax.set_title("Graph 7: Curtosis Rolling del portafolio (21 días)")
    ax.set_ylabel("Curtosis")
save_chart("curtosis.png", plot_curtosis, figsize=(10,5))

# 5.8 CVaR Rolling (21 días)
rolling_cvar = full_portfolio_returns.rolling(21).apply(lambda x: x[x < np.percentile(x,5)].mean()).dropna()
rolling_cvar_plot = rolling_cvar[(rolling_cvar.index >= fecha_analisis_inicio_dt) & (rolling_cvar.index <= fecha_analisis_cierre_dt)] # Slice for plotting
def plot_cvar(ax):
    rolling_cvar_plot.plot(ax=ax)
    ax.set_title("Graph 8: CVaR Rolling del portafolio (21 días)")
    ax.set_ylabel("CVaR")
save_chart("cvar_rolling.png", plot_cvar, figsize=(10,5))

# 5.9 Mapa de Drawdown Diario
daily_dd_monthly = drawdown.copy()
daily_dd_monthly.index = pd.MultiIndex.from_arrays([daily_dd_monthly.index.year, daily_dd_monthly.index.month, daily_dd_monthly.index.day], names=["year", "month", "day"])
dd_heatmap_daily = daily_dd_monthly.unstack(level="day")
def plot_dd_heatmap(ax):
    sns.heatmap(dd_heatmap_daily, annot=False, fmt=".2%", cmap="Reds", center=0, ax=ax, cbar=False, annot_kws={"color": "black"})
    ax.set_title("Graph 9: Drawdowns diarios del portafolio (mes vs día)")
    ax.set_ylabel("Año, Mes")
    ax.set_xlabel("Día del Mes")
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)

save_chart("dd_heatmap.png", plot_dd_heatmap, figsize=(10,5))

# 5.10 Backtesting VaR Diario
def plot_var_backtest(ax):
    ax.plot(portfolio_returns.index, portfolio_returns.values, label="Retornos", alpha=0.8)
    # Calculate daily VaR 95% using a rolling window of 21 days on full_portfolio_returns
    daily_var_95 = full_portfolio_returns.rolling(21).apply(lambda x: np.percentile(x, 5)).dropna()
    # Reindex daily_var_95 to match portfolio_returns index for comparison within the analysis period
    daily_var_95_reindexed = daily_var_95.reindex(portfolio_returns.index, method='nearest')

    ax.plot(daily_var_95_reindexed.index, daily_var_95_reindexed.values, color="blue", linestyle="--", label="VaR 95% Diario (21-day rolling)")
    # Find exceedances by comparing portfolio_returns with the reindexed daily_var_95 within the analysis period
    exceedances = portfolio_returns[portfolio_returns < daily_var_95_reindexed]
    ax.scatter(exceedances.index, exceedances, color="red", s=10, label="Excedencias")
    ax.set_title("Graph 10: Backtesting VaR 95% diario del portafolio")
    ax.legend()
save_chart("var_backtest.png", plot_var_backtest)

# 5.11 Portafolio vs Benchmark
bench_returns = data_for_analysis[benchmark].pct_change().dropna()
bench_cum = (1 + bench_returns).cumprod()
def plot_comp(ax):
    ax.plot(cumulative.index, cumulative.values, label="Portafolio", color="darkblue")
    ax.plot(bench_cum.index, bench_cum.values, label=benchmark, color="coral")
    # Check if fecha_compra_dt is within the analysis period before plotting
    if fecha_compra_dt >= fecha_analisis_inicio_dt and fecha_compra_dt <= fecha_analisis_cierre_dt:
        ax.axvline(fecha_compra_dt, color="red", linestyle="--", label="Fecha Compra")
    ax.set_title("Graph 11: Evolución de rendimientos del Portafolio vs Benchmark")
    ax.legend()
save_chart("comparacion_acumulada.png", plot_comp, figsize=(10,6))

# 5.12 Matriz de Correlación de Activos
correlation_matrix = returns.corr()

def plot_correlation_matrix(ax):
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Graph 12: Matriz de Correlación de activos")
    _apply_axes_bg(ax, background_color)

save_chart("correlation_matrix.png", plot_correlation_matrix, figsize=(10,6))

# 5.13 Contribución Semanal de Activos
weekly_returns = returns.resample('W').apply(lambda x: (1 + x).prod() - 1).dropna()
weekly_portfolio_returns = weekly_returns.dot(weights)

# Definir colores por activos
custom_colors = ['navy', 'royalblue', 'blue','lightsteelblue', 'cornflowerblue', 'slateblue']
#PORTAFOLIO orden = SPY, DIA, BND, JPM, GS, AMZN, AAPL
#OTRO ORDEN DE COLORES: 'lightsteelblue', 'comflowerblue', 'royalblue', 'blue', 'slateblue','navy'


# CONTRIBUCION SEMANA POR ACTIVO
weekly_contributions = weekly_returns.multiply(weights, axis=1)
weekly_contributions = weekly_contributions.iloc[-16:] 

def plot_weekly_contributions(ax, colors):
    # 1. Filtramos los datos localmente para usar solo las últimas 16 semanas
    data_plot = weekly_contributions.iloc[-16:]
    returns_plot = weekly_portfolio_returns.iloc[-16:]
  
    # 2. Dibujar las barras apiladas (width=0.85 hace las barras más anchas)
    data_plot.plot(kind='bar', stacked=True, ax=ax, width=0.85, color=colors)

    # 3. Títulos y etiquetas
    ax.set_title("Contribución Semanal de Activos (Últimas 16 Semanas)", 
                 fontsize=12, weight='bold')
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Contribución semanal (%)")
    
    # Leyenda ajustada a la derecha
    ax.legend(title="Activos", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')

    # 4. Corregir las fechas en el eje X
    labels = [item.strftime('%Y-%m-%d') for item in data_plot.index]
    ax.set_xticklabels(labels, rotation=45, ha='right')

    _apply_axes_bg(ax, background_color)
    ax.axhline(0, color='black', linestyle='-', linewidth=1.5)

    # 5. Formatear eje Y como porcentaje
    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter(1))

    # 6. Etiquetas externas (Total, MAX y MIN)
    y_min, y_max = ax.get_ylim()
    # AJUSTE SUPERIOR: Aumentamos la separación entre Total y MAX
    y_top_total = y_max * 1.25
    y_top_max = y_max * 1.05   
    y_bottom_min = y_min * 1.1 # Posición más arriba para el MIN Contributor
   
    for i in range(len(data_plot)):
        contribs = data_plot.iloc[i]
        
        # Obtenemos el rendimiento total semanal para la columna actual (i)
        total_return_value = returns_plot.iloc[i] 
        total_return_color = 'green' if total_return_value >= 0 else 'red'

        #Rendimiento Total del Portafolio (Máxima altura)
        ax.text(i, y_top_total, 
                f'{total_return_value:.2%}',
                ha='center', va='bottom', fontsize=7, 
                color=total_return_color, fontweight='bold', clip_on=True)

        # Máximo Contribuyente (Etiqueta 2: Verde)
        max_asset_name = contribs.idxmax()
        max_contrib_value = contribs.max()
        
        if max_contrib_value > 0:
            # 2) DOS LÍNEAS: TICKET arriba, DATO abajo, usando \n
            ax.text(i, y_top_max, 
                    f'MAX {max_asset_name}\n{max_contrib_value:.1%}',
                    ha='center', va='bottom', fontsize=7, 
                    color='darkgreen', fontweight='bold', clip_on=True)

        # Mínimo Contribuyente (Etiqueta 3: Rojo)
        min_asset_name = contribs.idxmin()
        min_contrib_value = contribs.min()

        if min_contrib_value < 0:
            # 2) DOS LÍNEAS: TICKET arriba, DATO abajo, usando \n
            ax.text(i, y_bottom_min, 
                    f'MIN {min_asset_name}\n{min_contrib_value:.1%}',
                    ha='center', va='top', fontsize=7, 
                    color='darkred', fontweight='bold', clip_on=True)

    # 8. Ajustar límites finales
    # Aseguramos suficiente espacio para TODAS las etiquetas externas (Total, MAX y MIN).
    ax.set_ylim(y_bottom_min * 1.25, y_max * 1.50)

# Pass the custom colors to save_chart
save_chart("weekly_contribution.png", lambda ax: plot_weekly_contributions(ax, custom_colors), figsize=(12,9))



# ============================
# 6. CONFIGURAR PDF
# ============================
doc = SimpleDocTemplate(f"Reporte_Cuantitativo_de_Portafolio.pdf", pagesize=letter,
                        rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='AnalisisBold', parent=styles['Normal'], fontName='Helvetica-Bold'))
styles.add(ParagraphStyle(name='Analisis', parent=styles['Normal'], leftIndent=0, spaceAfter=6))
styles.add(ParagraphStyle(
    name="Clean",
    parent=styles["Normal"],
    fontSize=10,
    leading=14,
    leftIndent=0,
    firstLineIndent=0,
    spaceAfter=6
))

# Checkear los bullets para descripción
if 'Bullet' not in styles:
    styles.add(ParagraphStyle(
        name="Bullet",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        leftIndent=18, # Adjust as needed for spacing
        firstLineIndent=-18, # Adjust as needed for alignment with bullet
        spaceAfter=6,
        bulletFontName='Helvetica',
        bulletFontSize=10,
        bulletIndent=0,
        bulletText='•'
    ))


story = []

def footer(canvas, doc):
    canvas.saveState()
    footer_text = f"Reporte Cuantitativo de Riesgo del Portafolio. Elaborado por Carlos Caballero."
    page_number = f"Página {doc.page}"
    canvas.setFont('Helvetica', 8)
    canvas.drawString(40, 20, footer_text)
    canvas.drawRightString(570, 20, page_number)
    canvas.restoreState()

story.append(Paragraph(f"Reporte Cuantitativo de Riesgo del Portafolio", styles["Title"]))
story.append(Spacer(1, 8))
story.append(Paragraph(f"Fecha de Ejecución: {fecha_ejecucion_peru.strftime('%A, %d de %B de %Y %H:%M:%S')}", styles["Clean"])) # Updated to show execution date and time
story.append(Spacer(1, 12))

# Resumen ejecutivo
summary_data = [
    ["Sharpe", f"{sharpe:.2f}", "Sortino", f"{sortino:.2f}", "Calmar", f"{calmar:.2f}"],
    ["Alpha Jensen", f"{jensen_alpha:.2%}", "Beta", f"{beta:.2}", "R²", f"{r2:.2%}"],
    ["Tracking Error", f"{tracking_error:.2%}", "Máx Drawdown", f"{max_dd:.2%}", "VaR 95%", f"{VaR_95:.2%}"],
    ["VaR 99%", f"{VaR_99:.2%}", "CVaR 95%", f"{CVaR_95:.2%}", "CVaR 99%", f"{CVaR_99:.2%}"]
]
summary_table = Table(summary_data, colWidths=[80,60,80,60,80,60], hAlign="CENTER")
summary_table.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,-1), colors.white),
    ("TEXTCOLOR", (0,0), (-1,-1), colors.black),
    ("ALIGN", (0,0), (-1,-1), "CENTER"),
    ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
    ("FONTSIZE", (0,0), (-1,-1), 9),
    ("GRID", (0,0), (-1,-1), 0.25, colors.grey)
]))
story.append(Paragraph("📌 Resumen Ejecutivo", styles["Heading1"]))
story.append(summary_table)
story.append(Spacer(1, 8))

# Tabla de rendimientos multihorizonte
fin_table_data = [
    ["1 día", f"{returns_1d:.2%}", "1 semana", f"{returns_1w:.2%}", "1 mes", f"{returns_1m:.2%}"],
    ["YTD", f"{returns_ytd:.2%}", "1 año", f"{returns_1y:.2%}", "Rendimiento", f"{total_return:.2%}"]
]
fin_table = Table(fin_table_data, colWidths=[85,70,85,70,85,70], hAlign="CENTER")
fin_table.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,-1), colors.white),
    ("TEXTCOLOR", (0,0), (-1,-1), colors.black),
    ("ALIGN", (0,0), (-1,-1), "CENTER"),
    ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
    ("FONTSIZE", (0,0), (-1,-1), 9),
    ("GRID", (0,0), (-1,-1), 0.25, colors.grey)
]))
story.append(Paragraph("📈 Rendimientos Multihorizonte", styles["Heading2"]))
story.append(fin_table)
story.append(Spacer(1, 8))

# Análisis editable para distribución
skewness_val = skew(portfolio_returns)
kurtosis_val = kurtosis(portfolio_returns)
jb_stat, jb_p_val = jarque_bera(portfolio_returns)
sample_days = len(portfolio_returns)
mean_return = np.nanmean(portfolio_returns)
median_return = np.nanmedian(portfolio_returns)
std_dev_return = np.nanstd(portfolio_returns)

# Análisis del portafolio (3 puntos de análisis)
story.append(Paragraph("Análisis del Portafolio", styles["Heading2"]))
story.append(Spacer(1, 8))
story.append(Paragraph("• El desempeño del portafolio se caracteriza por una eficiencia superior y una dinámica de crecimiento controlada, validada por un Rendimiento YTD de 16.67% y un Rendimiento Total de 6.66%. La eficiencia del retorno es notable: el Ratio de Sharpe es de 1.7 y el Ratio de Sortino de 2.5. Este alto Sortino confirma que la cartera es excepcionalmente eficiente en la retención de Alpha y la mitigación del riesgo a la baja. La dinámica muestra una fase inicial de acumulación (Agosto-Septiembre) con ganancia suave, seguida de una recuperación de tendencia que alcanzó un ATH de +10.5% del valor de su precio de compra.", styles["Bullet"]))
story.append(Spacer(1, 18))
story.append(Paragraph("• La gestión de riesgo es robusta y coherente. El Máx Drawdown fue contenido en -5.34%, lo que otorga un alto Ratio de Calmar de 4.69. La Volatilidad Rolling experimentó un drástico cambio de régimen, saltando de un mínimo de 0.06 a 0.14 post-shock. Este evento de estrés temporal se localiza con el pico de 7 en la Curtosis Rolling y la CVaR vs VaR es controlada (VaR 99%: -2% vs CVaR 99%: -2.2%). Finalmente, la validación del modelo es sólida: el Backtesting del VaR reporta una Tasa de Falla Empírica de 4.35%, validando su calibración (Target del 5%).", styles["Bullet"]))
story.append(Spacer(1, 18))
story.append(Paragraph("• El portafolio genera Alpha frente al Benchmark: el Alpha Jensen es de 0.074 con una Beta de 0.91 (R2 de 0.78), implicando menor sensibilidad al riesgo sistemático. La estructura de diversificación es clave: la Matriz de Correlación muestra que los bonos (BND) exhiben una correlación negativa con la renta variable, actuando como amortiguador. Los eventos macro como la caida por Imposición de aranceles a China por parte de Trump y los recortes de la FED tuvieron efectos dinámicos en los mercados, así como las expectativas de burbuja tecnológica a finales de Noviembre. Sin embargo, el portafolio mostró una mejor dinámica de recuperación.", styles["Bullet"]))
story.append(Spacer(1, 12))

### Insertar espacios de Análisis para cada
def add_graph_to_story(img_path, title, desc, analysis_text, story, styles, img_w=IMG_WIDTH, img_h=IMG_HEIGHT, counter=[0]):
    story.append(Paragraph(title, styles["Heading1"]))
    story.append(Paragraph(desc, styles["Clean"]))
    if analysis_text:
        story.append(Paragraph("<b>Análisis:</b> " + analysis_text, styles["Analisis"]))
    story.append(Spacer(1, 6))

    if "weekly_contribution" in img_path:
        img = Image(img_path, width=500, height=400)   # Más grande para este gráfico
    else:
        img = Image(img_path, width=400, height=200)

    story.append(img)
    story.append(Spacer(1, 12))
    counter[0] += 1
    if counter[0] % 2 == 0:
        story.append(PageBreak())

story.append(PageBreak())

analisis_textos = {
    "precio_hist.png": "El portafolio muestra una tendencia alcista, pasando de 100 a 106.6% en tres periodos: 1) Fase de acumulación (Agosto a Septiembre), una ganancia constante y relativamente suave terminando con una correción rápida (15 de Octubre), 2) Fase de recuperación de tendencia (2da mitad de Octubre a Noviembre) con un recuperación rápida y alcanzando ATH en 110.5%. 3) Fase de caida y recuperación: Finales de Noviembre por expectativas de sobrevaloración de activos en el mercado (burbuja tecnológica).",
    "retornos_diarios.png": "La serie de retornos confirma el fenómeno de Agrupamiento {Volatility Clustering}. Esto implica que la varianza de los retornos no es constante, lo que valida la necesidad de utilizar modelos  como GARCH en lugar del VaR Histórico. El pico de retorno negativo observado en octubre es el evento de riesgo definitorio del periodo.",
    "return_heatmap.png": "El Heatmap verifica que el riesgo de pérdida no presenta patrones estacionales o sesgos de calendario. La concentración de rojo oscuro en un día específico de octubre confirma la naturaleza de riesgo idiosincrático o evento concentrado del shock, lo cual es más fácil de modelar y diversificar que un riesgo sistémico prolongado",
    "rend_acumulado.png": "La cartera presenta una eficiencia superior con un Rendimiento de 6.67%. Los Ratios Sharpe (1.7) y Sortino (2.5) validan la tesis de crecimiento compensado por un bajo riesgo a la baja.",
    "volatilidad.png": "El gráfico demuestra un cambio de Régimen de Riesgo: la volatilidad anualizada se duplicó de 0.06 a 0.14 post-shock, manteniendo en niveles altos durante Noviembre. Esto justifica la reevaluación de coberturas y la necesidad de modelos de volatilidad condicional.",
    "asimetria.png": f"El portafolio presenta una Asimetría Negativa, confirmando la exposición al riesgo de cola evidenciando que existe un mayor efecto de caida ante eventualidades. Sin embargo, la baja Curtosis sugiere que la diversificación mitigó las colas pesadas. El p-value de la prueba JB significa que el VaR es estadísticamente viable, aunque no necesariamente el modelo óptimo. Muestras: Sample Days: {sample_days}; Mean: {mean_return:.4f}; Median: {median_return:.4f}; Std Dev: {std_dev_return:.4f}; Skewness: {skewness_val:.4f}; Kurtosis: {kurtosis_val:.4f}; Jarque-Bera Stat: {jb_stat:.2f}; p-value: {jb_p_val:.4f}.",
    "curtosis.png": "El gráfico localiza el riesgo de cola: la Curtosis se disparó a un pico extremo de 7 en octubre. Este evento de Leptokurtosis aguda demuestra que las colas pesadas existen en periodos de estrés. Se valida la necesidad de usar modelos t-Student para estimar el riesgo.",
    "cvar_rolling.png": "Las métricas de pérdida son controladas: VaR 99% (-2%) y cVaR 99% (-2.2%) muestran que la gravedad de la pérdida no se dispara significativamente más allá del umbral de VaR histórico, confirmando un riesgo de cola contenido.",
    "dd_heatmap.png": "El mapa valida la alta liquidez y la gestión eficaz, ya que las caídas son seguidas de recuperaciones intradía. No se observa ciclicidad en las pérdidas, demostrando que la exposición al riesgo no es estructural.",
    "var_backtest.png": "El Backtesting es robusto: la Tasa de Falla Empírica fue 4.35%, muy cercana al 5% esperado. Esto valida la calibración del modelo VaR histórico para la planificación de Budget risk.",
    "comparacion_acumulada.png": "El portafolio genera Alpha: Alpha Jensen es 0.074 con una Beta de 0.91. Esto confirma que el rendimiento superior se logra con menor sensibilidad al riesgo sistemático al de SPY, teniendo una dinámica de menor volatilidad.",
    "correlation_matrix.png": "BND es el amortiguador de riesgo esencial, exhibiendo una correlación baja/negativa con la renta variable. Esta baja correlación es la clave del alto Sharpe y el bajo Drawdown. La baja correlación entre activos tecnológicos y financieros refuerzan la tesis de diversificación y posicionamineto estratégico dentro del portafolio.",
    "weekly_contribution.png": "El Risk Budgeting es claro: AAPL, AMZN y JPM son los motores de Alpha (máximo rendimiento) mientras que SPY y DIA diversificadores en el portafolio. BND es el componente de estabilidad al tener una contribución marginal al retorno total, reduciendo la exposición a pérdidas semanales."
}

### Descripción de los graficos
graficos = [
    ("precio_hist.png", "Rendimiento de la cartera", "Muestra el rendimiento histórico de la cartera a lo largo del tiempo, normalizado a un punto de inicio. Es el indicador principal del crecimiento del capital. Es la prueba visual de la generación de Alfa (rendimiento superior)."),
    ("retornos_diarios.png", "Retornos diarios", "Ilustra la variación porcentual de valor del portafolio diario. Es la serie fundamental a partir de la cual se calculan casi todas las métricas de riesgo. Retornos centrados en 0.00 o ligeramente por encima. La métrica clave es el agrupamiento de volatilidad (grandes cambios seguidos de grandes cambios). Los picos negativos alimentan directamente los cálculos de VaR y CVaR."),
    ("return_heatmap.png", "Mapa de calor de retornos diarios", "Evaluación visual rápida de los retornos diarios por mes y día, usando colores para resaltar los retornos extremos (alto/bajo), identificando ausencia de patrones estacionales. Ayuda a la identificación de anomalías o eventos de riesgo concentrado."),
    ("rend_acumulado.png", "Rendimiento acumulado", "Mide la ganancia o pérdida total desde el inicio del período, asumiendo la reinversión de todos los retornos. Es la métrica decisiva de Performance. Esencial para calcular el Ratio de Calmar y valida la tesis de crecimiento positivo."),
    ("volatilidad.png", "Volatilidad anualizada rolling (21 días).", "Representa el nivel de riesgo (fluctuación) del portafolio, calculada y anualizada continuamente sobre una ventana de 21 días (aprox. un mes de cotizaciones). Mide la incertidumbre futura. La presencia de shocks validan los comportamientos típicos del mercado donde los grandes choques impulsan las expectativas de volatilidad futura."),
    ("asimetria.png", "Distribución de retornos", "Visualiza la forma de la distribución de retornos, comparándola con una distribución normal (curva roja). Evalúa el Riesgo de No-Normalidad. La Asimetría negativa implica que los retornos negativos extremos son más probables. Una Curtosis inferior a 3 sugiere colas menos pesadas que la normal, lo cual es positivo para la modelización de riesgo."),
    ("curtosis.png", "Curtosis rolling (21 días).", "Mide el grado en que la distribución de retornos presenta colas extremas (valores atípicos) en la ventana de 21 días. Un pico extremo indica la presencia temporal de leptokurtosis aguda. Es crucial para entender los períodos de riesgo extremo y calibrar escenarios de estrés."),
    ("cvar_rolling.png", "CVaR Rolling (21 días).", "Muestra el Expected Shortfall {CVaR), la pérdida promedio esperada en el 5% de los peores escenarios del VaR. Cuantifica la gravedad de las pérdidas extremas. El regreso a niveles iniciales y más negativos tras eventos de shock, sugiere que el portafolio se vuelve más vulnerable a pérdidas de cola."),
    ("dd_heatmap.png", "Mapa de Drawdown Diario", "Identifica visualmente la magnitud y ocurrencia de caídas de valor desde un pico máximo anterior. Evalúa el Máximo Drawdown (MDD) y el tiempo de recuperación. Las caídas con recuperación intradía indican buena liquidez y gestión de riesgo a corto plazo. Confirma la ausencia de ciclicidad en las pérdidas."),
    ("var_backtest.png", "Backtesting VaR Diario", "Evalúa la precisión del modelo de VaR al comparar las pérdidas reales con el límite de pérdida predicho. La Tasa de Falla Empírica (número de excedencias / total de días) debe ser igual o muy cercana al nivel de significancia (VaR 95%  equivalente al 5%), lo cual valida la fiabilidad del modelo de riesgo."),
    ("comparacion_acumulada.png", "Comparación vs Benchmark", "Compara el rendimiento acumulado de la cartera con un índice de referencia (Benchmark). La curva del Portafolio debe mantenerse consistentemente por encima de la curva del Benchmark. Determinación de Alfa, la superación sistemática y la mejor recuperación relativa 35 validan la estrategia de gestión activa y explican el alto Ratio de Sharpe del portafolio."),
    ("correlation_matrix.png", "Matriz de Correlación de Activos", "Visualiza el grado y la dirección en que los retornos de los diferentes activos se mueven juntos. Correlaciones altas/positivas (rojo) dentro de clases (e.g., acciones entre sí) y correlaciones bajas/negativas (azul/claro) entre clases (e.g., acciones vs. bonos)."),
    ("weekly_contribution.png", "Contribución Semanal de Activos", "Detalla el porcentaje del retorno total semanal que fue generado por cada activo individual. Esto ayuda a identificar los principales motores de rendimiento (ganadores) y las mayores fuentes de pérdidas (perdedores) en la cartera en períodos cortos.")
]

for g in graficos:
    analisis = analisis_textos.get(g[0], "")
    add_graph_to_story(g[0], g[1], g[2], analisis, story, styles) 
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("Reporte PDF generado: Reporte_Cuantitativo_de_Portafolio.pdf")

