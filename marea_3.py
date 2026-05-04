import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.time import Time
from astropy.coordinates import EarthLocation, get_sun, get_body
import astropy.units as u
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
from scipy.stats import pearsonr

# =============================
# Configuración básica
# =============================
lat = -40.48
lon = -64.53
location = EarthLocation(lat=lat * u.deg, lon=lon * u.deg)

# Intervalo horario para el modelo
model_start = datetime(2026, 3, 27)
model_end = datetime(2026, 6, 2)

times = []
t = model_start
while t <= model_end:
    times.append(t)
    t += timedelta(hours=1)

times_arr = np.array(times)
t_astropy = Time(times)

# =============================
# Cálculo astronómico
# =============================
moon = get_body("moon", t_astropy, location)
sun = get_sun(t_astropy)


def unit_vector(coord):
    cart = coord.cartesian
    norm = np.sqrt(cart.x**2 + cart.y**2 + cart.z**2)
    return cart.x / norm, cart.y / norm, cart.z / norm

moon_vec = unit_vector(moon)
sun_vec = unit_vector(sun)

lat_rad = np.radians(lat)
lon_rad = np.radians(lon)

nx = np.cos(lat_rad) * np.cos(lon_rad)
ny = np.cos(lat_rad) * np.sin(lon_rad)
nz = np.sin(lat_rad)

cos_theta_moon = nx * moon_vec[0] + ny * moon_vec[1] + nz * moon_vec[2]
cos_theta_sun = nx * sun_vec[0] + ny * sun_vec[1] + nz * sun_vec[2]


def P2(x):
    return 0.5 * (3 * x**2 - 1)

r_moon = moon.distance.to(u.km).value
r_sun = sun.distance.to(u.km).value

V_moon = P2(cos_theta_moon) / r_moon**3
V_sun = P2(cos_theta_sun) / r_sun**3

h_total = 1e16 * (V_moon + V_sun)
h_moon = 1e16 * V_moon
h_diff = h_total - h_moon

# =============================
# Datos reales
# =============================
# San Antonio Oeste: abril + mayo
sa_april = pd.read_csv("mareas_san_antonio_abril_2026.csv", parse_dates=["fecha"])
sa_may = pd.read_csv("mareas_san_antonio_mayo_2026.csv", parse_dates=["fecha"])
df_sa = pd.concat([sa_april, sa_may], ignore_index=True)
df_sa = df_sa.sort_values("fecha").reset_index(drop=True)
df_sa = df_sa[df_sa["altura"] > 5].reset_index(drop=True)

# Puerto Belgrano 
df_bg_raw = pd.read_csv(
    "mareas_pto_belgrano_3.csv",
    comment="#",
    header=None,
    skip_blank_lines=True,
    names=["obs_id", "timestart", "timeend", "valor", "timeupdate"],
    usecols=["timeend", "valor"],
)
df_bg_raw["timeend"] = pd.to_datetime(df_bg_raw["timeend"], errors="coerce")
df_bg = df_bg_raw.rename(columns={"timeend": "fecha", "valor": "altura"})
df_bg = df_bg.dropna(subset=["fecha", "altura"]).reset_index(drop=True)
df_bg = df_bg[df_bg["altura"] > -10].sort_values("fecha").reset_index(drop=True)

print(f"San Antonio: {len(df_sa)} mediciones")
print(f"  Rango: {df_sa['fecha'].min()} a {df_sa['fecha'].max()}")
print(f"  Altura: {df_sa['altura'].min():.2f}m a {df_sa['altura'].max():.2f}m")
print()
print(f"Puerto Belgrano: {len(df_bg)} mediciones")
print(f"  Rango: {df_bg['fecha'].min()} a {df_bg['fecha'].max()}")
print(f"  Altura: {df_bg['altura'].min():.2f}m a {df_bg['altura'].max():.2f}m")

# =============================
# Funciones auxiliares
# =============================

def normalize_series(y):
    y = np.array(y, dtype=float)
    mask = ~np.isnan(y)
    if np.sum(mask) == 0:
        return np.zeros_like(y)
    y_valid = y[mask]
    if np.nanmax(y_valid) == np.nanmin(y_valid):
        return np.zeros_like(y)
    return (y - np.nanmin(y_valid)) / (np.nanmax(y_valid) - np.nanmin(y_valid))


def rescale_to_data(y_model, y_data):
    y_model = np.array(y_model, dtype=float)
    y_data = np.array(y_data, dtype=float)
    y_mask = ~np.isnan(y_model)
    if np.sum(y_mask) == 0:
        return np.zeros_like(y_model)
    y_model_valid = y_model[y_mask]
    y0 = np.nanpercentile(y_model_valid, 5)
    y1 = np.nanpercentile(y_model_valid, 95)
    if y1 == y0:
        return np.zeros_like(y_model)
    y_data_min = np.nanpercentile(y_data, 5)
    y_data_max = np.nanpercentile(y_data, 95)
    return ((y_model - y0) / (y1 - y0)) * (y_data_max - y_data_min) + y_data_min


def interpolate_to_grid(times_grid, data_times, data_values, method="cubic"):
    t_grid = np.array([(t - model_start).total_seconds() for t in times_grid])
    t_data = np.array([(t - model_start).total_seconds() for t in data_times])
    interp = interp1d(
        t_data,
        data_values,
        kind=method,
        bounds_error=False,
        fill_value=np.nan,
    )
    return interp(t_grid)

# =============================
# Gráfico 1: tres señales temporales
# =============================
split_date = datetime(2026, 4, 27)
past_mask = times_arr < split_date
future_mask = times_arr >= split_date

fig1, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

axs[0].plot(times_arr[past_mask], h_total[past_mask], label="Reconstrucción")
axs[0].plot(times_arr[future_mask], h_total[future_mask], "--", label="Predicción")
axs[0].axvline(split_date, linestyle=":", color="black", label="Separación 27/04")
axs[0].set_title("Marea total (Luna + Sol)")
axs[0].set_ylabel("Nivel relativo")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(times_arr[past_mask], h_moon[past_mask], label="Reconstrucción")
axs[1].plot(times_arr[future_mask], h_moon[future_mask], "--", label="Predicción")
axs[1].axvline(split_date, linestyle=":", color="black", label="Separación 27/04")
axs[1].set_title("Componente lunar")
axs[1].set_ylabel("Nivel relativo")
axs[1].legend()
axs[1].grid(True)

axs[2].plot(times_arr[past_mask], h_diff[past_mask], label="Reconstrucción")
axs[2].plot(times_arr[future_mask], h_diff[future_mask], "--", label="Predicción")
axs[2].axvline(split_date, linestyle=":", color="black", label="Separación 27/04")
axs[2].set_title("Contribución solar")
axs[2].set_ylabel("Nivel relativo")
axs[2].set_xlabel("Fecha")
axs[2].legend()
axs[2].grid(True)

fig1.tight_layout()

# =============================
# Gráfico 2: comparaciones temporales y espectrales
# =============================
# Interpolar el modelo a los tiempos de los datos
model_time_sa = np.array([(t - model_start).total_seconds() for t in df_sa["fecha"]])
model_time_bg = np.array([(t - model_start).total_seconds() for t in df_bg["fecha"]])
interp_model = interp1d(
    np.array([(t - model_start).total_seconds() for t in times_arr]),
    V_moon,
    kind="cubic",
    fill_value="extrapolate",
)
h_model_sa = interp_model(model_time_sa)
h_model_bg = interp_model(model_time_bg)

h_model_sa_rescaled = rescale_to_data(h_model_sa, df_sa["altura"])
h_model_bg_rescaled = rescale_to_data(h_model_bg, df_bg["altura"])


compare_start = datetime(2026, 4, 2)
compare_end = datetime(2026, 5, 31)
compare_mask = (times_arr >= compare_start) & (times_arr <= compare_end)
times_compare = times_arr[compare_mask]

_df_sa_hourly = interpolate_to_grid(times_compare, df_sa["fecha"], df_sa["altura"])
_df_bg_hourly = interpolate_to_grid(times_compare, df_bg["fecha"], df_bg["altura"])


sa_fft_values = _df_sa_hourly.copy()
if np.all(np.isnan(sa_fft_values)):
    sa_fft_values = np.zeros_like(sa_fft_values)
else:
    sa_fft_values[np.isnan(sa_fft_values)] = np.nanmean(sa_fft_values)

bg_fft_values = _df_bg_hourly.copy()
if np.all(np.isnan(bg_fft_values)):
    bg_fft_values = np.zeros_like(bg_fft_values)
else:
    bg_fft_values[np.isnan(bg_fft_values)] = np.nanmean(bg_fft_values)

model_compare = h_total[compare_mask]
v_total_for_fft = model_compare - np.mean(model_compare)

n = len(v_total_for_fft)
freqs = np.fft.rfftfreq(n, d=1.0)

mag_model = np.abs(np.fft.rfft(v_total_for_fft))
mag_sa = np.abs(np.fft.rfft(sa_fft_values - np.nanmean(sa_fft_values)))
mag_bg = np.abs(np.fft.rfft(bg_fft_values - np.nanmean(bg_fft_values)))

fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=False)

# --- Subplot 1: Comparación Temporal 
y_sa_norm = normalize_series(_df_sa_hourly)
y_bg_norm = normalize_series(_df_bg_hourly)
y_model_norm = normalize_series(model_compare)

ax1.plot(times_compare, y_sa_norm / 1.25 + 0.1, '.-', label="Predicción del SHN (San Antonio Oeste)", alpha=0.7, markersize=4)
ax1.plot(times_compare[~np.isnan(y_bg_norm)], y_bg_norm[~np.isnan(y_bg_norm)] / 1.25 + 0.1, '.-', label="Mareógrafo Pto. Belgrano (SHN)", alpha=0.7, markersize=4)
ax1.plot(times_compare, y_model_norm / 1.25 + 0.1, '-', label="Modelo (solo Luna)")
ax1.axvline(split_date, linestyle=":", color="black", label="Separación 27/04")
ax1.set_title("Comparación Temporal: Modelo vs Datos Reales")
ax1.set_ylabel("Altura de marea normalizada")
ax1.set_xlim(compare_start, compare_end)
ax1.set_ylim(0, 1)
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.loglog(freqs[1:], mag_sa[1:] / np.max(mag_sa[1:]), label="Espectro San Antonio (normalizado)")
ax2.loglog(freqs[1:], mag_bg[1:] / np.max(mag_bg[1:]), label="Espectro Puerto Belgrano (normalizado)")
ax2.loglog(freqs[1:], mag_model[1:] / np.max(mag_model[1:]), label="Espectro Modelo (normalizado)")

# Frecuencias Teóricas
f_D = 1/24
f_S = f_D * 1.92
f_quincenal = 1/(14.77 * 24)
f_anual = 1/(365.25 * 24)

for f_val, label1, col, label2 in zip([f_anual, f_quincenal, f_D, f_S], 
                             ['Anual', 'Quincenal', 'Diurno', 'Semidiurno'],
                             ['red', 'gray', 'green', 'purple'],
                             [f'1/f_sol (Anual ~365d)', f'1/f_quincenal (Quincenal ~14.77d)', f'1/f_D (Diurno ~24h)', f'1/f_S (Semidiurno ~12.5h)']):
    ax2.axvline(f_val, color=col, linestyle='--',label=label2, alpha=0.6)
    ax2.annotate(label1, xy=(f_val, 2e-4), rotation=90, color=col, fontsize=9)

ax2.set_title("Análisis espectral: datos reales vs modelo")
ax2.set_xlabel("Frecuencia [1/hora]")
ax2.set_ylabel("Magnitud normalizada")
ax2.set_xlim(freqs[1], 0.1)
ax2.set_ylim(1e-4, 2)
ax2.legend()
ax2.grid(True, which="both", ls="--", alpha=0.2)

fig2.tight_layout()

plt.show()
