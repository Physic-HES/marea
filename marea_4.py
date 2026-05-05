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
location = EarthLocation.from_geodetic(lat=lat * u.deg, lon=lon * u.deg, height=0 * u.m)

# Intervalo horario para el modelo
model_start = datetime(2026, 3, 27)
model_end = datetime(2026, 6, 2)

times = []
t = model_start
while t <= model_end:
    times.append(t)
    t += timedelta(hours=0.5)

times_arr = np.array(times)
t_astropy = Time(times)

gcrs = location.get_gcrs(t_astropy)
posicion_gcrs = gcrs.cartesian.xyz.to(u.km).value.T
moon = get_body("moon", t_astropy)
posicion_moon = moon.cartesian.xyz.to(u.km).value.T
sol = get_sun(t_astropy)
posicion_sol = sol.cartesian.xyz.to(u.km).value.T

def unit_vector(position):
    norm = np.sqrt(position[:,0]**2 + position[:,1]**2 + position[:,2]**2)
    return np.c_[position[:,0] / norm, position[:,1] / norm, position[:,2] / norm]

# cos theta para la luna
moon_vec = unit_vector(posicion_moon)
location_vec = unit_vector(posicion_gcrs)
sol_vec = unit_vector(posicion_sol)

dot_product_moon = np.sum(location_vec * moon_vec, axis=1)
dot_product_sol = np.sum(location_vec * sol_vec, axis=1)

# Constantes para el modelo
G = 6.67430e-20  # km^3 kg^-1 s^-2
M_moon = 7.342e22  # kg
M_sol = 1.989e30  # kg

# distancias de la locacion y de la luna al centro de la Tierra
r_location = np.linalg.norm(posicion_gcrs, axis=1)
r_moon = np.linalg.norm(posicion_moon, axis=1)
r_diff_moon = np.linalg.norm(posicion_gcrs - posicion_moon, axis=1)
r_sol = np.linalg.norm(posicion_sol, axis=1)
r_diff_sol = np.linalg.norm(posicion_gcrs - posicion_sol, axis=1)

# Aceleración de marea ( Modelo de marea de segundo orden, con término de Legendre P2 )
a_moon = 1e3 * G * M_moon * (r_location / r_moon**3)*(3 * dot_product_moon**2 - 1)
a_sol = 1e3 * G * M_sol * (r_location / r_sol**3)*(3 * dot_product_sol**2 - 1)
A_total = a_moon + a_sol

# Plotear modelo de aceleración de marea lunar
fig = plt.figure(figsize=(10, 6))
ax = fig.subplots(3, 1)
ax[0].plot(times_arr, a_moon, label='Aceleración de marea lunar')
ax[1].plot(times_arr, A_total, label='Aceleración total de marea')
ax[2].plot(times_arr, a_sol, label='Aceleración de marea solar')
ax[2].set_xlabel('Fecha')
ax[0].set_ylabel('Aceleración (m/s²)')
ax[1].set_ylabel('Aceleración (m/s²)')
ax[2].set_ylabel('Aceleración (m/s²)')
ax[0].set_title('Señal de aceleración de marea en San Antonio Oeste (marzo-junio 2026)')
ax[0].grid()
ax[1].grid()
ax[2].grid()
ax[0].legend()
ax[1].legend()
ax[2].legend()
plt.tight_layout()

# importar prediccion de marea para san antonio  y concatenar abril y mayo
marea_abril = pd.read_csv('mareas_san_antonio_abril_2026.csv')
marea_mayo = pd.read_csv('mareas_san_antonio_mayo_2026.csv')
marea_abril['fecha'] = pd.to_datetime(marea_abril['fecha'])
marea_mayo['fecha'] = pd.to_datetime(marea_mayo['fecha'])
marea_san_antonio = pd.concat([marea_abril, marea_mayo], ignore_index=True)

# importar datos de marea real para puerto belgrano
marea_real = pd.read_csv('mareas_pto_belgrano_3.csv', header=20)
print(marea_real.head())
marea_real['timeend'] = pd.to_datetime(marea_real['timeend'])
# convertir a float la columna de valor, manejando errores
marea_real['valor'] = pd.to_numeric(marea_real['valor'], errors='coerce')
# eliminar filas sin datos reales (fila final "End of data" con NaN)
marea_real = marea_real.dropna(subset=['valor'])

# plot de A_total y de las predicciones de marea para san antonio abril y mayo, y de la marea real para puerto belgrano
fig = plt.figure(figsize=(10, 6))
ax = fig.subplots(2, 1)
ax[0].plot(times_arr, A_total*1e6, label='Señal reescalada de aceleración total de marea')
# plot de marea
ax[0].plot(marea_san_antonio['fecha'], marea_san_antonio['altura'], label='Predicción de marea en San Antonio')
ax[0].plot(marea_real['timeend'], marea_real['valor'], label='Marea real en Puerto Belgrano')
ax[0].set_xlabel('Fecha')
ax[0].set_ylabel('Altura de marea (m)')
ax[0].set_title('Predicciones y registros de marea en San Antonio Oeste y Puerto Belgrano (marzo-junio 2026)')
ax[0].grid()
ax[0].legend()
ax[0].set_xlim(np.min(marea_san_antonio['fecha']), np.max(marea_san_antonio['fecha']))
plt.tight_layout()

# Plotear transformadas de Fourier
from scipy.fft import rfft, rfftfreq
# Transformada de Fourier de la aceleración total de marea
N = len(A_total)
T = 0.5 * 3600  # intervalo de tiempo en segundos (0.5 horas)
yf = rfft((A_total - np.mean(A_total))*1e6)  # restar la media y escalar para mejor visualización
xf = rfftfreq(N, T)

# Transformada de Fourier de la predicción de marea para San Antonio abril
N_san_antonio = len(marea_san_antonio)
T_san_antonio = 6 * 3600  # intervalo de tiempo en segundos
yf_san_antonio = rfft(marea_san_antonio['altura'] - np.mean(marea_san_antonio['altura']))
xf_san_antonio = rfftfreq(N_san_antonio, T_san_antonio)
# Transformada de Fourier de la marea real en Puerto Belgrano
N_real = len(marea_real)
T_real = 1 * 3600  # intervalo de tiempo en segundos
real_valor = marea_real['valor'] - np.nanmean(marea_real['valor'])
yf_real = rfft(real_valor)
xf_real = rfftfreq(N_real, T_real)

ax[1].loglog(xf, np.abs(yf), label='Señal de aceleración total de marea')
ax[1].loglog(xf_san_antonio, np.abs(yf_san_antonio), label='Predicción de marea en San Antonio')
ax[1].loglog(xf_real, np.abs(yf_real), label='Marea real en Puerto Belgrano')
ax[1].set_xlabel('Frecuencia (Hz)')
ax[1].set_ylabel('Amplitud')
ax[1].set_title('Transformada de Fourier de las señales de marea')
ax[1].grid()

# Frecuencias Teóricas
f_D = 1/(24 * 3600)  # Convertir a Hz
f_S = f_D * 1.92
f_quincenal = 1/(14.77 * 24 * 3600)
f_anual = 1/(365.25 * 24 * 3600)

for f_val, label1, col, label2 in zip([f_anual, f_quincenal, f_D, f_S], 
                             ['Anual', 'Quincenal', 'Diurno', 'Semidiurno'],
                             ['red', 'gray', 'green', 'purple'],
                             [f'1/f_sol (Anual ~365d)', f'1/f_quincenal (Quincenal ~14.77d)', f'1/f_D (Diurno ~24h)', f'1/f_S (Semidiurno ~12.5h)']):
    ax[1].axvline(f_val, color=col, linestyle='--',label=label2, alpha=0.6)
    ax[1].annotate(label1, xy=(f_val, 2e-2), rotation=90, color=col, fontsize=9)

plt.legend()
plt.xlim(0, 1e-4)  # Limitar el eje x para enfocarse en frecuencias relevantes
plt.ylim(1e-2, 1000)  # Limitar el eje y para mejor visualización
plt.tight_layout()
plt.show()