import streamlit as st
import pandas as pd
import numpy as np
import folium
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
from scipy.signal import butter, filtfilt
import scipy.signal as signal
from scipy.fft import fft, fftfreq
from math import radians, sin, cos, sqrt, asin

urlacc = "https://raw.githubusercontent.com/Moti555/fymalopputehtava/refs/heads/main/Linear%20Accelerometer.csv"
df = pd.read_csv(urlacc)


def butter_lowpass_filter(data, cutoff, fs, nyq, order):
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    y = filtfilt(b, a, data)
    return y


T = df["Time (s)"][len(df["Time (s)"]) - 1] - df["Time (s)"][0] 
n = len(df["Time (s)"]) 
fs = n / T  
nyq = fs / 2  
cutoff = 1 / (0.2)  
order = 4  

df["filter_a_x"] = butter_lowpass_filter(df["X (m/s^2)"], cutoff, fs, nyq, order)


acc = df["filter_a_x"]
peaks, _ = signal.find_peaks(acc, height=0)
valleys, _ = signal.find_peaks(-acc, height=0)
steps = len(peaks)


acc_fft = fft(acc)
frequencies = fftfreq(len(acc), 1/fs)
magnitude = np.abs(acc_fft)
dominant_frequency = frequencies[np.argmax(magnitude[1:]) + 1]
steps_fft = int(dominant_frequency * T)


R = 6371  
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return c * R * 1000  
urlloc = "https://raw.githubusercontent.com/Moti555/fymalopputehtava/refs/heads/main/Location.csv"
df_loc = pd.read_csv(urlloc)
df_loc["dist"] = np.zeros(len(df_loc))
df_loc["time_diff"] = np.zeros(len(df_loc))

for i in range(len(df_loc) - 1):
    df_loc.loc[i, "dist"] = haversine(df_loc.loc[i, "Longitude (°)"], df_loc.loc[i, "Latitude (°)"],
    df_loc.loc[i+1, "Longitude (°)"], df_loc.loc[i+1, "Latitude (°)"])
    df_loc.loc[i, "time_diff"] = df_loc.loc[i+1, "Time (s)"] - df_loc.loc[i, "Time (s)"]

df_loc["velocity"] = df_loc["dist"] / df_loc["time_diff"]
df_loc["velocity"] = df_loc["velocity"].fillna(0)

df_loc["tot_dist"] = np.cumsum(df_loc["dist"])
total_distance = df_loc["tot_dist"].iloc[-2]
total_time = df_loc["Time (s)"].iloc[-1] - df_loc["Time (s)"].iloc[0]
average_speed = total_distance / total_time
average_speed_kmh = average_speed * 3.6

step_count = 779  
step_length = total_distance / step_count if step_count > 0 else 0

start_lat = df_loc["Latitude (°)"].mean()
start_long = df_loc["Longitude (°)"].mean()
my_map = folium.Map(location=[start_lat, start_long], zoom_start=14)
folium.PolyLine(df_loc[["Latitude (°)", "Longitude (°)"]], color="red", weight=2.5, opacity=1).add_to(my_map)

fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(df["Time (s)"], df["filter_a_x"], label="Filtered a_x")
ax1.set_title("Suodatettu kiihtyvyys X-akselilla")
ax1.set_xlabel("Aika (s)")
ax1.set_ylabel("Suodatettu_x (m/s²)")
ax1.legend()


N = len(acc)
dt = df["Time (s)"][1] - df["Time (s)"][0]
fourier = np.fft.fft(acc, N)
psd = (fourier * np.conj(fourier)).real / N
freq = 1 / (dt * N) * np.arange(N)
L = np.arange(1, np.floor(N / 2).astype("int"))
PSD = np.array([freq[L], psd[L].real])
freq_limit = 30
mask = PSD[0, :] <= freq_limit
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(PSD[0, mask], PSD[1, mask])
ax2.set_title("Tehospektri kiihtyvyysdatan X-komponentille")
ax2.set_xlabel("Taajuus (Hz)")
ax2.set_ylabel("Teho")


st.title("Fysiikan Loppuprojekti")

st.write(f"Askelmäära laskettuna suodatuksen avulla: {steps} askelta")
st.write(f"Askelmäära laskettuna Fourier-analyysin avulla: {steps_fft} askelta")
st.write(f"Keskinopeus: {average_speed:.2f} m/s")
st.write(f"Kokonaismatka: {total_distance/1000:.2f} km")
st.write(f"Askelpituus on: {step_length*100:.2f} cm")
 
st.title("Suodatettu kiihtyvyysdata ")
st.pyplot(fig1)

st.title("Tehospektri")
st.pyplot(fig2)

st.title("Karttakuva")
st_folium(my_map, width=700, height=500)