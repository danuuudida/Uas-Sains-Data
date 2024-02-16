import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df_day = pd.read_csv('day.csv')
df_hour = pd.read_csv('hour.csv')

# Set judul dashboard
st.title("Simple Streamlit Dashboard")

# Tampilkan dataset day.csv
st.subheader("Dataset Day")
st.dataframe(df_day)

# Tampilkan dataset hour.csv
st.subheader("Dataset Hour")
st.dataframe(df_hour)

# Visualisasi sederhana
st.subheader("Visualisasi Data")
# Histogram jumlah peminjaman sepeda per hari
fig, ax = plt.subplots()
sns.histplot(df_day['cnt'], bins=20, kde=True)
ax.set(xlabel='Jumlah Peminjaman Sepeda', ylabel='Frekuensi', title='Histogram Peminjaman Sepeda per Hari')
st.pyplot(fig)

# Visualisasi jumlah peminjaman sepeda per jam
fig, ax = plt.subplots()
sns.lineplot(data=df_hour, x='hr', y='cnt', ci=None)
ax.set(xlabel='Jam', ylabel='Jumlah Peminjaman Sepeda', title='Grafik Jumlah Peminjaman Sepeda per Jam')
st.pyplot(fig)
