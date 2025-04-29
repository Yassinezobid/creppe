import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Configuration de la page
st.set_page_config(page_title="Q-Learning Visualisation", layout="wide")

st.title("Q-Learning - Grille Dynamique")

# --- Sidebar pour les paramètres fixes ---
with st.sidebar:
    st.header("Paramètres Environnement")
    n_rows = st.number_input("Nombre de
