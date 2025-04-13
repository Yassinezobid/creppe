
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Crêp'Time - Simulateur Simple", layout="wide")
st.title("🥞 Crêp'Time - Simulateur de Rentabilité")

# === Produits : prix de vente et coût MP ===
st.sidebar.header("🧾 Produits & Marges")

prix_crepe = st.sidebar.number_input("Prix de vente crêpe (MAD)", value=30)
cout_crepe = st.sidebar.number_input("Coût MP crêpe (MAD)", value=10)

prix_jus = st.sidebar.number_input("Prix de vente jus (MAD)", value=20)
cout_jus = st.sidebar.number_input("Coût MP jus (MAD)", value=7)

prix_cafe = st.sidebar.number_input("Prix de vente café (MAD)", value=12)
cout_cafe = st.sidebar.number_input("Coût MP café (MAD)", value=3)

prix_glace = st.sidebar.number_input("Prix vente glace (MAD)", value=15)
cout_glace = st.sidebar.number_input("Coût MP glace (MAD)", value=5)

# Panier moyen net par client (fixé ou modifiable selon besoin)
marge_crepe = prix_crepe - cout_crepe
marge_jus = prix_jus - cout_jus
marge_cafe = prix_cafe - cout_cafe
marge_glace = prix_glace - cout_glace

# Panier moyen estimé selon mix habituel
panier_moyen = marge_crepe + marge_jus + marge_cafe + marge_glace

# === Paramètres généraux ===
st.sidebar.header("⚙️ Paramètres généraux")
clients_min = st.sidebar.slider("Clients/jour (min)", 5, 100, 15)
clients_max = st.sidebar.slider("Clients/jour (max)", 100, 200, 80)
pas = st.sidebar.slider("Pas variation", 1, 10, 5)
jours_mois = st.sidebar.slider("Jours d'activité par mois", 20, 31, 30)
associes = st.sidebar.number_input("Nombre d'associés", value=6)
impot_taux = st.sidebar.slider("Taux impôt (%)", 0, 50, 20) / 100

# === Charges mensuelles ===
st.sidebar.header("📦 Charges Mensuelles")
charges_fixes = st.sidebar.number_input("Total charges fixes (MAD)", value=11500)

# === Simulation ===
clients_range = list(range(clients_min, clients_max + 1, pas))
data = []

for clients in clients_range:
    revenu_brut = clients * panier_moyen * jours_mois
    benefice_avant_impot = revenu_brut - charges_fixes
    impot = max(0, benefice_avant_impot * impot_taux)
    profit_net = benefice_avant_impot - impot
    part_associe = profit_net / associes
    data.append([
        clients, panier_moyen, revenu_brut, benefice_avant_impot,
        impot, profit_net, part_associe
    ])

df = pd.DataFrame(data, columns=[
    "Clients/Jour", "Panier Moyen Net", "Revenu Brut",
    "Bénéfice Avant Impôt", "Impôt", "Profit Net", "Part par Associé"
])

# === Affichage ===
st.subheader("📊 Résultats de Simulation")
st.dataframe(df.style.format("{:,.0f}"))

st.subheader("📈 Graphique : Profit Net & Part Associé")
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(df["Clients/Jour"], df["Profit Net"], color='orange', label="Profit Net")
ax.plot(df["Clients/Jour"], df["Part par Associé"], marker='o', color='green', label="Part par Associé")
ax.set_title("Profit Net mensuel selon la fréquentation")
ax.set_xlabel("Clients par jour")
ax.set_ylabel("MAD")
ax.grid(True)
ax.legend()
st.pyplot(fig)
