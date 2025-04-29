import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  

# Configuration de la page
st.set_page_config(page_title="Simulateur Q-Learning", layout="wide")

st.title("🧠 Simulateur d'Apprentissage par Q-Learning")

# --- Barre latérale : paramètres de l'environnement ---
with st.sidebar:
    st.header("🔧 Paramètres de l'environnement")
    
    n_rows = st.number_input("Nombre de lignes de la grille", min_value=2, max_value=10, value=3)
    n_cols = st.number_input("Nombre de colonnes de la grille", min_value=2, max_value=10, value=3)

    st.markdown("---")
    st.subheader("État Initial et Final")
    etat_initial = (
        st.number_input("Ligne état initial (1-indexé)", 1, n_rows, 1) - 1,
        st.number_input("Colonne état initial (1-indexé)", 1, n_cols, 1) - 1
    )

    etat_final = (
        st.number_input("Ligne état final (1-indexé)", 1, n_rows, n_rows) - 1,
        st.number_input("Colonne état final (1-indexé)", 1, n_cols, n_cols) - 1
    )

    recompense_normale = st.number_input("Récompense pour cases normales", value=-0.01)
    recompense_finale = st.number_input("Récompense pour état final", value=1.0)

    st.markdown("---")
    st.header("🔧 Paramètres Q-Learning")
    alpha = st.slider("Alpha (Taux d'apprentissage)", 0.0, 1.0, 0.5, 0.01)
    gamma = st.slider("Gamma (Facteur d'actualisation)", 0.0, 1.0, 0.9, 0.01)
    epsilon = st.slider("Epsilon (Exploration)", 0.0, 1.0, 0.1, 0.01)

    st.markdown("---")
    st.header("🔧 Critères d'arrêt")
    max_steps = st.number_input("Max étapes par épisode", 1, 100, 15)
    seuil_convergence = st.number_input("Seuil de convergence", 0.0001, 1.0, 0.01, step=0.001)
    episodes_sans_changement = st.number_input("Épisodes consécutifs stables", 1, 20, 2)

# --- Slider pour nombre d'épisodes ---
st.header("🎯 Nombre d'épisodes d'apprentissage")
n_episodes = st.slider("Choisissez le nombre d'épisodes", 10, 5000, 1000, step=10)

# --- Fonction d'apprentissage Q-Learning ---
def q_learning(n_episodes):
    Q = np.zeros((n_rows, n_cols, 4))
    actions = ['↑', '↓', '←', '→']

    def choisir_action(s):
        if np.random.uniform(0, 1) < epsilon:
            return np.random.randint(4)
        else:
            return np.argmax(Q[s[0], s[1]])

    def faire_etape(s, a):
        i, j = s
        if a == 0 and i > 0: i -= 1
        if a == 1 and i < n_rows - 1: i += 1
        if a == 2 and j > 0: j -= 1
        if a == 3 and j < n_cols - 1: j += 1
        s2 = (i, j)
        r = recompense_finale if s2 == etat_final else recompense_normale
        done = s2 == etat_final
        return s2, r, done

    recompenses = []
    stable_count = 0

    for ep in range(n_episodes):
        s = etat_initial
        total_r = 0
        max_change = 0

        for _ in range(max_steps):
            a = choisir_action(s)
            s2, r, done = faire_etape(s, a)
            old_q = Q[s[0], s[1], a]
            target = r + gamma * np.max(Q[s2[0], s2[1]])
            Q[s[0], s[1], a] += alpha * (target - old_q)
            max_change = max(max_change, abs(Q[s[0], s[1], a] - old_q))
            s = s2
            total_r += r
            if done:
                break

        recompenses.append(total_r)

        if max_change < seuil_convergence:
            stable_count += 1
        else:
            stable_count = 0

        if stable_count >= episodes_sans_changement:
            break

    return Q, recompenses

# --- Calcul apprentissage ---
Q, recompenses = q_learning(n_episodes)

# --- Affichage des résultats ---
st.header("📊 Résultats de l'apprentissage")

# Affichage Q-Table
st.subheader("Q-Table (Valeurs maximales par état)")
q_values = np.round(np.max(Q, axis=2), 2)
q_df = pd.DataFrame(q_values, index=[f"Ligne {i+1}" for i in range(n_rows)], columns=[f"Colonne {j+1}" for j in range(n_cols)])

# Marquer l'état final en rouge
def highlight_goal(val, row, col):
    if (row, col) == etat_final:
        return 'background-color: lightcoral'
    return ''

styled_q = q_df.style.format(precision=2).apply(lambda x: [highlight_goal(v, i, j) for j, v in enumerate(x)], axis=1, result_type='expand')\
    .set_properties(**{'text-align': 'center', 'font-size': '20px'})

st.dataframe(styled_q, use_container_width=True, height=int(n_rows * 70))

# Politique déduite
st.subheader("Politique Déduite (Meilleure Action)")
actions = ['↑', '↓', '←', '→']
politique = np.empty((n_rows, n_cols), dtype=object)
for i in range(n_rows):
    for j in range(n_cols):
        politique[i, j] = actions[np.argmax(Q[i, j])]

politique_df = pd.DataFrame(politique, index=[f"Ligne {i+1}" for i in range(n_rows)], columns=[f"Colonne {j+1}" for j in range(n_cols)])
styled_politique = politique_df.style.set_properties(**{'text-align': 'center', 'font-size': '24px'})

st.dataframe(styled_politique, use_container_width=True, height=int(n_rows * 70))

# Courbe d'apprentissage
st.subheader("Évolution des Récompenses")
fig, ax = plt.subplots()
ax.plot(range(1, len(recompenses)+1), recompenses)
ax.set_xlabel("Épisode")
ax.set_ylabel("Récompense totale")
ax.grid(True)
st.pyplot(fig)
