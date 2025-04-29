import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Q-Learning Visualisation", layout="wide")

st.title("Q-Learning - Grille et Convergence")

# --- Layout principal avec deux colonnes ---
col_param, col_affichage = st.columns([1, 2])

with col_param:
    st.header("Paramètres du simulateur")

    st.subheader("1. Environnement")
    n_rows = st.number_input("Nombre de lignes", min_value=2, max_value=10, value=3)
    n_cols = st.number_input("Nombre de colonnes", min_value=2, max_value=10, value=3)

    etat_initial = (
        st.number_input("Ligne de l'état initial (1-indexé)", 1, n_rows, 1) - 1,
        st.number_input("Colonne de l'état initial (1-indexé)", 1, n_cols, 1) - 1
    )

    etat_final = (
        st.number_input("Ligne de l'état final (1-indexé)", 1, n_rows, n_rows) - 1,
        st.number_input("Colonne de l'état final (1-indexé)", 1, n_cols, n_cols) - 1
    )

    recompense_normale = st.number_input("Récompense par défaut (cases normales)", value=-0.01)
    recompense_finale = st.number_input("Récompense de l'état final", value=1.0)

    st.subheader("2. Paramètres Q-Learning")
    alpha = st.slider("Alpha (taux d'apprentissage)", 0.0, 1.0, 0.5, 0.01)
    gamma = st.slider("Gamma (facteur de réduction)", 0.0, 1.0, 0.9, 0.01)
    epsilon = st.slider("Epsilon (exploration)", 0.0, 1.0, 0.1, 0.01)

    st.subheader("3. Critère de Convergence")
    n_episodes = st.number_input("Nombre max d'épisodes", 1, 10000, 1000)
    max_steps = st.number_input("Nombre max d'étapes par épisode", 1, 100, 15)
    seuil_convergence = st.number_input("Seuil de convergence", 0.0001, 1.0, 0.01, step=0.001, format="%.4f")
    episodes_sans_changement = st.number_input("Épisodes consécutifs sans changement", 1, 20, 2)

    start = st.button("Lancer l'apprentissage")

with col_affichage:
    def q_learning():
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
                st.success(f"Convergence atteinte à l'épisode {ep+1}")
                break

        return Q, recompenses

    if start:
        Q, recompenses = q_learning()

        st.subheader("Q-Table (Valeur maximale par case)")
        grille_q = np.round(np.max(Q, axis=2), 2)
        grille_df = pd.DataFrame(grille_q, index=[f"Ligne {i+1}" for i in range(n_rows)],
                                 columns=[f"Colonne {j+1}" for j in range(n_cols)])
        st.dataframe(
            grille_df.style
                .format(precision=2)
                .background_gradient(cmap="YlGn")
                .set_properties(**{'text-align': 'center', 'font-size': '20px'}),
            use_container_width=True,
            height=int(n_rows * 60)
        )

        st.subheader("Politique Déduite (Meilleure direction par case)")
        actions = ['↑', '↓', '←', '→']
        politique = np.empty((n_rows, n_cols), dtype=object)
        for i in range(n_rows):
            for j in range(n_cols):
                meilleure_action = np.argmax(Q[i, j])
                politique[i, j] = actions[meilleure_action]

        politique_df = pd.DataFrame(politique, index=[f"Ligne {i+1}" for i in range(n_rows)],
                                    columns=[f"Colonne {j+1}" for j in range(n_cols)])
        st.dataframe(
            politique_df.style
                .set_properties(**{'text-align': 'center', 'font-size': '24px'}),
            use_container_width=True,
            height=int(n_rows * 60)
        )

        st.subheader("Courbe d'Apprentissage")
        max_affiche = st.slider("Afficher les N premiers épisodes", 10, len(recompenses), len(recompenses), step=10)
        fig, ax = plt.subplots()
        ax.plot(range(1, max_affiche + 1), recompenses[:max_affiche])
        ax.set_xlabel("Épisode")
        ax.set_ylabel("Récompense Totale")
        ax.grid(True)
        st.pyplot(fig)
