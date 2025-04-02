import streamlit as st
from itertools import combinations
from typing import List, Tuple
from math import comb
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Parlay Optimizer", layout="centered")

# Define a bet as a tuple of (name, decimal_odds, system_win_rate)
Bet = Tuple[str, float, float]

def calculate_parlay_ev(parlay: List[Bet]) -> Tuple[str, float, float, float, float]:
    names = [bet[0] for bet in parlay]
    total_odds = 1.0
    win_prob = 1.0
    for _, odds, win_rate in parlay:
        total_odds *= odds
        win_prob *= win_rate
    ev = (total_odds - 1) * win_prob - (1 - win_prob)
    confidence_score = ev * win_prob
    return (
        ' + '.join(names),
        round(total_odds, 2),
        round(win_prob * 100, 2),
        round(ev, 3),
        round(confidence_score, 3)
    )

def kelly_stake(prob: float, odds: float) -> float:
    b = odds - 1
    kelly = (prob * b - (1 - prob)) / b
    return max(0, round(kelly * 100, 2))

def simulate_parlay_plus_straights(parlay: List[Bet], straights: List[Bet]) -> List[Tuple[int, float, float]]:
    from itertools import combinations
    parlay_odds = np.prod([b[1] for b in parlay])
    parlay_prob = np.prod([b[2] for b in parlay])
    n = len(straights)
    outcomes = []
    for wins in range(n + 1):
        straight_combos = combinations(straights, wins)
        results_with_win = []
        results_with_loss = []
        for win_combo in straight_combos:
            win_set = set(win_combo)
            profit = 0
            for bet in straights:
                if bet in win_set:
                    profit += bet[1] - 1
                else:
                    profit -= 1
            profit_with_parlay_win = profit + (parlay_odds - 1)
            profit_with_parlay_loss = profit - 1
            results_with_win.append(profit_with_parlay_win)
            results_with_loss.append(profit_with_parlay_loss)
        avg_win = round(np.mean(results_with_win), 2) if results_with_win else 0
        avg_loss = round(np.mean(results_with_loss), 2) if results_with_loss else 0
        outcomes.append((wins, avg_win, avg_loss))
    return outcomes

def simulate_profit_outcomes(bets: List[Bet]) -> Tuple[list, list]:
    n = len(bets)
    win_ranges = []
    net_ranges = []
    for wins in range(n + 1):
        all_win_combos = combinations(bets, wins)
        profits = []
        for win_combo in all_win_combos:
            win_set = set(win_combo)
            net = 0
            for bet in bets:
                if bet in win_set:
                    net += bet[1] - 1
                else:
                    net -= 1
            profits.append(net)
        avg_net = round(sum(profits) / len(profits), 2)
        win_ranges.append(wins)
        net_ranges.append(avg_net)
    return win_ranges, net_ranges

def generate_parlay_combinations(bets: List[Bet], size: int) -> List[Tuple[str, float, float, float, float]]:
    combos = combinations(bets, size)
    results = [calculate_parlay_ev(list(combo)) for combo in combos]
    return sorted(results, key=lambda x: x[3], reverse=True)

# Streamlit App UI
st.title("📊 Parlay Optimizer Tool")

st.markdown("Enter your bets below with decimal odds and system win probability (as a %). You can also select bets to simulate a parlay.")

bet_names = st.text_area("Bet Names (one per line)", value=st.session_state.get("bet_names", ""), key="bet_names")
bet_odds = st.text_area("Decimal Odds (one per line, same order)", value=st.session_state.get("bet_odds", ""), key="bet_odds")
bet_probs = st.text_area("System Win Rate % (one per line, same order)", value=st.session_state.get("bet_probs", ""), key="bet_probs")
recent_win_inputs = st.text_area("Recent Wins (out of 10, one per line, same order — e.g. `7` or `7-10`)", value=st.session_state.get("recent_wins", ""), key="recent_wins")
bankroll = st.number_input("Enter Bankroll", min_value=1, value=100)
submit_button = st.button("📊 Analyze Bets")
clear_button = st.button("🧹 Clear All Inputs")

if clear_button:
    for key in ["bet_names", "bet_odds", "bet_probs", "recent_wins"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

elif submit_button and bet_names and bet_odds and bet_probs and recent_win_inputs:
    try:
        names = bet_names.strip().split("\n")
        odds = list(map(float, bet_odds.strip().split("\n")))
        raw_probs = [float(p) / 100 for p in bet_probs.strip().split("\n")]

        recent_win_data = []
        for entry in recent_win_inputs.strip().split("\n"):
            if '-' in entry:
                try:
                    w, l = map(float, entry.split('-'))
                    t = w + l
                except:
                    w, t = 0, 10
            else:
                w, t = float(entry), 10
            recent_win_data.append((w, t))

        if len(names) != len(odds) or len(names) != len(raw_probs) or len(names) != len(recent_win_data):
            st.error("Each input section must have the same number of lines (including recent wins).")
            raise ValueError("Input length mismatch")

        # Apply Bayesian smoothing
        alpha, beta = 2, 2
        probs = [((alpha + w) / (alpha + beta + t)) for w, t in recent_win_data]

        bets = list(zip(names, odds, probs))

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

    else:
        st.subheader("🎯 Select Bets to Form a Parlay")
        selected_parlay = st.multiselect("Choose bets for your custom parlay:", options=names)
        selected_bets = [b for b in bets if b[0] in selected_parlay]
        remaining_bets = [b for b in bets if b[0] not in selected_parlay]

        if selected_bets:
            parlay_name, parlay_odds, parlay_win_pct, parlay_ev, parlay_conf = calculate_parlay_ev(selected_bets)
            kelly = kelly_stake(parlay_win_pct / 100, parlay_odds)
            stake_amt = round((kelly / 100) * bankroll, 2)
            st.markdown(f"**Custom Parlay:** `{parlay_name}`  \
            📈 Odds: `{parlay_odds}` | 🎯 Win %: `{parlay_win_pct}%` | 💰 EV: `{parlay_ev}u` | 🔐 Confidence: `{parlay_conf}` | 🧮 Kelly: `{kelly}%` → **{stake_amt} units**")

        st.subheader("💰 Kelly Staking Suggestions")
        for name, odds, prob in bets:
            kelly = kelly_stake(prob, odds)
            stake_amt = round((kelly / 100) * bankroll, 2)
            st.markdown(f"**{name}**  \
            📈 Odds: `{odds}` | 🎯 Win %: `{round(prob * 100, 2)}%` | 🧮 Kelly: `{kelly}%` → **{stake_amt} units**")

        st.subheader("📉 Break-Even Simulation")
        wins, net = simulate_profit_outcomes(bets)
        for w, n in zip(wins, net):
            st.markdown(f"**{w} Wins** → Avg Net Result: `{n}u`")

        if selected_bets and remaining_bets:
            st.subheader("⚖️ Parlay + Straight Combo Break-Even Analysis")
            combo_results = simulate_parlay_plus_straights(selected_bets, remaining_bets)

            win_counts = []
            avg_with_parlay_win = []
            avg_with_parlay_loss = []

            for w, net_win, net_loss in combo_results:
                st.markdown(f"**Straight Wins: {w}**  \
                ✅ If Parlay Wins → Avg Net: `{net_win}u`  \
                ❌ If Parlay Loses → Avg Net: `{net_loss}u`")
                win_counts.append(w)
                avg_with_parlay_win.append(net_win)
                avg_with_parlay_loss.append(net_loss)

            import plotly.graph_objects as go
            fig_plotly = go.Figure()
            fig_plotly.add_trace(go.Scatter(x=win_counts, y=avg_with_parlay_win, mode='lines+markers', name='Parlay Wins'))
            fig_plotly.add_trace(go.Scatter(x=win_counts, y=avg_with_parlay_loss, mode='lines+markers', name='Parlay Loses'))
            fig_plotly.add_hline(y=0, line=dict(color='gray', dash='dash'))
            fig_plotly.update_layout(title='Net Outcome with vs. without Parlay Hit', xaxis_title='Number of Straight Wins', yaxis_title='Net Profit (Units)')
            st.plotly_chart(fig_plotly)

            st.markdown("**📈 Scenario Probabilities:**")
            parlay_prob = np.prod([b[2] for b in selected_bets])
            st.markdown(f"🔢 Parlay Win Probability: `{round(parlay_prob * 100, 2)}%`")
            st.markdown(f"🔢 Parlay Loss Probability: `{round((1 - parlay_prob) * 100, 2)}%`")
            weighted_expected_value = round(parlay_prob * np.mean(avg_with_parlay_win) + (1 - parlay_prob) * np.mean(avg_with_parlay_loss), 2)
            st.markdown(f"📌 Probability-Weighted Avg Profit: `{weighted_expected_value}u`")

        st.subheader("🏆 Top 3 Straight Bets by EV")
        straight_evs = [((b[0], b[1], b[2], round((b[1] - 1) * b[2] - (1 - b[2]), 3), round(((b[1] - 1) * b[2] - (1 - b[2])) * b[2], 3))) for b in bets]
        top_straights = sorted(straight_evs, key=lambda x: x[3], reverse=True)[:3]
        for i, (name, odds_val, prob, ev_val, conf) in enumerate(top_straights, 1):
            kelly = kelly_stake(prob, odds_val)
            st.markdown(f"**Top #{i}:** `{name}`  \
            📈 Odds: `{odds_val}` | 🎯 Win %: `{round(prob * 100, 2)}%` | 💰 EV: `{ev_val}u` | 🔐 Confidence: `{conf}` | 🧮 Kelly: `{kelly}%`) ")

        st.subheader("🔍 Parlay Explorer (Top EV Combos)")
        max_parlay_size = min(4, len(bets))
        parlay_size = st.slider("Choose parlay size to explore best combos:", 2, max_parlay_size, 2)
        top_parlays = generate_parlay_combinations(bets, parlay_size)[:5]
        for name, odds_val, win_pct, ev_val, conf in top_parlays:
            st.markdown(f"🧪 **{name}**  \
            📈 Odds: `{odds_val}` | 🎯 Win %: `{win_pct}%` | 💰 EV: `{ev_val}u` | 🔐 Confidence: `{conf}`")

        st.subheader("📈 Multi-Parlay Impact Simulation")
        multi_choices = [p[0] for p in top_parlays]  # Auto-select all top parlays by default
        for parlay_name in multi_choices:
            parlay_legs = parlay_name.split(" + ")
            parlay_bets = [b for b in bets if b[0] in parlay_legs]
            leftover = [b for b in bets if b not in parlay_bets]
            st.markdown(f"### 🔁 `{parlay_name}`")
            multi_result = simulate_parlay_plus_straights(parlay_bets, leftover)
            for w, win_net, lose_net in multi_result:
                st.markdown(f"**Straight Wins: {w}**  \
                ✅ Parlay Wins → `{win_net}u`  \
                ❌ Parlay Loses → `{lose_net}u`")
            prob = np.prod([b[2] for b in parlay_bets])
            weighted_multi = round(prob * np.mean([r[1] for r in multi_result]) + (1 - prob) * np.mean([r[2] for r in multi_result]), 2)
            st.markdown(f"📌 Weighted Avg Profit: `{weighted_multi}u` (P_win = {round(prob*100, 2)}%)")

