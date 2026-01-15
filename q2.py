import json
import operator
import requests
import streamlit as st
from typing import List, Dict, Any


OPS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}

def evaluate_condition(facts, cond):
    field, op, value = cond
    return OPS[op](facts[field], value)

def rule_matches(facts, rule):
    return all(evaluate_condition(facts, c) for c in rule["conditions"])

def run_rules(facts, rules):
    fired = [r for r in rules if rule_matches(facts, r)]
    if not fired:
        return {"ac_mode": "OFF", "fan_speed": "LOW", "setpoint": None, "reason": "No rule matched"}, []

    fired_sorted = sorted(fired, key=lambda r: r["priority"], reverse=True)
    return fired_sorted[0]["action"], fired_sorted

RULES = json.loads('json_q2.json')


st.set_page_config(page_title="Smart Home AC Controller", layout="wide")
st.title("Rule-Based Smart Home Air-Conditioner")

with st.sidebar:
    st.header("Home Conditions")
    temperature = st.number_input("Temperature (°C)", value=28.0)
    humidity = st.number_input("Humidity (%)", value=65.0)
    occupancy = st.selectbox("Occupancy", ["OCCUPIED", "EMPTY"])
    time_of_day = st.selectbox("Time of day", ["MORNING", "AFTERNOON", "EVENING", "NIGHT"])
    windows_open = st.checkbox("Windows open")

    st.divider()
    st.header("Rules Source")
    st.caption("Rules loaded directly from GitHub JSON file")
    st.code(json.dumps(RULES, indent=2), language="json")

    run = st.button("Evaluate", type="primary")

facts = {
    "temperature": float(temperature),
    "humidity": float(humidity),
    "occupancy": occupancy,
    "time_of_day": time_of_day,
    "windows_open": windows_open,
}

st.subheader("Current Home State")
st.json(facts)

st.divider()

if run:
    action, fired = run_rules(facts, RULES)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("AC Decision")
        st.success(f"""
AC Mode: **{action['ac_mode']}**  
Fan Speed: **{action['fan_speed']}**  
Setpoint: **{action['setpoint']} °C**  
Reason: **{action['reason']}**
""")

    with col2:
        st.subheader("Matched Rules")
        if not fired:
            st.info("No rules matched.")
        else:
            for r in fired:
                st.write(f"{r['name']} (priority {r['priority']})")
else:
    st.info("Set home conditions and click **Evaluate**.")
