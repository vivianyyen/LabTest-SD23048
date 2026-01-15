import json
from typing import List, Dict, Any, Tuple
import operator
import streamlit as st

# ----------------------------
# 1) Rule engine (UNCHANGED)
# ----------------------------
OPS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}

# ----------------------------
# 2) Default Smart Home AC Rules (use your JSON keys)
# ----------------------------
DEFAULT_RULES: List[Dict[str, Any]] = [
   {
    "name": "Windows open \u2192 turn AC off",
    "priority": 100,
    "conditions": [
      [
        "windows_open",
        "==",
        true
      ]
    ],
    "action": {
      "ac_mode": "OFF",
      "fan_speed": "LOW",
      "setpoint": null,
      "reason": "Windows are open"
    }
  },
  {
    "name": "No one home \u2192 eco mode",
    "priority": 90,
    "conditions": [
      [
        "occupancy",
        "==",
        "EMPTY"
      ],
      [
        "temperature",
        ">=",
        24
      ]
    ],
    "action": {
      "ac_mode": "ECO",
      "fan_speed": "LOW",
      "setpoint": 27,
      "reason": "Home empty; save energy"
    }
  },
  {
    "name": "Hot & humid (occupied) \u2192 cool strong",
    "priority": 80,
    "conditions": [
      [
        "occupancy",
        "==",
        "OCCUPIED"
      ],
      [
        "temperature",
        ">=",
        30
      ],
      [
        "humidity",
        ">=",
        70
      ]
    ],
    "action": {
      "ac_mode": "COOL",
      "fan_speed": "HIGH",
      "setpoint": 23,
      "reason": "Hot and humid"
    }
  },
  {
    "name": "Hot (occupied) \u2192 cool",
    "priority": 70,
    "conditions": [
      [
        "occupancy",
        "==",
        "OCCUPIED"
      ],
      [
        "temperature",
        ">=",
        28
      ]
    ],
    "action": {
      "ac_mode": "COOL",
      "fan_speed": "MEDIUM",
      "setpoint": 24,
      "reason": "Temperature high"
    }
  },
  {
    "name": "Slightly warm (occupied) \u2192 gentle cool",
    "priority": 60,
    "conditions": [
      [
        "occupancy",
        "==",
        "OCCUPIED"
      ],
      [
        "temperature",
        ">=",
        26
      ],
      [
        "temperature",
        "<",
        28
      ]
    ],
    "action": {
      "ac_mode": "COOL",
      "fan_speed": "LOW",
      "setpoint": 25,
      "reason": "Slightly warm"
    }
  },
  {
    "name": "Night (occupied) \u2192 sleep mode",
    "priority": 75,
    "conditions": [
      [
        "occupancy",
        "==",
        "OCCUPIED"
      ],
      [
        "time_of_day",
        "==",
        "NIGHT"
      ],
      [
        "temperature",
        ">=",
        26
      ]
    ],
    "action": {
      "ac_mode": "SLEEP",
      "fan_speed": "LOW",
      "setpoint": 26,
      "reason": "Night comfort"
    }
  },
  {
    "name": "Too cold \u2192 turn off",
    "priority": 85,
    "conditions": [
      [
        "temperature",
        "<=",
        22
      ]
    ],
    "action": {
      "ac_mode": "OFF",
      "fan_speed": "LOW",
      "setpoint": null,
      "reason": "Already cold"
    }
  }
]

# ----------------------------
# 3) Rule Engine Functions (UNCHANGED)
# ----------------------------
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

# ----------------------------
# 4) Streamlit UI (same template)
# ----------------------------
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
    st.header("Rules (JSON)")
    rules_text = st.text_area("Edit rules", json.dumps(DEFAULT_RULES, indent=2), height=300)

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

try:
    rules = json.loads(rules_text)
except:
    st.error("Invalid JSON. Using default rules.")
    rules = DEFAULT_RULES

st.divider()

if run:
    action, fired = run_rules(facts, rules)

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
    st.info("Set values and click **Evaluate**.")
