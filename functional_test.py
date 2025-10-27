# functional_tests.py
import requests, json, sys

ENDPOINT = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5000/predict"

tests = {
    "fake1": "BREAKING: Scientists confirm Moon is made of cheese!",
    "fake2": "Celebrity clone replaces world leader, sources say!",
    "real1": "The University of Toronto announced new research funding today.",
    "real2": "The Bank of Canada held its policy interest rate steady this month.",
}

for name, text in tests.items():
    r = requests.post(ENDPOINT, json={"text": text}, timeout=10)
    print(name, r.status_code, r.json())
