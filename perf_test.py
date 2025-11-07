# perf_test.py  (AWS-friendly)
import csv, os, time, math, statistics, requests, sys
from datetime import datetime, timezone
import matplotlib.pyplot as plt

# --- config --- #
PREDICT_URL = sys.argv[1] if len(sys.argv) > 1 else "http://detect-fake-news-env.eba-seezhc2y.us-east-1.elasticbeanstalk.com/predict"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "perf_results_aws"
N_CALLS = 100  # number of calls per test case
os.makedirs(OUT_DIR, exist_ok=True)

TEST_CASES = {
    "fake1": "BREAKING: Scientists confirm Moon is made of cheese!",
    "fake2": "Celebrity clone replaces world leader, sources say!",
    "real1": "The University of Toronto announced new research funding today.",
    "real2": "The Bank of Canada held its policy interest rate steady this month.",
}

session = requests.Session()

def run_case(case_name, text):
    path = os.path.join(OUT_DIR, f"{case_name}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["test_case","iteration","start_iso","end_iso","latency_ms","status_code","label"])
        for i in range(1, N_CALLS + 1):
            start = time.perf_counter_ns()
            start_iso = datetime.now(timezone.utc).isoformat()
            try:
                resp = session.post(PREDICT_URL, json={"message": text}, timeout=15)
                end = time.perf_counter_ns()
                end_iso = datetime.now(timezone.utc).isoformat()
                latency_ms = (end - start) / 1_000_000.0
                try:
                    data = resp.json()
                except Exception:
                    data = {}
                w.writerow([case_name, i, start_iso, end_iso, f"{latency_ms:.3f}", resp.status_code, data.get("label")])
            except Exception:
                end = time.perf_counter_ns()
                end_iso = datetime.now(timezone.utc).isoformat()
                latency_ms = (end - start) / 1_000_000.0
                w.writerow([case_name, i, start_iso, end_iso, f"{latency_ms:.3f}", "ERROR", None])
    return path

def gather_latencies(csv_path):
    vals = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("status_code") == "200":
                vals.append(float(row["latency_ms"]))
    return vals

def main():
    # Warm up the env (don’t record)
    for _ in range(3):
        try:
            session.post(PREDICT_URL, json={"message": "warm up"}, timeout=10)
        except Exception:
            pass

    csv_paths = []
    for name, text in TEST_CASES.items():
        print(f"Running {name} …")
        csv_paths.append(run_case(name, text))

    # Summaries + plots
    data, labels = [], []
    summary_rows = [["test_case","count_200","avg_ms","p50_ms","p90_ms","p99_ms"]]

    for p in csv_paths:
        name = os.path.splitext(os.path.basename(p))[0]
        lat = gather_latencies(p)
        if lat:
            lat.sort()
            def pct(q):
                idx = max(0, min(len(lat)-1, int(math.ceil(q*len(lat))-1)))
                return lat[idx]
            avg = statistics.fmean(lat)
            summary_rows.append([name, len(lat), f"{avg:.2f}", f"{pct(0.50):.2f}", f"{pct(0.90):.2f}", f"{pct(0.99):.2f}"])
            data.append(lat); labels.append(name)
        else:
            summary_rows.append([name, 0, "", "", "", ""])

    with open(os.path.join(OUT_DIR, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(summary_rows)

    if data:
        # Linear boxplot
        plt.figure()
        plt.boxplot(data, tick_labels=labels, showmeans=True)
        plt.title("API Latency per Test Case (ms)")
        plt.ylabel("Latency (ms)")
        plt.savefig(os.path.join(OUT_DIR, "latency_boxplot.png"), bbox_inches="tight", dpi=150)
        plt.close()

        # Log-scale boxplot
        plt.figure()
        plt.boxplot(data, tick_labels=labels, showmeans=True)
        plt.yscale("log")
        plt.title("API Latency per Test Case (ms, log scale)")
        plt.ylabel("Latency (ms, log)")
        plt.savefig(os.path.join(OUT_DIR, "latency_boxplot_log.png"), bbox_inches="tight", dpi=150)
        plt.close()

    print("Done. CSVs and plots saved in:", OUT_DIR)

if __name__ == "__main__":
    main()
