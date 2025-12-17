import subprocess
import os

log_file = "git_output_v2.txt"

def run(cmd):
    try:
        # 5 second timeout
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, shell=True, timeout=5)
        return f"CMD: {cmd}\nRC: {result.returncode}\nOUT: {result.stdout}\nERR: {result.stderr}\n"
    except subprocess.TimeoutExpired:
        return f"CMD: {cmd} TIMEOUT\n"
    except Exception as e:
        return f"CMD: {cmd} ERR: {e}\n"

with open(log_file, "w") as f:
    f.write("Starting...\n")

out = run("git status")
with open(log_file, "a") as f:
    f.write(out)

# Try checkout directly
out = run("git checkout origin/main -- Stock_Sentiment_Project/googl_daily_prices.csv")
with open(log_file, "a") as f:
    f.write(out)
