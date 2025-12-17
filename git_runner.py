import subprocess
import os

def run_git_cmd(cmd):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, shell=True)
        return f"CMD: {cmd}\nRC: {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\n{'-'*20}\n"
    except Exception as e:
        return f"CMD: {cmd}\nException: {e}\n{'-'*20}\n"

log_file = "git_output.txt"
output = ""

# Check status
output += run_git_cmd("git status")

# Fetch origin
output += run_git_cmd("git fetch origin")

# Try to checkout from origin/main
output += run_git_cmd("git checkout origin/main -- Stock_Sentiment_Project/googl_daily_prices.csv")
# Fallback to origin/master if main fails
output += run_git_cmd("git checkout origin/master -- Stock_Sentiment_Project/googl_daily_prices.csv")

with open(log_file, "w") as f:
    f.write(output)

print("Done")
