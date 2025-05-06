from isaaclab_eureka import EUREKA_ROOT_DIR
import os

# Correct ISAACLAB_LOG_DIR based on what you observed
ISAACLAB_LOG_DIR = os.path.abspath(os.path.join(EUREKA_ROOT_DIR, "..", "isaaclab", "logs"))

# Path to the dummy file
dummy_file_path = os.path.join(ISAACLAB_LOG_DIR, "can_you_find_me.txt")

# Try reading the file
try:
    with open(dummy_file_path, "r") as f:
        content = f.read()
    print("✅ Successfully found and read the file!")
    print(f"Content:\n{content}")
except FileNotFoundError:
    print("❌ Could not find the dummy file. Check your path!")
except Exception as e:
    print(f"❌ An error occurred: {e}")
