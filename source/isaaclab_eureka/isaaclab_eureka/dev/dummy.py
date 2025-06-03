import re
task = "Isaac-Lift-Cube-Franka-v0"
match = re.search(r"Isaac-([A-Za-z]+)", task)
print( match.group(1).lower())