import os
import sys

def get_size(path):
    total = 0

    for root, dirs, files in os.walk(path):
        for file in files:
            try:
                total += os.path.getsize(os.path.join(root, file))
            except (OSError, PermissionError):
                pass

    return total

python_folder = os.path.dirname(sys.executable)
size = get_size(python_folder)

print(f"Python + Libraries total size: {size / (1024**3):.2f} GB")
print(f"Total size in MB: {size / (1024**2):.2f} MB")