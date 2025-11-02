import os

# Path to the folder containing your .jpg files
folder = "train"

# Get all jpg files sorted by name
files = sorted([f for f in os.listdir(folder) if f.endswith(".jpg")])

# Loop through files and rename them sequentially
for i, filename in enumerate(files):
    new_index = i * 8
    new_name = f"frame_{new_index:06d}.jpg"
    src = os.path.join(folder, filename)
    dst = os.path.join(folder, new_name)
    os.rename(src, dst)
    print(f"Renamed: {filename} -> {new_name}")

print("Renaming complete.")
