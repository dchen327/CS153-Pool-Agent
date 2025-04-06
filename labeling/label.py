#%%
import shutil
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

label_map = {
    '1': 'stripe',
    '2': 'solid',
    '3': 'cue',
    '4': 'eight-ball',
    '5': 'bad'
}

base_path = Path('.')
unlabeled_path = base_path / 'unlabeled'
output_paths = {label: base_path / label for label in label_map.values()}

for path in output_paths.values():
    path.mkdir(parents=True, exist_ok=True)

counters = {label: len(list(output_paths[label].glob('*.png'))) + 1 for label in label_map.values()}

for img_path in sorted(unlabeled_path.glob('*.*')):  # Match any image extension
    try:
        img = Image.open(img_path)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'File: {img_path.name}')
        plt.show(block=False)

        while True:
            print("Label this image:")
            print("[1] Stripe   [2] Solid   [3] Cue   [4] Eight-ball   [5] Bad")
            label_input = input("Enter 1–5: ").strip()
            if label_input in label_map:
                label = label_map[label_input]
                break
            else:
                print("Invalid input. Please enter a number from 1 to 5.")

        plt.close()

        # move and rename the file
        count = counters[label]
        new_filename = f"{label}{count}.png"
        shutil.move(str(img_path), str(output_paths[label] / new_filename))
        counters[label] += 1

        print(f"Labeled as {label}, saved as {new_filename}\n")

    except Exception as e:
        print(f"Error processing {img_path.name}: {e}")

# %%
