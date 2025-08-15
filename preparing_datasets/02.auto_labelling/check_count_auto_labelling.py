import os
import argparse

def count_labels(image_dir, label_dir, vegetable='all'):
    images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(images)} images in {image_dir}")
    # if vegetable != 'all':
    #     images = [f for f in images if vegetable in f]

    labelled = []
    unlabelled = []

    for img in images:
        label_file = os.path.splitext(img)[0] + '.txt'
        label_path = os.path.join(label_dir, label_file)
        if os.path.exists(label_path):
            labelled.append(img)
        else:
            unlabelled.append(img)

    print(f"Total images: {len(images)}")
    print(f"Labelled: {len(labelled)}")
    print(f"Unlabelled: {len(unlabelled)}")
    # for img in unlabelled:
    #     print(f" - {img}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check image labelling status.")
    parser.add_argument('--vegetable', default='all', help='Vegetable type to filter (default: all)')
    args = parser.parse_args()

    # Set image_dir and label_dir automatically
    if args.vegetable == 'all':
        vegetable = 'all'
    else:
        vegetable = args.vegetable
    image_dir = os.path.abspath(os.path.join('..', '..', 'data', 'raw', vegetable))
    print(f"Using image directory: {image_dir}")
    label_dir = os.path.abspath(os.path.join('..', '..', 'data', 'yolo_labels'))

    count_labels(image_dir, label_dir, vegetable)