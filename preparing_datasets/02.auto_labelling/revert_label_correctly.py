import os

LABELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'yolo_labels')
LABELS_DIR = os.path.normpath(LABELS_DIR)

for filename in os.listdir(LABELS_DIR):
    if filename.endswith('.txt'):
        filepath = os.path.join(LABELS_DIR, filename)
        with open(filepath, 'r') as f:
            lines = f.readlines()

        new_lines = []
        if filename.startswith('potato'):
            for line in lines:
                parts = line.strip().split()
                if parts and parts[0] == '0':
                    parts[0] = '1'
                new_lines.append(' '.join(parts) + '\n')
        elif filename.startswith('IMG'):
            for line in lines:
                parts = line.strip().split()
                if parts and parts[0] == '0':
                    parts[0] = '2'
                new_lines.append(' '.join(parts) + '\n')
        else:
            continue  # skip files that don't match

        with open(filepath, 'w') as f:
            f.writelines(new_lines)