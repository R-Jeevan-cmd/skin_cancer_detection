"""
federated_simulation.py
Federated Learning with ResNet50 (from resnet50_fl.py),
corrupted-image skipping, weighted averaging, and charts.

Usage:
python3 federated_simulation.py --data_dir data --rounds 3 --local_epochs 1
"""
import argparse
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, metrics
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt

# NEW: use your FL ResNet file
import resnet50_fl


# --------------------------------------------------------------
# IMAGE LOADING (SAFE) — SKIPS CORRUPTED IMAGES
# --------------------------------------------------------------
def verify_and_list_images(client_dir, allowed_exts=(".jpg", ".jpeg", ".png")):
    client_dir = Path(client_dir)
    classes = sorted([d for d in client_dir.iterdir() if d.is_dir()])

    filepaths, labels, class_names = [], [], []
    class_names = [d.name for d in classes]

    for idx, cls in enumerate(classes):
        for p in sorted(cls.iterdir()):
            if not p.is_file(): 
                continue
            if p.suffix.lower() not in allowed_exts:
                continue
            try:
                with Image.open(p) as im:
                    im.verify()
            except Exception:
                print(f"[skip] corrupted: {p}")
                continue

            filepaths.append(str(p))
            labels.append(idx)

    return filepaths, labels, class_names


def build_tf_dataset_from_lists(files, labels, image_size=(128,128), batch_size=16):
    if len(files) == 0:
        raise RuntimeError("No valid images for dataset.")

    ds = tf.data.Dataset.from_tensor_slices((files, labels))

    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, 3, expand_animations=False)
        img = tf.image.resize(img, image_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    ds = ds.shuffle(len(files))
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# --------------------------------------------------------------
# WEIGHTED FEDERATED AVERAGING
# --------------------------------------------------------------
def weighted_average_weights(client_weights, client_sizes):
    total = float(sum(client_sizes))

    averaged = []
    num_layers = len(client_weights[0])

    for l in range(num_layers):
        weighted = None
        for i, w in enumerate(client_weights):
            part = w[l] * (client_sizes[i] / total)
            weighted = part if weighted is None else weighted + part
        averaged.append(weighted)

    return averaged


# --------------------------------------------------------------
# GLOBAL VAL EVALUATION (optional)
# --------------------------------------------------------------
def evaluate_global(model, val_dir, image_size=(128,128), batch_size=16):
    if not val_dir or not os.path.isdir(val_dir):
        return None

    files, labels, _ = verify_and_list_images(val_dir)
    if len(files) == 0:
        return None

    ds = build_tf_dataset_from_lists(files, labels, image_size, batch_size)
    model.compile(optimizer="adam",
                  loss=losses.SparseCategoricalCrossentropy(),
                  metrics=[metrics.SparseCategoricalAccuracy()])
    return model.evaluate(ds, verbose=1)


# --------------------------------------------------------------
# MAIN FEDERATED PROCESS
# --------------------------------------------------------------
def main(args):
    data_dir = Path(args.data_dir)
    client_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    # Detect class count
    sample_files, sample_labels, class_names = verify_and_list_images(client_dirs[0])
    num_classes = len(class_names)

    print("Clients:", client_dirs)
    print("Number of classes:", num_classes)

    # --------- MODEL: using your new FL ResNet50 ----------------
    def build_global():
        return resnet50_fl.create_resnet50_fl_model(
            input_shape=(128,128,3),
            num_classes=num_classes
        )

    global_model = build_global()
    global_model.build((None,128,128,3))
    global_model.compile(optimizer="adam",
                         loss=losses.SparseCategoricalCrossentropy(),
                         metrics=[metrics.SparseCategoricalAccuracy()])


    # Preload file lists for speed
    client_data = []
    client_sizes = []
    for cd in client_dirs:
        files, labels, _ = verify_and_list_images(cd)
        client_data.append((files, labels))
        client_sizes.append(len(files))

    print("Client valid sample counts:", client_sizes)

    # Tracking metrics
    global_acc = []
    global_loss = []
    round_client_metrics = {}

    # ----------------------------------------------------------
    #       TRAINING ROUNDS
    # ----------------------------------------------------------
    for rnd in range(1, args.rounds+1):
        print(f"\n=== FEDERATED ROUND {rnd} ===")
        local_weights = []
        local_counts = []

        for idx, cd in enumerate(client_dirs):
            files, labels = client_data[idx]
            samples = client_sizes[idx]

            if samples == 0:
                continue

            print(f"-> Client {idx+1} ({samples} samples)")

            ds = build_tf_dataset_from_lists(files, labels,
                                             image_size=(128,128),
                                             batch_size=args.batch_size)

            local_model = build_global()
            local_model.set_weights(global_model.get_weights())

            local_model.compile(optimizer="adam",
                                loss=losses.SparseCategoricalCrossentropy(),
                                metrics=[metrics.SparseCategoricalAccuracy()])

            history = local_model.fit(ds, epochs=args.local_epochs, verbose=1)

            # Save weights
            local_weights.append([np.array(w) for w in local_model.get_weights()])
            local_counts.append(samples)

            # Track metrics
            if rnd not in round_client_metrics:
                round_client_metrics[rnd] = []

            round_client_metrics[rnd].append({
                "client_id": idx+1,
                "accuracy": history.history.get("sparse_categorical_accuracy", []),
                "loss": history.history.get("loss", [])
            })

        # --------- AGGREGATION ----------
        averaged = weighted_average_weights(local_weights, local_counts)
        global_model.set_weights(averaged)
        print("Averaged global weights set.")

        # --------- GLOBAL EVAL ----------
        if args.val_dir:
            res = evaluate_global(global_model, args.val_dir,
                                  image_size=(128,128),
                                  batch_size=args.batch_size)
        else:
            res = None

        if res is not None:
            global_loss.append(res[0])
            global_acc.append(res[1])
        else:
            global_loss.append(None)
            global_acc.append(None)


    # ----------------------------------------------------------
    # SAVE FINAL GLOBAL MODEL
    # ----------------------------------------------------------
    out = args.output or "federated_global_model.keras"
    global_model.save(out)
    print("Saved:", out)


    # ----------------------------------------------------------
    # PLOTTING
    # ----------------------------------------------------------
    print("\nGenerating charts...")

    # Global accuracy
    if any(a is not None for a in global_acc):
        plt.figure()
        rounds = list(range(1, len(global_acc)+1))
        plt.plot(rounds, [a if a else 0 for a in global_acc], marker='o')
        plt.title("Global Accuracy per Round")
        plt.xlabel("Round")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.savefig("global_accuracy.png")
        print("Saved global_accuracy.png")

    # Client metrics per round
    for rnd, entries in round_client_metrics.items():
        # Accuracy
        plt.figure()
        for e in entries:
            plt.plot(e["accuracy"], marker='o', label=f"Client {e['client_id']}")
        plt.title(f"Client Accuracy – Round {rnd}")
        plt.xlabel("Local Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"client_accuracy_round{rnd}.png")

        # Loss
        plt.figure()
        for e in entries:
            plt.plot(e["loss"], marker='o', label=f"Client {e['client_id']}")
        plt.title(f"Client Loss – Round {rnd}")
        plt.xlabel("Local Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"client_loss_round{rnd}.png")

    print("All charts generated.")


# --------------------------------------------------------------
# CLI
# --------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--val_dir", type=str, default="")
    parser.add_argument("--output", type=str, default="federated_global_model.keras")
    args = parser.parse_args()

    main(args)