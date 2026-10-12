"""HW2: Gaussian Naive Bayes on MNIST (part 1) and SVD on CIFAR-10 (part 3).

Run it with: python hw_bayes_svd.py
Data goes in ./data. If it's not there the script downloads it the first time
(MNIST is small, CIFAR is ~170 MB and the server is slow, so be patient).
"""
import gzip
import os
import pickle
import tarfile
import urllib.request

import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = "data"
rng = np.random.default_rng(0)


# ============================== Part 1: MNIST ==============================

def read_idx(path):
    with gzip.open(path, "rb") as f:
        raw = f.read()
    # MNIST uses this weird IDX binary format. Byte 3 tells you how many
    # dimensions there are, then each dimension's size is a 4-byte big-endian
    # int, and after that it's just the raw pixel/label bytes.
    num_dims = raw[3]
    shape = tuple(int.from_bytes(raw[4 + 4 * i: 8 + 4 * i], "big") for i in range(num_dims))
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * num_dims).reshape(shape)


def load_mnist(split):
    prefix = "train" if split == "train" else "t10k"
    images, labels = [], []
    for kind, out in [("images-idx3", images), ("labels-idx1", labels)]:
        name = f"{prefix}-{kind}-ubyte.gz"
        path = os.path.join(DATA_DIR, "mnist", name)
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            urllib.request.urlretrieve("https://raw.githubusercontent.com/fgnt/mnist/master/" + name, path)
        out.append(read_idx(path))
    # Flatten each 28x28 image into 784 numbers and scale pixels to 0..1
    return images[0].reshape(-1, 784) / 255.0, labels[0].astype(int)


class GaussianNaiveBayes:
    # Fit is O(N*D) time, predict is O(M*C*D), and we only store O(C*D) numbers
    # (N train images, M test images, C = 10 classes, D = 784 pixels)

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, x, y):
        self.classes = np.unique(y)
        # Prior P(c) is just how often each digit shows up in training
        self.log_priors = np.log(np.array([np.mean(y == c) for c in self.classes]))
        # Likelihood: every class gets one mean and one variance per pixel.
        # We never store how two pixels vary together, and that's literally
        # the "naive" part (each pixel is treated as independent given the class)
        self.means = np.array([x[y == c].mean(axis=0) for c in self.classes])
        self.variances = np.array([x[y == c].var(axis=0) for c in self.classes])
        # Problem: edge pixels are 0 in every image, so their variance is 0 and
        # the gaussian turns into a spike that kills any image with a stray
        # pixel there. Fix is to add a small floor to every variance, scaled to
        # the biggest pixel variance (same trick sklearn uses)
        self.variances += self.var_smoothing * x.var(axis=0).max()
        return self

    def log_posterior_unnormalized(self, x):
        scores = np.empty((x.shape[0], len(self.classes)))
        for i in range(len(self.classes)):
            # Since the pixels are independent, the likelihood is a product of
            # 784 gaussians. Taking the log turns that into a sum, which also
            # keeps it from underflowing to 0 (784 tiny numbers multiplied = 0)
            log_norm = -0.5 * np.sum(np.log(2 * np.pi * self.variances[i]))
            dist = np.sum((x - self.means[i]) ** 2 / self.variances[i], axis=1)
            scores[:, i] = self.log_priors[i] + log_norm - 0.5 * dist
        return scores

    def predict(self, x):
        # P(x) is the same no matter which class, so we can skip it and just
        # take whichever class has the biggest prior * likelihood
        return self.classes[np.argmax(self.log_posterior_unnormalized(x), axis=1)]


def run_part1():
    x_train, y_train = load_mnist("train")
    x_test, y_test = load_mnist("test")

    baseline = GaussianNaiveBayes(1e-9).fit(x_train, y_train)
    print(f"test accuracy, alpha=1e-9: {np.mean(baseline.predict(x_test) == y_test):.4f}")

    # Pick the smoothing amount using 10k images held out from TRAINING data.
    # If I tuned on the test set the final accuracy would be cheating a bit
    perm = rng.permutation(len(x_train))
    val, fit = perm[:10_000], perm[10_000:]
    alphas = [1e-9, 1e-4, 1e-2, 0.05, 0.1, 0.2, 0.5]
    val_acc = [np.mean(GaussianNaiveBayes(a).fit(x_train[fit], y_train[fit]).predict(x_train[val])
                       == y_train[val]) for a in alphas]
    for a, acc in zip(alphas, val_acc):
        print(f"  alpha={a:<6g} val acc={acc:.4f}")
    best_alpha = alphas[int(np.argmax(val_acc))]

    model = GaussianNaiveBayes(best_alpha).fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(f"test accuracy, alpha={best_alpha}: {np.mean(y_pred == y_test):.4f}")

    confusion = np.zeros((10, 10), dtype=int)
    np.add.at(confusion, (y_test, y_pred), 1)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(confusion, cmap="Blues")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, confusion[i, j], ha="center", va="center", fontsize=5.5,
                    color="white" if confusion[i, j] > 500 else "black")
    ax.set(xticks=range(10), yticks=range(10), xlabel="predicted", ylabel="true")
    fig.tight_layout()
    fig.savefig("fig_confusion.pdf")

    # ---- Part 2 numbers: checking how wrong the naive assumption is ----
    sevens = x_train[y_train == 7]
    center = 14 * 28 + 14
    # Even inside one class, a pixel and the one below it move together a lot.
    # The model assumes that correlation is 0
    r_below = np.corrcoef(sevens[:, center], sevens[:, center + 28])[0, 1]
    print(f"class 7: corr(pixel (14,14), pixel below) = {r_below:.2f}")
    print(f"class 7: pixel (14,14) is exactly 0 in {np.mean(sevens[:, center] == 0):.0%} of images")
    # Turn scores into actual probabilities (log-sum-exp so nothing overflows).
    # Because correlated pixels get counted as separate evidence, the model
    # ends up way too sure of itself, often >99.9% even when it's wrong
    scores = model.log_posterior_unnormalized(x_test)
    row_max = scores.max(axis=1, keepdims=True)
    log_post = scores - row_max - np.log(np.exp(scores - row_max).sum(axis=1, keepdims=True))
    sure = np.exp(log_post.max(axis=1)) > 0.999
    print(f"predictions with posterior > 0.999: {sure.mean():.0%}, "
          f"accuracy on those: {np.mean(y_pred[sure] == y_test[sure]):.0%}")


# ============================ Part 3: CIFAR-10 =============================

def load_cifar10_test():
    tar_path = os.path.join(DATA_DIR, "cifar10", "cifar-10-python.tar.gz")
    image_dir = os.path.join(DATA_DIR, "cifar10_images", "test")
    if not os.path.exists(tar_path) and not os.path.isdir(image_dir):
        os.makedirs(os.path.dirname(tar_path), exist_ok=True)
        print("downloading CIFAR-10 (~170 MB, one time only)...")
        urllib.request.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", tar_path)
    if os.path.exists(tar_path):
        with tarfile.open(tar_path) as tar:
            batch = pickle.load(tar.extractfile("cifar-10-batches-py/test_batch"), encoding="bytes")
        # Each row is all the red values, then green, then blue, so reshape
        # and reorder it into normal (N, 32, 32, 3) images
        return batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), np.array(batch[b"labels"])
    # Backup plan if you have the images as jpg folders instead of the tarball
    from PIL import Image
    root = os.path.join(DATA_DIR, "cifar10_images", "test")
    images, labels = [], []
    for label, name in enumerate(sorted(os.listdir(root))):
        for f in sorted(os.listdir(os.path.join(root, name))):
            images.append(np.asarray(Image.open(os.path.join(root, name, f)).convert("RGB")))
            labels.append(label)
    return np.stack(images), np.array(labels)


def run_part3():
    images, labels = load_cifar10_test()
    a = images.reshape(len(images), -1) / 255.0      # one image per row, 3072 values each
    mean_image = a.mean(axis=0)
    a_centered = a - mean_image                      # subtract the average image (this makes it PCA)

    # A = U S V^T. full_matrices=False skips the huge unused part of U.
    # O(N*D^2) time since N > D, O(N*D) space
    u, s, vt = np.linalg.svd(a_centered, full_matrices=False)
    energy = np.cumsum(s ** 2) / np.sum(s ** 2)

    k_values = [10, 50, 75, 100, 500]
    show = [np.flatnonzero(labels == c)[3] for c in range(10)]
    fig, axes = plt.subplots(len(k_values) + 1, 10, figsize=(7.2, 4.6))
    for col, idx in enumerate(show):
        axes[0, col].imshow(images[idx])
    for row, k in enumerate(k_values, start=1):
        # Keep only the top k singular values and rebuild the images.
        # eAch image is now just k numbers instead of 3072 pixels
        recon = np.clip((u[:, :k] * s[:k]) @ vt[:k] + mean_image, 0, 1)
        mse = np.mean((recon - a) ** 2)
        # Eckart-Young says the error is just the leftover singular values,
        # so this should match the measured error
        rel_err = np.sqrt(np.sum(s[k:] ** 2) / np.sum(s ** 2))
        print(f"k={k:<4} energy={energy[k - 1]:.3f}  MSE={mse:.5f}  "
              f"PSNR={10 * np.log10(1 / mse):.2f} dB  rel. error={rel_err:.3f}")
        for col, idx in enumerate(show):
            axes[row, col].imshow(recon[idx].reshape(32, 32, 3))
        axes[row, 0].set_ylabel(f"k={k}", fontsize=7)
    axes[0, 0].set_ylabel("original", fontsize=7)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(pad=0.3)
    fig.savefig("fig_svd.pdf")


if __name__ == "__main__":
    run_part1()
    run_part3()