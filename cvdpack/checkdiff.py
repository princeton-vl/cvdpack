
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args()

    before = np.load(args.before)
    after = np.load(args.after)

    print(before.shape)
    print(after.shape)
    print(np.isclose(before, after).astype(np.float32).mean())

    plt.subplot(1, 4, 1)
    plt.imshow(before)
    plt.colorbar()
    plt.subplot(1, 4, 2)
    plt.imshow(after)
    plt.colorbar()
    plt.subplot(1, 4, 3)
    cmap = plt.get_cmap("bwr")
    plt.imshow((before - after).clip(-1, 1), cmap=cmap)
    plt.colorbar()
    plt.subplot(1, 4, 4)
    plt.imshow(np.isclose(before, after, atol=1e-3).astype(np.float32))
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.scatter(before.flatten().clip(0, 100), after.flatten().clip(0, 100))
    plt.show()

if __name__ == "__main__":
    main()