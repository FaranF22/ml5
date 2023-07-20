import matplotlib.pyplot as plt
import numpy as np


def displayData(X, example_width=None):
    # DISPLAYDATA Display 2D data in a nice grid
    #   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
    #   stored in X in a nice grid. It returns the figure handle h and the 
    #   displayed array if requested.

    # Set example_width automatically if not passed in
    if not example_width:
        example_width = int(round(np.sqrt(X.shape[1])))

    # Compute rows, cols
    m, n = X.shape
    example_height = int(n / example_width)

    # Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    # Between images padding
    pad = 1

    # Setup blank display
    display_array = - np.ones((pad + display_rows * (example_height + pad),
                               pad + display_cols * (example_width + pad)))

    # Copy each example into a patch on the display array
    curr_ex = 1
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex >= m:
                break

            # Copy the patch

            # Get the max value of the patch
            max_val = max(abs(X[curr_ex, :]));
            ind1 = pad + (j - 1) * (example_height + pad) + np.arange(example_height)
            ind2 = pad + (i - 1) * (example_width + pad) + np.arange(example_width)

            display_array[np.ix_(ind1, ind2)] = np.reshape(X[curr_ex, :], (example_height, example_width)) / max_val
            curr_ex = curr_ex + 1
        if curr_ex >= m:
            break

            # Plot the grid
    plt.imshow(np.transpose(display_array))
    plt.gray()
    plt.show()

    return display_array
