import numpy as np

# Define the number of dimensions (768)
n_dims = 768

# Generate two 768-dimensional vectors with values ranging from 0 to 0.768
A = np.random.uniform(0, 0.768, size=n_dims)
B = np.random.uniform(0, 0.768, size=n_dims)

print("Vector A:", A)
print("Vector B:", B)