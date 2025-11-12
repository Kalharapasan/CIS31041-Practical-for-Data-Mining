# 01. Matrix Implementation
print("01. Matrix Implementation")
matrix = [[1, 2], [3, 4]]
print("Original Matrix:", matrix)

# Transform the matrix (e.g., transpose)
transformed_matrix = [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
print("Transformed Matrix (Transpose):", transformed_matrix)

print("\n02. Tuples Implementation")
# 02. Tuples Implementation
tuple1 = (1, 2, 3)
tuple2 = (4, 5, 6)
merged_tuple = tuple1 + tuple2
print("Merged Tuple:", merged_tuple)
print("Elements in Merged Tuple:")
for item in merged_tuple:
    print(item)

print("\n03. Array")
# 03. Array
array = [10, 20, 30, 40, 50]
print("Defined Array:", array)
print("Last Element:", array[-1])

print("\n04. Date Functions")
# 04. Date functions
from datetime import date

today = date.today()
birthday = date(2000, 1, 1)  # Example birthday
print("Today's Date:", today)
print("Birthday:", birthday)

print("\n05. NumPy Tasks")
# 05. Numpy
import numpy as np

# a. Import numpy as np (already done)
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print("a:", a)
print("b:", b)

# b. Concatenate
concatenated = np.concatenate((a, b))
print("Concatenated:", concatenated)

# c. Create empty array
empty = np.empty((2, 2))
print("Empty array:\n", empty)

# d. Consecutive integers
consec_array = np.arange(0, 10)
print("Consecutive Integers Array:", consec_array)

# e. Two-dimensional array
two_d_array = np.array([[1, 2], [3, 4]])
print("2D Array:\n", two_d_array)

# f. Sort
unsorted_array = np.array([3, 1, 4, 2])
sorted_array = np.sort(unsorted_array)
print("Sorted Array:", sorted_array)

# g. Shape, size, dimension
print("Shape:", two_d_array.shape)
print("Size:", two_d_array.size)
print("Dimension:", two_d_array.ndim)

# h. Reshape
reshaped = consec_array.reshape(2, 5)
print("Reshaped Array:\n", reshaped)

# i. Increase dimension
expanded = np.expand_dims(a, axis=0)
print("Increased Dimension (Row Vector):", expanded)

# j. Indexing and slicing
print("First Element:", a[0])
print("Slice [1:]:", a[1:])

# k. Array from existing data
existing_data = list(range(6))
array_from_data = np.array(existing_data)
print("Array from existing data:", array_from_data)

# l. Reshape existing element
reshaped_existing = array_from_data.reshape(2, 3)
print("Reshaped Existing Data:\n", reshaped_existing)

# m. Vertical and horizontal stack
v_stack = np.vstack((a, b))
h_stack = np.hstack((a, b))
print("Vertical Stack:\n", v_stack)
print("Horizontal Stack:", h_stack)

# n. Split array
split_arr = np.split(consec_array, 2)
print("Split Array:", split_arr)

# o. Basic operations
print("Addition:", a + b)
print("Multiplication:", a * b)

# p. Summary
print("Mean:", np.mean(a))
print("Max:", np.max(a))
print("Min:", np.min(a))

# q. Multiply each element
multiplied = a * 10
print("Multiplied by 10:", multiplied)
