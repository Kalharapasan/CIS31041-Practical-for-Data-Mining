import numpy as np
print("Import NumPy as np (already done)\n")

# a. Import NumPy as np (already done)

print("a.Import NumPy as np (already done)\n")
# b. Concatenation of two arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
concatenated_arr = np.concatenate((arr1, arr2))
print("Concatenated Array:", concatenated_arr)
print("\n")

print("c. Create an empty array\n")
# c. Create an empty array
empty_array = np.empty((2, 2))
print("Empty Array:\n", empty_array)
print("\n")

print("d. Create an array of consecutive integers starting from 0\n")
# d. Create an array of consecutive integers starting from 0
consecutive_arr = np.arange(10)
print("Array of Consecutive Integers:", consecutive_arr)
print("\n")

print("e. Create a two-dimensional array\n")
# e. Create a two-dimensional array
two_d_array = np.array([[1, 2, 3], [4, 5, 6]])
print("2D Array:\n", two_d_array)
print("\n")

print("f. Sort the array\n")
# f. Sort the array
unsorted_arr = np.array([3, 1, 4, 1, 5, 9])
sorted_arr = np.sort(unsorted_arr)
print("Sorted Array:", sorted_arr)
print("\n")

print("g. Find shape, size, dimension\n")
# g. Find shape, size, dimension
print("Shape:", two_d_array.shape)
print("Size:", two_d_array.size)
print("Dimension:", two_d_array.ndim)
print("\n")

print("h. Reshape the array\n")
# h. Reshape the array
reshaped_arr = np.reshape(consecutive_arr, (2, 5))
print("Reshaped Array:\n", reshaped_arr)
print("\n")

print("i. Increase dimension of the array\n")
# i. Increase dimension of the array
expanded_arr = np.expand_dims(arr1, axis=0)
print("Increased Dimension Shape:", expanded_arr.shape)
print("\n")

print("j. Indexing and slicing\n")
# j. Indexing and slicing
print("First element of arr1:", arr1[0])
print("Slice of arr1:", arr1[1:])
print("\n")

print("k. Create an array from existing data\n")
# k. Create an array from existing data
copied_arr = np.copy(arr1)
print("Copied Array:", copied_arr)
print("\n")

print("l. Reshape the existing array\n")
# l. Reshape the existing array
reshaped_existing = np.reshape(arr1, (3, 1))
print("Reshaped Existing Array:\n", reshaped_existing)
print("\n")

print("m. Print the array in vertical and horizontal stack\n")
# m. Print the array in vertical and horizontal stack
vstacked = np.vstack((arr1, arr2))
hstacked = np.hstack((arr1, arr2))
print("Vertical Stack:\n", vstacked)
print("Horizontal Stack:", hstacked)
print("\n")

print("n. Split the array\n")
# n. Split the array
split_arr = np.array_split(concatenated_arr, 2)
print("Split Arrays:", split_arr)
print("\n")

print("o. Basic array operations\n")
# o. Basic array operations
sum_arr = arr1 + arr2
diff_arr = arr1 - arr2
prod_arr = arr1 * arr2
print("Sum:", sum_arr)
print("Difference:", diff_arr)
print("Product:", prod_arr)
print("\n")

print("p. Summary of the array\n")
# p. Summary of the array
print("Mean:", np.mean(arr1))
print("Standard Deviation:", np.std(arr1))
print("\n")

print("q. Multiply elements in array\n")
# q. Multiply elements in array
multiplied_arr = arr1 * 2
print("Multiplied Array:", multiplied_arr)
print("\n")

print("r. Find max and min elements\n")
# r. Find max and min elements
print("Max:", np.max(arr1))
print("Min:", np.min(arr1))
print("\n")

print("s. Transpose of a 2D array\n")
# s. Transpose of a 2D array
transposed_two_d = np.transpose(two_d_array)
print("Transposed 2D Array:\n", transposed_two_d)
print("\n")
