import numpy as np
import datetime

# Task 1: Matrix Implementation
# a. Create the matrix
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print("Original Matrix:")
for row in matrix:
    print(row)

# b. Transform the matrix (Transpose)
transposed_matrix = [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
print("\nTransposed Matrix:")
for row in transposed_matrix:
    print(row)

# Task 2: Tuples Implementation
# a. Creating a tuple
tuple1 = (1, 2, 3, 4, 5)
print("\nTuple1:", tuple1)

# b. Create two tuples and merge together
tuple2 = ('a', 'b', 'c')
merged_tuple = tuple1 + tuple2
print("Merged Tuple:", merged_tuple)

# c. Print the elements in tuple
print("Elements in tuple:", merged_tuple)

# Task 3: Array Implementation
# a. Defining an array
array = [10, 20, 30, 40, 50]

# b. Print the array
print("\nArray:", array)

# c. Print the last element
print("Last element of the array:", array[-1])

# Task 4: Date functions
# a. Today's date
print("\nToday's Date:", datetime.date.today())

# b. Birthday (Example)
birthday = datetime.date(2000, 5, 15)
print("Example Birthday:", birthday)

# Task 5: Numpy Implementation
# a. Import numpy as np (already done)
# b. Concatenation of two arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
concatenated_arr = np.concatenate((arr1, arr2))
print("\nConcatenated Array:", concatenated_arr)

# c. Create an empty array
empty_array = np.empty((2, 2))
print("Empty Array:\n", empty_array)

# d. Create an array of consecutive integers
consecutive_arr = np.arange(10)
print("Array of Consecutive Integers:", consecutive_arr)

# e. Create a 2D array
two_d_array = np.array([[1, 2, 3], [4, 5, 6]])
print("2D Array:\n", two_d_array)

# f. Sort the array
unsorted_arr = np.array([3, 1, 4, 1, 5, 9])
sorted_arr = np.sort(unsorted_arr)
print("Sorted Array:", sorted_arr)

# g. Find shape, size, dimension
print("Shape:", two_d_array.shape)
print("Size:", two_d_array.size)
print("Dimension:", two_d_array.ndim)

# h. Reshape the array
reshaped_arr = np.reshape(consecutive_arr, (2, 5))
print("Reshaped Array:\n", reshaped_arr)

# i. Increase dimension
expanded_arr = np.expand_dims(arr1, axis=0)
print("Expanded Array Shape:", expanded_arr.shape)

# j. Indexing and slicing
print("First element of arr1:", arr1[0])
print("Slice of arr1:", arr1[1:])

# k. Create an array from existing data
copied_arr = np.copy(arr1)
print("Copied Array:", copied_arr)

# l. Reshape existing array
reshaped_existing = np.reshape(arr1, (3, 1))
print("Reshaped Existing Array:\n", reshaped_existing)

# m. Print the array in vertical and horizontal stack
vstacked = np.vstack((arr1, arr2))
hstacked = np.hstack((arr1, arr2))
print("Vertical Stack:\n", vstacked)
print("Horizontal Stack:", hstacked)

# n. Split the array
split_arr = np.array_split(concatenated_arr, 2)
print("Split Arrays:", split_arr)

# o. Basic array operations
sum_arr = arr1 + arr2
diff_arr = arr1 - arr2
prod_arr = arr1 * arr2
print("Sum:", sum_arr)
print("Difference:", diff_arr)
print("Product:", prod_arr)

# p. Summary of the array
print("Mean:", np.mean(arr1))
print("Standard Deviation:", np.std(arr1))

# q. Multiply elements in array
multiplied_arr = arr1 * 2
print("Multiplied Array:", multiplied_arr)

# r. Find max and min
print("Max:", np.max(arr1))
print("Min:", np.min(arr1))

# s. Transpose of array
transposed_two_d = np.transpose(two_d_array)
print("Transposed 2D Array:\n", transposed_two_d)
