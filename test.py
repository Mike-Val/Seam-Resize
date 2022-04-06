# A left-rotation of an array A is defined as a permutation of A such that every element 
# is shifted by one position to the left except for the fist element that is moved to the last position. 
# For example, with A = [1, 2, 3, 4, 5, 6, 7, 8, 9], a left-rotation would change A into A = [2,3,4,5,6,7,8,9,1].
# Write an algorithm rotate(A,k) that takes an array A and performs k left-rotations on A. 
# The complexity of your algorithm must be O(n), which means that the complexity must not depend on k.
def rotate(A, k):
    if len(A) == 0:
        return A
    k = k % len(A)
    if k == 0:
        return A
    else:
        B = A[:]
        for i in range(len(A)):
            B[i] = A[(i + k) % len(A)]
        return B

A = [1, 2, 3, 4, 5, 6, 7, 8, 9]
print(rotate(A, 1200))