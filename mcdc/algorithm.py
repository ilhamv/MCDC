import numba as nb


@nb.njit
def binary_search(val, grid, length=-1):
    """
    Binary search that returns the bin index of the value `val` given the grid
    `grid` with specified length to be searched in.
    If length is not specified, it searches in the entire grid.

    Some special cases:
        val < min(grid)  --> -1
        val > max(grid)  --> size of bins
        val = a grid point --> bin location whose upper bound is val
                                 (-1 if val = min(grid)
    """

    left = 0
    if length == -1:
        right = len(grid) - 1
    else:
        right = length - 1
    mid = -1
    while left <= right:
        mid = int((left + right) / 2)
        if grid[mid] < val:
            left = mid + 1
        else:
            right = mid - 1
    return int(right)
