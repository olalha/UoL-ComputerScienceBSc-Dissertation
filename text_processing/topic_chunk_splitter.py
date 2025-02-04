from math import ceil

def find_ideal_chunks(N, min_val, max_val):
    """
    Given a total N and bounds such that for each chunk X we must have
         min_val < X <= max_val,
    this function:
      - Finds all numbers of chunks i for which a legal partition exists.
      - Among these, it picks the i for which the average of the evenly distributed
        arithmetic progression (which is N/i) is closest to (min_val+max_val)/2.
      - For that i, it returns the list of evenly (ideally) distributed chunks.
    If no legal division exists, the function returns None.
    """

    def evenly_distributed_chunks(N, i, min_val, max_val):
        """
        For a given i (number of chunks), try to find integers A and d such that:
            X_j = A + (j-1)*d,  for j = 1,..., i,
        with the constraints:
            - sum_{j=1}^{i} X_j = N,
            - each X_j is in (min_val, max_val] (i.e. X_j >= min_val and X_j <= max_val).
        The function iterates d from the maximum possible value down to 0 and returns
        the first valid arithmetic progression it finds.
        """
        lower = min_val  # smallest allowed chunk value
        upper = max_val      # largest allowed chunk value

        # Check basic feasibility for this i
        if not (i * lower <= N <= i * upper):
            return None

        # Special case: only one chunk.
        if i == 1:
            return [N] if lower <= N <= upper else None

        # The maximum possible common difference:
        d_max = (upper - lower) // (i - 1)
        for d in range(d_max, -1, -1):
            # Using the sum formula, we have:
            #   i * A + (i*(i-1)//2)*d = N  =>  A = (N - (i*(i-1)//2)*d) / i.
            # We require A to be an integer.
            if (N - (i * (i - 1) // 2) * d) % i == 0:
                A = (N - (i * (i - 1) // 2) * d) // i
                if A >= lower and A + (i - 1) * d <= upper:
                    return [A + j * d for j in range(i)]
        return None

    # Determine the range of feasible i.
    # Since every chunk must be at least min_val and at most max_val,
    # we require:
    #    i*min_val <= N <= i*max_val.
    i_lower = ceil(N / max_val)       # smallest possible i
    i_upper = N // (min_val)        # largest possible i

    if i_lower > i_upper:
        # No legal partition exists.
        return None

    target_avg = (min_val + max_val) / 2.0  # ideal average value of a chunk
    best_error = None
    best_chunks = None

    # Loop over all candidate i in the feasible range.
    for i in range(i_lower, i_upper + 1):
        chunks = evenly_distributed_chunks(N, i, min_val, max_val)
        if chunks is not None:
            avg_chunks = sum(chunks) / len(chunks)
            error = abs(avg_chunks - target_avg)
            if best_error is None or error < best_error:
                best_error = error
                best_chunks = chunks

    return best_chunks

# =======================
# Example Usage:
# -----------------------
# For instance, if N = 300, min = 50, and max = 150 then the ideal target average is:
#      (50 + 150) / 2 = 100
# The function will search for a legal number of chunks i and, for example,
# it finds that i = 3 gives an evenly distributed partition:
#      [50, 100, 150]
# whose average is 100.
# =======================

# TODO: FIX THIS - This is broken
if __name__ == '__main__':
    # Try an example:
    total_word_count = 301
    min_val = 29
    max_val = 35
    result = find_ideal_chunks(total_word_count, min_val, max_val)
    if result is not None:
        print("Ideal evenly distributed chunks:", result)
    else:
        print("No valid partition exists.")
