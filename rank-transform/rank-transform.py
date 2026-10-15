def rank_transform(values: list) -> list:
    """
    Returns the one-based average rank of every value.
    """
    # Write code here
    values_sorted = sorted(values)

    ranks = {}

    i = 0
    while i < len(values):
        if i < len(values) - 1 and values_sorted[i] == values_sorted[i+1]:
            start_i = i + 1
            while i < len(values) and values_sorted[i] == values_sorted[start_i]:
                i += 1
            end_i = i
            avg = start_i + (end_i - start_i) / 2
            ranks[values_sorted[start_i]] = float(avg)
        else:
            ranks[values_sorted[i]] = float(i + 1)
            i += 1

    out = [0 for _ in range(len(values))]

    for i in range(len(values)):
        out[i] = ranks[values[i]]

    return out
    