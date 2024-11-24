#!/usr/bin/env python3

import random
import sys

span = tuple[float, float] # life span

  
def generate_span(n_vec: int, pattern: list[tuple[int, int, int]]) -> list[span]:
    # generate type
    ratios = [sum([l[2] for l in pattern[:i+1]]) for i in range(len(pattern))]
    ratio_f = lambda x: len([r for r in ratios if r <= x])
    
    # generate span
    span = []
    for _ in range(n_vec):
        r = random.randint(0, ratios[-1] - 1)
        vec_type = ratio_f(r)

        life_len = random.randint(pattern[vec_type][0], pattern[vec_type][1])

        start = random.randint(0, 100-life_len)
        span.append((start, start + life_len))

    return span
    

def generate_sequence(n_basevec: int, n_qvec: int, n_qvalid: int, pattern: list[tuple[int, int, int]]) -> tuple[list[int], list[int]]:
    ADD = 0x0ADD
    DEL = 0x0DE1

    # generate_span
    vectors = zip(range(n_basevec), generate_span(n_basevec, pattern))
    
    # generate operations
    operations = []
    for i, (start, end) in vectors:
        operations.append((start, ADD, i))
        operations.append((end, DEL, i))
    operations.sort(key=lambda x: x[0])

    basevec_count = [(0, -1)]
    for i, op in enumerate(operations):
        if op[1] == ADD:
            basevec_count.append((basevec_count[-1][0] + 1, i))
        else:
            basevec_count.append((basevec_count[-1][0] - 1, i))
    basevec_count.pop(0)

    filtered_basevec_count = [(c, i) for c, i in basevec_count if c >= n_qvalid]

    if len(filtered_basevec_count) == 0:
        print("No valid query time")
        sys.exit(1)

    random.shuffle(filtered_basevec_count)

    selected_basevec_count = filtered_basevec_count[:n_qvec]

    query_time = [op[1] for op in selected_basevec_count]
    
    while len(query_time) < n_qvec:
        query_time += query_time

    query_time = query_time[:n_qvec]

    random.shuffle(query_time)

    return [op[2] for op in operations], query_time
    
    
def abbv(n: int) -> str:
    if n < 1000:
        return str(n)
    elif n < 1000000:
        return f"{n//1000}k"
    else:
        return f"{n//1000000}m"


if __name__ == '__main__':

    n_basevec = 10000 # number of base vector
    n_qvec = 100 # number of queries

    ops, qs = generate_sequence(n_basevec, n_qvec, 10, [(1, 99, 10)])

    with open("." + "/{}_op_{}base.txt".format("uniform", abbv(n_basevec)), 'w') as f:
        f.write('\n'.join(map(str, ops)))

    with open("." + "/{}_q{}_{}base.txt".format("uniform", abbv(n_qvec), abbv(n_basevec)), 'w') as f:
        f.write('\n'.join(map(str, qs)))