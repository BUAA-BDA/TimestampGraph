import os
import time
import multiprocessing
import multiprocessing.pool

os.environ['TZ'] = 'Asia/Shanghai'
time.tzset()
cwd = "."

dim_t = int
path_t = str
dataset_t = tuple[path_t, dim_t, str]

def abbv(n: int) -> str:
    if n < 1000:
        return str(n)
    elif n < 1000000:
        return f"{n//1000}k"
    else:
        return f"{n//1000000}m"

def generate_workload(binary: path_t, ks: list[int], datasets: list[tuple[dataset_t, dataset_t]], data_nums: list[int], query_num: int, patterns: list[str], it: int = 10) -> list[str]:

    def op_path(pattern: str) -> str:
        return f"{cwd}/data/{pattern}_op_{abbv(data_num)}base.txt"
    
    def q_path(pattern: str) -> str:
        return f"{cwd}/data/{pattern}_q{abbv(10000)}_{abbv(data_num)}base.txt"

    workload: list[tuple[str, str]] = []
    for pattern in patterns:
        for dataset, queryset in datasets:
            for data_num in data_nums:
                for k in ks:
                    workload.append((
                        f"{binary} {dataset[1]} {data_num} {k} {dataset[0]} {queryset[0]} {query_num} " + \
                        f"{op_path(pattern)} {q_path(pattern)} " + \
                        f"{op_path(pattern).removesuffix("txt") + dataset[2] + ".gt.ivecs"}",
                        f"{dataset[2]}{abbv(data_num)}_k{k}_{pattern}"
                    ))

    commands = []
    for work in workload:
                for i in range(it):
                    commands.append(work[0] + " " + f" > {cwd}/log/{time.strftime("%m%d_%H%M%S")}_{work[1]}_{i}.log")
        
    
    return commands

def run_in_parrallel(cmd: list[str], pool: multiprocessing.pool.Pool):
    while len(cmd) > 0:
        workload = cmd.pop(0)
        pool.apply_async(os.system, args=(workload,))
    pool.close()
    pool.join()


if __name__ == '__main__':
    os.chdir(cwd)
    print(os.getcwd())

    # example datasets(in .fvecs format)
    GIST_DATA = ((f"{cwd}/data/GIST1M/gist_base.fvecs", 960, "GIST"), (f"{cwd}/data/GIST1M/gist_query.fvecs", 960, ""))
    SIFT_DATA = ((f"{cwd}/data/SIFT1M/sift_base.fvecs", 128, "SIFT"), (f"{cwd}/data/SIFT1M/sift_query.fvecs", 128, ""))
    GLOVE_DATA = ((f"{cwd}/data/GLOVE/glove-200-base.fvecs", 200, "GLOVE"), (f"{cwd}/data/GLOVE/glove-200-query.fvecs", 200, ""))
    DEEP_DATA = ((f"{cwd}/data/DEEP/deep_base.fvecs", 96, "DEEP"), (f"{cwd}/data/DEEP/deep_query.fvecs", 96, ""))

    # example for generating workloads
    workload = generate_workload(
        binary=f"{cwd}/build/timestamp_graph_test", 
        ks=[10],
        datasets=[SIFT_DATA, GIST_DATA, DEEP_DATA, GLOVE_DATA],
        data_nums=[1000000],
        query_num=10000,
        patterns=["short", "mix", "long", "uniform"],
        it=1
    )
    print(workload)

    # run in parrallel
    pool: multiprocessing.pool.Pool = multiprocessing.Pool(processes=16)
    run_in_parrallel(workload, pool)




