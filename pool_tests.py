import multiprocessing as mp
import time


def simulation(run_id):
    if mp.current_process().name != 'MainProcess':
        rank = mp.current_process()._identity[0]
        print(f'Running on process: {rank}')
    for i in range(100000):
        pass


def run_in_parallel():
    runs_id = list(range(Nsim))

    with mp.Pool(8) as p:
        p.map(simulation, runs_id)


def run():
    for i in range(Nsim):
        simulation(i)


Nsim = 100
start = time.time()
run_in_parallel()
end = time.time()
print("Time elapsed:", end - start)

start = time.time()
run()
end = time.time()
print("Time elapsed:", end - start)
