import itertools
import sys
from multiprocessing import cpu_count, Pool
from grid_search_utils import generate_single_score
import gc


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def do_grid_search(params_grid, pool):

    tasks = list(product_dict(**params_grid))
#    results = []

    results = pool.map(generate_single_score, tasks)

#    for i, res in enumerate(pool.imap_unordered(generate_single_score,tasks),1):
#        results.append(res)
#        sys.stderr.write('\rdone {0:%}'.format(i/len(tasks)))
#        sys.stderr.flush()

    return results

def grid_search_non_parallel(params_grid):
    
    with open('grid_results.txt', 'w') as results:
        
        for params in product_dict(**params_grid):
            
            res = generate_single_score(params)
            print(res)
            
            results.write(str(res))
            results.flush()
            gc.collect()
        
    

if __name__ == '__main__':
    
#    usable_cores = min(cpu_count()//2, 3)
#    p = Pool(usable_cores)

    grid = {'dim': range(100, 600, 100), 'wordNgrams': range(1,11)}

    grid_search_non_parallel(grid)

#    r = do_grid_search(grid, p)
