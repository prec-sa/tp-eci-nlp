import itertools
import sys
from multiprocessing import cpu_count, Pool
from grid_search_utils import generate_single_score


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def do_grid_search(params_grid, pool):

    tasks = list(product_dict(**params_grid))
    results = []

    for i, res in enumerate(pool.imap_unordered(generate_single_score,tasks),1):
        results.append(res)
        sys.stderr.write('\rdone {0:%}'.format(i/len(tasks)))
        sys.stderr.flush()

    return results


if __name__ == '__main__':

    p = Pool(cpu_count()//2)

    grid = {'dim': range(100, 600, 100), 'wordNgrams': range(1,11)}

    r = do_grid_search(grid, p)
