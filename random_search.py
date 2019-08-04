from grid_search_utils import generate_single_score
import gc
from scipy.stats import randint


def parameters_from_dists(param_dists):

    params = {param: value['dist'](**value['dist_params']).rvs() for \
              param, value in param_dists.items() }

    return params


def do_random_search(param_dists, n_iter):

    with open('random_results.txt', 'a') as results:

        for i in range(n_iter):

            print('\n Iteration ', i, ' of ', n_iter, '\n')

            params = parameters_from_dists(param_dists)

            res = generate_single_score(params)
            print(res)

            results.write(str(res) + '\n')
            results.flush()
            gc.collect()


if __name__ == '__main__':

    dists = {
            'dim': {'dist': randint, 'dist_params': {'low':100, 'high':1000} },
            'wordNgrams': {'dist': randint, 'dist_params': {'low':1, 'high':10}},
            'ws': {'dist': randint, 'dist_params': {'low':1, 'high':10}}
            }

    do_random_search(dists, n_iter=10)
