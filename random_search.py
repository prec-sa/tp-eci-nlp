from grid_search_utils import generate_single_score
import gc
from scipy.stats import randint


def parameters_from_dists(param_dists, fixed_params = {}):

    params = {param: value['dist'](**value['dist_params']).rvs() for \
              param, value in param_dists.items() }

    return {**params, **fixed_params}


def do_random_search(param_dists, n_iter, fixed_params={}):

    with open('random_results_frame.csv', 'a') as results:

        for i in range(n_iter):

            print('\n Iteration ', i, ' of ', n_iter, '\n')

            params = parameters_from_dists(param_dists, fixed_params)

            res = generate_single_score(params)
            print(res)
            
            acc = res[1]
            
            out = str(params['dim'])+','+str(params['wordNgrams'])+','+str(params['ws'])+','+str(acc)+'\n'

            results.write(out)
            results.flush()
            gc.collect()


if __name__ == '__main__':

    dists = {
            'dim': {'dist': randint, 'dist_params': {'low':100, 'high':1000} },
            'wordNgrams': {'dist': randint, 'dist_params': {'low':1, 'high':10}},
            #'wordNgrams': {'dist': randint, 'dist_params': {'low':2, 'high':2}},
            'ws': {'dist': randint, 'dist_params': {'low':1, 'high':10}}
            }

    do_random_search(dists, n_iter=10, fixed_params={'wordNgrams':2})
