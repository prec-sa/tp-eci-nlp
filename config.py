from multiprocessing import cpu_count


TRAIN_THREADS = cpu_count() #(4 if cpu_count() == 4 else 12)
