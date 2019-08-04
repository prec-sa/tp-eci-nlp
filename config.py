from multiprocessing import cpu_count


TRAIN_THREADS = (4 if cpu_count() == 2 else 12)
