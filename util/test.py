from multiprocessing import Process, Manager
import time
import numpy as np
import scipy.signal

def f(image, random_filter):
    # Do some image processing.
    return scipy.signal.convolve2d(image, random_filter)[::5, ::5]

image = np.zeros((3000, 3000))

filters = [np.random.normal(size=(4, 4)) for i in range(2)]

if __name__ == '__main__':
    manager = Manager()

    d = manager.dict()
    d[1] = '1'
    d['2'] = 2

    start = time.time()
    processes = []
    for i in range(2):
        start = time.time()
        p = Process(target=f, args=(image, filters[i],))
        processes.append(p)
        p.start()
        print(time.time()-start)

    for process in processes: process.join()
    print(time.time()-start)

    print(dict(d))
