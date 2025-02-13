import multiprocessing
import time

def worker():
    name = multiprocessing.current_process().name
    print(name, 'Starting')
    time.sleep(2)
    print(name, 'Exiting')
    
def my_service():
    name = multiprocessing.current_process().name
    print(name, 'Starting')
    time.sleep(3)
    print(name, 'Exiting')
    
if __name__ == '__main__':

    # Create 3 processes 
    service = multiprocessing.Process(name='my_service', target=my_service)
    worker_1 = multiprocessing.Process(name='worker 1', target=worker)
    worker_2 = multiprocessing.Process(target=worker) # use default name

    # Start 3 processes
    worker_1.start()
    worker_2.start()
    service.start()

    #Use join() to make the processes run and complete before any main program process,
    #which is the fourth process in this case
    worker_1.join()
    worker_2.join()
    service.join()

