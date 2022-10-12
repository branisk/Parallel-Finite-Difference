"""
This file demonstrates sub-processes and how to communicate with them.

"""
import multiprocessing as mp


def add(x, queue):

    while True:
        q = queue.get()
        if type(q) == str and q =='Done':
            print('terminating ...')
            break
        else:
            # print(f'add = {x+q}', file=sys.stderr)
            print(f'add: {x}+{q}')



# It seems that on Mac/Windows, each sub-process imports the 
# full python script. The following prevents problems with 
# recursive sub-processes.
if __name__ == '__main__':

    queue = mp.Queue()
    process = mp.Process(target=add, args=(10, queue))
    process.start()
    
    # To observe the timing between the processes go to 
    # IPython and run
    #
    #     from process import *
    #
    # and then the three lines above. Afterwards, type in
    # the queue.put statements below and observe when
    # the sub-process handels the messages.

    queue.put(4)
    print('put finished')
    queue.put(5)
    print('put finished')
    queue.put('Done')
    print('put finished')
    
    process.join()
