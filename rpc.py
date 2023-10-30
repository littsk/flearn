import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp

import os


def rpc_test(rank, size):
    rpc.init_rpc(f"worker{rank}", rank=rank, world_size=size)
    if rank == 0:
        ret = rpc.rpc_sync(f"worker1", torch.add, args=(torch.ones(2), 3))
        print(f"rank{rank}: {ret}")
    rpc.shutdown()

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '16388'
    dist.init_process_group(backend, rank=rank, world_size=size)
    torch.cuda.set_device(rank)
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, rpc_test, "gloo"))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
