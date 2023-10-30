import os
import torch
import torch.distributed as dist
from torch.distributed import get_world_size, get_rank
import torch.multiprocessing as mp

def run_gather(rank, size):
    tensor = torch.tensor([rank])
    if get_rank() == 0:
        gather_list = [torch.empty_like(tensor) for _ in range(size)]
    else:
        gather_list = None
    dist.gather(tensor=tensor, gather_list=gather_list, dst=0)
    print('Rank ', rank, ' has data ', gather_list)


def run_scatter(rank, size):
    tensor = torch.empty(1, dtype=torch.float)
    if get_rank() == 0:
        scatter_list = [torch.tensor([_], dtype=torch.float) for _ in range(size)]
    else:
        scatter_list = None
    dist.scatter(tensor=tensor, scatter_list=scatter_list, src=0)
    print('Rank ', rank, ' has data ', tensor)

def run_broadcast(rank, size):
    tensor = torch.tensor([rank]).cuda()
    dist.broadcast(tensor=tensor, src=0)
    print('Rank ', rank, ' has data ', tensor)

def my_allreduce(send, recv):
    rank = dist.get_rank()
    size = dist.get_world_size()

    # cal_index = list(range(size -1, -1, -1))
    # shift = size - rank - 1
    # cal_index = cal_index[shift:] + cal_index[:shift]

    # left = ((rank - 1) + size) % size
    # right = (rank + 1) % size

    # recv_buf = send.clone()

    # for idx, t in enumerate(cal_index):
    #     send_buf = recv_buf.clone()

    #     if rank % 2 == 0:
    #         send_work = dist.isend(send_buf, right)
    #     else:
 
    
    # 初始化 recv 缓冲区和累积器
    recv_buff = send.clone()
    accum = torch.zeros_like(send)

    left = ((rank - 1) + size) % size
    right = (rank + 1) % size

    for i in range(size - 1):
        send_buff = recv_buff.clone()
        
        if rank % 2 == 0:
            # 非阻塞发送到右边
            send_req = dist.isend(send_buff, right)
        else:
            # 非阻塞接收从左边
            recv_req = dist.irecv(recv_buff, left)

        if rank % 2 == 0:
            recv_req = dist.irecv(recv_buff, left)
        else:
            send_req = dist.isend(send_buff, right)

        send_req.wait()
        recv_req.wait()

        accum += recv_buff

    # 将自己的数据加入累加器
    accum += send

    # 复制最终结果到接收缓冲区
    recv[:] = accum[:]

    #     if rank == 0:
    #     send_req = dist.isend(send_buff, right)
    # if rank == 1:
    #     dist.recv(recv_buff, left)
    # dist.barrier()
    # recv[:] = recv_buff[:]

def b_send_recv(rank, size):
    send_tensor = torch.zeros([2, 2]) + rank
    recv_tensor = torch.zeros_like(send_tensor)
    send_tensor = send_tensor.cuda()
    recv_tensor = recv_tensor.cuda()

    right = (rank + 1) % size
    left = (rank - 1 + size) % size

    ops = [] 

    send_op = dist.P2POp(dist.isend, send_tensor, right)
    recv_op = dist.P2POp(dist.irecv, recv_tensor, left)

    ops.extend([send_op, recv_op])
    # if rank % 2 == 0:
    #     ops = ops[::-1]
    
    reqs = dist.batch_isend_irecv(ops)

    for req in reqs:
        req.wait()

    print('Rank ', rank, ' has data ', send_tensor, recv_tensor)

def run_my_allreduce(rank, size):
    send = torch.tensor([rank]).cuda()
    recv = torch.empty_like(send).cuda()
    my_allreduce(send, recv)
    print('Rank ', rank, ' has data ', send, recv)


def run(*args, **kwags):
    b_send_recv(*args, **kwags)


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
        p = mp.Process(target=init_process, args=(rank, size, run, "nccl"))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


