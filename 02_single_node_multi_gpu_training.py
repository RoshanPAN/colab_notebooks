from random import Random

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # print(x.size())  # torch.Size([128, 1, 28, 28])
        x = self.conv1(x)
        # print(x.size()) # torch.Size([128, 10, 24, 24])
        x = F.relu(F.max_pool2d(x, 2))
        # print(x.size()) # torch.Size([128, 10, 12, 12])
        x = self.conv2(x)
        # print(x.size()) # torch.Size([128, 20, 8, 8])
        x = self.conv2_drop(x)
        x = F.max_pool2d(x, 2)
        # print(x.size()) # torch.Size([128, 20, 4, 4])
        x = F.relu(x)
        # print(x.size())
        x = x.view(-1, 320)  # torch.Size([128, 320])
        # print(x.size())
        x = F.relu(self.fc1(x))
        # print(x.size()) # torch.Size([128, 50])
        x = F.dropout(x, training=self.training)
        # print(x.size())
        x = self.fc2(x)
        # print(x.size()) # torch.Size([128, 10])
        x = F.log_softmax(x, dim=1)
        # print(x.size()) # torch.Size([128, 10])
        return x


import math, os
from typing import Tuple


WORLD_SIZE = 8
BSZ = 256 * WORLD_SIZE
EPOCH = 30 * WORLD_SIZE


class Partition(object):
    """Dataset-like object, but only access a subset of it."""

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """Partitions a dataset into different chuncks."""

    # TODO: the data partition here loads the whole copy per process
    def __init__(self, data, sizes, seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = list(range(data_len))
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


def process_dataset() -> Tuple[torch.utils.data.DataLoader, int]:
    dataset = datasets.MNIST(
        "./data",
        train=True,  # creates dataset from train-images-idx3-ubyte
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=(0.1307,), std=(0.3081,))]
        ),
    )  # 60000 x 28 x 28
    world_size, rank = dist.get_world_size(), dist.get_rank()
    bsz = BSZ // world_size
    partitioner = DataPartitioner(
        dataset,
        sizes=([1.0 / world_size] * world_size),
    )
    partitioned_dataset = partitioner.use(rank)
    train_set = torch.utils.data.DataLoader(
        partitioned_dataset,
        batch_size=bsz,
        shuffle=True,
    )
    return train_set, bsz


def gradient_averaging(model):
    world_size = dist.get_world_size()
    for param in model.parameters():
        if type(param) is torch.Tensor:
            dist.all_reduce(param.gradient.data, op=dist.ReduceOp.SUM, group=0)
            param.gradient.data /= world_size


def run(rank: int):
    world_size = dist.get_world_size()  # noqa
    prefix = f"[rank: {rank}] {torch.cuda.is_available()}"
    # 0) dataset
    train_set, bsz = process_dataset()
    print(
        f"{prefix} Dataset size: {len(train_set.dataset)}, batch size {train_set.batch_size}"
    )
    # 1) nn
    torch.manual_seed(3)
    model = Net().cuda(rank)
    print(f"{prefix} Net initialized")
    # 3) optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    # 4) training loop
    num_batches = math.ceil(len(train_set.dataset) / float(bsz))
    print(f"{prefix} num_batches: {num_batches}")
    for epoch in range(EPOCH):
        epoch_loss = 0.0
        for data, target in train_set:
            data, target = Variable(data), Variable(target)
            data, target = Variable(data.cuda(rank)), Variable(target.cuda(rank))
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss
            loss.backward()
            gradient_averaging(model)
            optimizer.step()
        print(f"{prefix} epoch {epoch} : {epoch_loss / num_batches}")


def init_processes(rank, size, fn, backend):
    """Initialize the distributed environment"""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    print(f"[Rank {rank}]")
    dist.init_process_group(backend, rank=rank, world_size=size)
    # assert torch.distributed.is_initialized()
    fn(rank)


if __name__ == "__main__":
    processes = []
    mp.set_start_method("spawn", force=True)

    for rank in range(0, WORLD_SIZE):
        p = mp.Process(
            target=init_processes,
            args=(rank, WORLD_SIZE, run, "gloo"),
            name=f"[rank {rank}]",
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

"""
[Rank 4]
[Rank 0]
[Rank 2]
[Rank 7]
[Rank 3]
[Rank 6]
[Rank 1]
[Rank 5]
[rank: 0] True Dataset size: 7500, batch size 256
[rank: 2] True Dataset size: 7500, batch size 256
[rank: 1] True Dataset size: 7500, batch size 256
[rank: 5] True Dataset size: 7500, batch size 256
[rank: 7] True Dataset size: 7500, batch size 256
[rank: 3] True Dataset size: 7500, batch size 256
[rank: 6] True Dataset size: 7500, batch size 256
[rank: 4] True Dataset size: 7500, batch size 256
INFO:2023-02-11 23:47:52 255731:255731 CuptiActivityProfiler.cpp:167] CUDA versions. CUPTI: 14; Runtime: 11040; Driver: 11040
INFO:2023-02-11 23:47:52 255731:255731 DynoConfigLoader.cpp:61] Setting communication fabric enabled = 0
INFO:2023-02-11 23:47:52 255731:255731 IpcFabricConfigClient.cpp:93] Setting up IPC Fabric at endpoint: dynoconfigclient6f2c407c-b260-4f9c-a45c-ea2a8e184149 status = failed (null)
INFO:2023-02-11 23:47:52 255721:255721 CuptiActivityProfiler.cpp:167] CUDA versions. CUPTI: 14; Runtime: 11040; Driver: 11040
INFO:2023-02-11 23:47:52 255721:255721 DynoConfigLoader.cpp:61] Setting communication fabric enabled = 0
INFO:2023-02-11 23:47:52 255721:255721 IpcFabricConfigClient.cpp:93] Setting up IPC Fabric at endpoint: dynoconfigclient08711484-e3db-4fef-a00e-8b627b123dc7 status = failed (null)
INFO:2023-02-11 23:47:52 255730:255730 CuptiActivityProfiler.cpp:167] CUDA versions. CUPTI: 14; Runtime: 11040; Driver: 11040
INFO:2023-02-11 23:47:52 255730:255730 DynoConfigLoader.cpp:61] Setting communication fabric enabled = 0
INFO:2023-02-11 23:47:52 255730:255730 IpcFabricConfigClient.cpp:93] Setting up IPC Fabric at endpoint: dynoconfigclient3360b517-52d1-442d-b178-11ded8b7b153 status = failed (null)
INFO:2023-02-11 23:47:53 255799:255799 CuptiActivityProfiler.cpp:167] CUDA versions. CUPTI: 14; Runtime: 11040; Driver: 11040
INFO:2023-02-11 23:47:53 255799:255799 DynoConfigLoader.cpp:61] Setting communication fabric enabled = 0
INFO:2023-02-11 23:47:53 255799:255799 IpcFabricConfigClient.cpp:93] Setting up IPC Fabric at endpoint: dynoconfigclientc91b3bce-4811-406a-88c4-cc9c381f675d status = failed (null)
INFO:2023-02-11 23:47:53 255821:255821 CuptiActivityProfiler.cpp:167] CUDA versions. CUPTI: 14; Runtime: 11040; Driver: 11040
INFO:2023-02-11 23:47:53 255821:255821 DynoConfigLoader.cpp:61] Setting communication fabric enabled = 0
INFO:2023-02-11 23:47:53 255821:255821 IpcFabricConfigClient.cpp:93] Setting up IPC Fabric at endpoint: dynoconfigclientefe9e5aa-ad89-49f9-af14-a5d211bbb399 status = failed (null)
INFO:2023-02-11 23:47:53 255733:255733 CuptiActivityProfiler.cpp:167] CUDA versions. CUPTI: 14; Runtime: 11040; Driver: 11040
INFO:2023-02-11 23:47:53 255733:255733 DynoConfigLoader.cpp:61] Setting communication fabric enabled = 0
INFO:2023-02-11 23:47:53 255733:255733 IpcFabricConfigClient.cpp:93] Setting up IPC Fabric at endpoint: dynoconfigclient41423d16-a111-431e-97d1-39e5581f483c status = failed (null)
INFO:2023-02-11 23:47:53 255734:255734 CuptiActivityProfiler.cpp:167] CUDA versions. CUPTI: 14; Runtime: 11040; Driver: 11040
INFO:2023-02-11 23:47:53 255811:255811 CuptiActivityProfiler.cpp:167] CUDA versions. CUPTI: 14; Runtime: 11040; Driver: 11040
INFO:2023-02-11 23:47:53 255734:255734 DynoConfigLoader.cpp:61] Setting communication fabric enabled = 0
INFO:2023-02-11 23:47:53 255811:255811 DynoConfigLoader.cpp:61] Setting communication fabric enabled = 0
INFO:2023-02-11 23:47:53 255734:255734 IpcFabricConfigClient.cpp:93] Setting up IPC Fabric at endpoint: dynoconfigclient5e5c1579-8a74-417b-a580-c5d492d4f3dd status = failed (null)
INFO:2023-02-11 23:47:53 255811:255811 IpcFabricConfigClient.cpp:93] Setting up IPC Fabric at endpoint: dynoconfigclient74313fe6-5b92-4914-add1-c0c34ae5c59d status = failed (null)
INFO:2023-02-11 23:47:53 255731:255731 DynoConfigLoader.cpp:61] Setting communication fabric enabled = 0
INFO:2023-02-11 23:47:53 255721:255721 DynoConfigLoader.cpp:61] Setting communication fabric enabled = 0
INFO:2023-02-11 23:47:53 255730:255730 DynoConfigLoader.cpp:61] Setting communication fabric enabled = 0
INFO:2023-02-11 23:47:53 255799:255799 DynoConfigLoader.cpp:61] Setting communication fabric enabled = 0
INFO:2023-02-11 23:47:53 255733:255733 DynoConfigLoader.cpp:61] Setting communication fabric enabled = 0
INFO:2023-02-11 23:47:53 255821:255821 DynoConfigLoader.cpp:61] Setting communication fabric enabled = 0
INFO:2023-02-11 23:47:53 255734:255734 DynoConfigLoader.cpp:61] Setting communication fabric enabled = 0
INFO:2023-02-11 23:47:53 255811:255811 DynoConfigLoader.cpp:61] Setting communication fabric enabled = 0
[rank: 2] True Net initialized
[rank: 2] True num_batches: 30
[rank: 0] True Net initialized
[rank: 0] True num_batches: 30
[rank: 1] True Net initialized
[rank: 1] True num_batches: 30
[rank: 5] True Net initialized
[rank: 5] True num_batches: 30
[rank: 3] True Net initialized
[rank: 3] True num_batches: 30
[rank: 7] True Net initialized
[rank: 7] True num_batches: 30
[rank: 4] True Net initialized
[rank: 4] True num_batches: 30
[rank: 6] True Net initialized
[rank: 6] True num_batches: 30
[rank: 0] True epoch 0 : 2.2814841270446777
[rank: 3] True epoch 0 : 2.2846927642822266
[rank: 1] True epoch 0 : 2.2866501808166504
[rank: 2] True epoch 0 : 2.284261465072632
[rank: 5] True epoch 0 : 2.282500982284546
...
[rank: 7] True epoch 235 : 0.14042983949184418
[rank: 2] True epoch 239 : 0.14255361258983612
[rank: 5] True epoch 239 : 0.1458708941936493
[rank: 3] True epoch 237 : 0.14031003415584564
[rank: 0] True epoch 237 : 0.14030063152313232
[rank: 1] True epoch 238 : 0.13560369610786438
[rank: 6] True epoch 238 : 0.14736245572566986
[rank: 4] True epoch 239 : 0.13152943551540375
[rank: 7] True epoch 236 : 0.1417667120695114
[rank: 3] True epoch 238 : 0.14096489548683167
[rank: 0] True epoch 238 : 0.13335058093070984
[rank: 1] True epoch 239 : 0.14195725321769714
[rank: 6] True epoch 239 : 0.13597626984119415
[rank: 7] True epoch 237 : 0.13710376620292664
[rank: 3] True epoch 239 : 0.14444339275360107
[rank: 0] True epoch 239 : 0.13454923033714294
[rank: 7] True epoch 238 : 0.12891024351119995
[rank: 7] True epoch 239 : 0.13575246930122375
"""
