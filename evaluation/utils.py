import queue
import threading
from collections.abc import MutableMapping, Sequence

import torch
from torch.utils.data import DataLoader


def to_cuda(packed_data):
    if isinstance(packed_data, torch.Tensor):
        packed_data = packed_data.to(device="cuda", non_blocking=True)
    elif isinstance(packed_data, (int, float, str, bool, complex)):
        packed_data = packed_data
    elif isinstance(packed_data, MutableMapping):
        for key, value in packed_data.items():
            packed_data[key] = to_cuda(value)
    elif isinstance(packed_data, Sequence):
        for i, value in enumerate(packed_data):
            packed_data[i] = to_cuda(value)
    return packed_data


# class CUDADataLoader(DataLoader):

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.stream = torch.cuda.Stream() # create a new cuda stream in each process
#         self.queue = queue.Queue(64)

#     def preload(self):
#         batch = next(self.iter)
#         if batch is None:
#             return None
#         torch.cuda.current_stream().wait_stream(self.stream)  # wait tensor to put on GPU
#         with torch.cuda.stream(self.stream):
#             batch = to_cuda(batch)
#         self.queue.put(batch)

#     def __iter__(self):
#         # setting a queue for storing prefetched data
#         self.queue.queue.clear()
#         # reset data iterator
#         self.iter = super().__iter__()
#         # starting a new thread to prefetch data
#         def data_to_cuda_then_queue():
#             while True:
#                 try:
#                     self.preload()
#                 except StopIteration:
#                     break
#             # NOTE: end flag for the queue
#             self.queue.put(None)

#         self.thread = threading.Thread(target=data_to_cuda_then_queue, args=())
#         self.thread.daemon = True

#         (self.preload() for _ in range(16))
#         self.thread.start()
#         return self

#     def __next__(self):
#         next_item = self.queue.get()
#         # NOTE: __iter__ will be stopped when __next__ raises StopIteration 
#         if next_item is None:
#             raise StopIteration
#         return next_item

#     def __del__(self):
#         # NOTE: clean up the thread
#         try:
#             self.thread.join(timeout=10)
#         finally:
#             if self.thread.is_alive():
#                 self.thread._stop()
#         # NOTE: clean up the stream
#         self.stream.synchronize()
#         # NOTE: clean up the queue
#         self.queue.queue.clear()

class CUDADataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        # strongly recommended: pin_memory=True, persistent_workers=True
        super().__init__(*args, **kwargs)
        self._stream = torch.cuda.Stream()

    def __iter__(self):
        self._iter = super().__iter__()
        self._next = None
        self._preload()
        return self

    def _to_cuda(self, batch):
        # ensure non_blocking=True; write your own tree-walk if needed
        def move(x):
            if torch.is_tensor(x):
                return x.cuda(non_blocking=True)
            elif isinstance(x, dict):
                return {k: move(v) for k, v in x.items()}
            elif isinstance(x, (list, tuple)):
                t = [move(v) for v in x]
                return type(x)(t) if isinstance(x, tuple) else t
            return x
        return move(batch)

    def _record_on(self, batch, stream):
        def rec(x):
            if torch.is_tensor(x):
                x.record_stream(stream)
            elif isinstance(x, dict):
                for v in x.values(): rec(v)
            elif isinstance(x, (list, tuple)):
                for v in x: rec(v)
        rec(batch)

    def _preload(self):
        try:
            batch = next(self._iter)
        except StopIteration:
            self._next = None
            return
        # move to GPU on the prefetch stream
        with torch.cuda.stream(self._stream):
            batch = self._to_cuda(batch)
        self._next = batch

    def __next__(self):
        if self._next is None:
            raise StopIteration
        # **CRITICAL**: make current (consumer) stream wait for prefetch stream
        torch.cuda.current_stream().wait_stream(self._stream)
        batch = self._next
        # prevent allocator reuse while batch is in-flight on current stream
        self._record_on(batch, torch.cuda.current_stream())
        # kick off the next transfer
        self._preload()
        return batch
