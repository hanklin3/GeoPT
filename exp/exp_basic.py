import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, CPUOffload, BackwardPrefetch
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from functools import partial
from models.model_factory import get_model
from data_provider.data_factory import get_data


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.use_fsdp = bool(getattr(args, 'use_fsdp', 0))
        self.distributed = False
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._setup_runtime()
        self.dataset, self.train_loader, self.test_loader, args.shapelist = get_data(args)
        _, _, self.test_loader_full, _ = get_data(args, full_mesh=True)
        if self.distributed:
            self.train_loader = self._to_distributed_loader(self.train_loader, shuffle=True)

        base_model = get_model(args).to(self.device)
        if self.use_fsdp:
            self.model = FSDP(
                base_model,
                device_id=self.local_rank,
                sync_module_states=True,
                sharding_strategy=self._get_sharding_strategy(),
                mixed_precision=self._get_mixed_precision_policy(),
                cpu_offload=CPUOffload(offload_params=bool(getattr(self.args, 'fsdp_cpu_offload', 0))),
                auto_wrap_policy=self._get_auto_wrap_policy(),
                limit_all_gathers=bool(getattr(self.args, 'fsdp_limit_all_gathers', 1)),
                backward_prefetch=self._get_backward_prefetch(),
            )
        else:
            self.model = base_model

        if self.is_main_process:
            print(self.args)
            print(self.model)
            count_parameters(self.unwrap_model())

    @property
    def is_main_process(self):
        return self.rank == 0

    def _setup_runtime(self):
        if self.use_fsdp:
            if not torch.cuda.is_available():
                raise RuntimeError('FSDP requires CUDA.')
            required_env = ('RANK', 'WORLD_SIZE', 'LOCAL_RANK')
            if not all(k in os.environ for k in required_env):
                raise RuntimeError('FSDP requires torchrun. Missing one of RANK/WORLD_SIZE/LOCAL_RANK.')

            if not dist.is_initialized():
                dist.init_process_group(backend='nccl')

            self.distributed = True
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.local_rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f'cuda:{self.local_rank}')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')

    def _to_distributed_loader(self, loader, shuffle):
        sampler = DistributedSampler(
            loader.dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle,
            drop_last=loader.drop_last,
        )
        loader_kwargs = {
            'dataset': loader.dataset,
            'batch_size': loader.batch_size,
            'sampler': sampler,
            'shuffle': False,
            'num_workers': loader.num_workers,
            'collate_fn': loader.collate_fn,
            'pin_memory': loader.pin_memory,
            'drop_last': loader.drop_last,
            'timeout': loader.timeout,
            'worker_init_fn': loader.worker_init_fn,
            'persistent_workers': loader.persistent_workers,
        }
        if loader.num_workers > 0 and loader.prefetch_factor is not None:
            loader_kwargs['prefetch_factor'] = loader.prefetch_factor
        return DataLoader(**loader_kwargs)

    def _get_sharding_strategy(self):
        strategy = str(getattr(self.args, 'fsdp_sharding_strategy', 'full_shard')).lower()
        if strategy == 'full_shard':
            return ShardingStrategy.FULL_SHARD
        if strategy == 'shard_grad_op':
            return ShardingStrategy.SHARD_GRAD_OP
        if strategy == 'no_shard':
            return ShardingStrategy.NO_SHARD
        raise ValueError(f'Unsupported fsdp_sharding_strategy: {strategy}')

    def _get_mixed_precision_policy(self):
        mp = str(getattr(self.args, 'fsdp_mixed_precision', 'none')).lower()
        if mp == 'none':
            return None
        if mp == 'fp16':
            dtype = torch.float16
        elif mp == 'bf16':
            dtype = torch.bfloat16
        else:
            raise ValueError(f'Unsupported fsdp_mixed_precision: {mp}')
        return MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)

    def _get_auto_wrap_policy(self):
        min_params = int(getattr(self.args, 'fsdp_min_params', 0))
        if min_params <= 0:
            return None
        return partial(size_based_auto_wrap_policy, min_num_params=min_params)

    def _get_backward_prefetch(self):
        prefetch = str(getattr(self.args, 'fsdp_backward_prefetch', 'pre')).lower()
        if prefetch == 'pre':
            return BackwardPrefetch.BACKWARD_PRE
        if prefetch == 'post':
            return BackwardPrefetch.BACKWARD_POST
        if prefetch == 'none':
            return None
        raise ValueError(f'Unsupported fsdp_backward_prefetch: {prefetch}')

    def set_epoch(self, epoch):
        if self.distributed and isinstance(self.train_loader.sampler, DistributedSampler):
            self.train_loader.sampler.set_epoch(epoch)

    def unwrap_model(self):
        return self.model.module if isinstance(self.model, FSDP) else self.model

    def save_checkpoint(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if self.use_fsdp:
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, cfg):
                state_dict = self.model.state_dict()
            if self.is_main_process:
                torch.save(state_dict, path)
        else:
            torch.save(self.model.state_dict(), path)

        if self.distributed:
            dist.barrier()

    def load_checkpoint(self, path):
        checkpoint_path = os.path.abspath(path)
        if self.use_fsdp:
            state_dict = None
            if self.is_main_process:
                state_dict = torch.load(path, map_location='cpu')
            if self.distributed:
                buffer = [state_dict]
                dist.broadcast_object_list(buffer, src=0)
                state_dict = buffer[0]

            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, cfg):
                self.model.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(torch.load(path))

        if self.is_main_process:
            model_name = getattr(self.unwrap_model(), '__name__', self.unwrap_model().__class__.__name__)
            print(f"Loaded checkpoint: {checkpoint_path}")
            print(f"Loaded model: {model_name} (save_name={self.args.save_name}, device={self.device})")

        if self.distributed:
            dist.barrier()

    def finalize(self):
        if self.distributed and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

    def _gpu_memory_snapshot(self):
        if not torch.cuda.is_available():
            return []

        snapshots = []
        for i in range(torch.cuda.device_count()):
            snapshots.append({
                'gpu': i,
                'name': torch.cuda.get_device_name(i),
                'allocated_gb': torch.cuda.memory_allocated(i) / (1024 ** 3),
                'reserved_gb': torch.cuda.memory_reserved(i) / (1024 ** 3),
                'peak_allocated_gb': torch.cuda.max_memory_allocated(i) / (1024 ** 3),
                'peak_reserved_gb': torch.cuda.max_memory_reserved(i) / (1024 ** 3),
            })
        return snapshots

    def report_gpu_memory(self, tag=''):
        if not torch.cuda.is_available():
            return

        snapshots = self._gpu_memory_snapshot()
        label = f'[{tag}] ' if tag else ''
        if self.distributed:
            payload = {'rank': self.rank, 'snapshots': snapshots}
            gathered = [None for _ in range(self.world_size)] if self.is_main_process else None
            dist.gather_object(payload, gathered, dst=0)
            if self.is_main_process:
                print(f'{label}GPU memory report:')
                for item in sorted(gathered, key=lambda x: x['rank']):
                    for s in item['snapshots']:
                        print(
                            f"  rank {item['rank']} gpu {s['gpu']} ({s['name']}): "
                            f"alloc={s['allocated_gb']:.2f}G reserv={s['reserved_gb']:.2f}G "
                            f"peak_alloc={s['peak_allocated_gb']:.2f}G peak_reserv={s['peak_reserved_gb']:.2f}G"
                        )
        else:
            print(f'{label}GPU memory report:')
            for s in snapshots:
                print(
                    f"  gpu {s['gpu']} ({s['name']}): "
                    f"alloc={s['allocated_gb']:.2f}G reserv={s['reserved_gb']:.2f}G "
                    f"peak_alloc={s['peak_allocated_gb']:.2f}G peak_reserv={s['peak_reserved_gb']:.2f}G"
                )

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
