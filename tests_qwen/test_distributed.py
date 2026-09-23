"""Two real CPU/Gloo processes exercise DDP, exact scans and adapter resume."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from qwen_smope.data import ExactShard
from qwen_smope.train import (correct_classifier, evaluate, optimizer_for, restore_rng,
                              save_checkpoint, scan, train_epoch)
from test_qwen import tiny_inputs, tiny_model


class TinyData(Dataset):
    def __len__(self):
        return 7

    def __getitem__(self, i):
        with torch.random.fork_rng():
            torch.manual_seed(i)
            return tiny_inputs(), torch.tensor([i % 2])


def collate_one(items):
    return items[0]


def worker(rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=(Path(directory) / "rendezvous").as_uri(), rank=rank, world_size=2)
    try:
        torch.manual_seed(9)
        model = tiny_model()
        model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        network = DDP(model, broadcast_buffers=False)
        args = SimpleNamespace(head_lr=.001, prompt_lr=.001, weight_decay=0., accumulation=3,
                               batch_size=1, clip_grad=1., log_every=1, output=directory,
                               correction_epochs=1, correction_lr=.001, prototype_samples=2)
        data = TinyData()
        loader = DataLoader(data, batch_size=1, sampler=DistributedSampler(data, num_replicas=2, rank=rank), collate_fn=collate_one)
        optimizer = optimizer_for(model, args)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2)
        # Partial accumulation window + unused expert slots + checkpoint recomputation.
        result = train_epoch(network, model, loader, optimizer, scheduler, args, torch.device("cpu"),
                             0, 2, False, 0, 0, "train", rank, 2)
        assert result["samples_including_distributed_padding"] == 8
        for p in model.parameters():
            if p.requires_grad:
                reference = p.detach().clone()
                dist.broadcast(reference, 0)
                torch.testing.assert_close(p, reference)
        exact = DataLoader(data, batch_size=1, sampler=ExactShard(data, rank, 2), collate_fn=collate_one)
        prototypes = {}
        scan(model, exact, torch.device("cpu"), 6, 0, 2, prototypes)
        for expert in model.expert_modules():
            assert int(expert.frequency.sum()) == len(data) * 4 * 2
        assert set(prototypes) == {0, 1}
        correct_classifier(model, prototypes, 2, args, torch.device("cpu"), rank)
        row, confusion, totals, correct = evaluate(model, exact, torch.device("cpu"), 6, 2, 2)
        assert int(confusion.sum()) == 7
        assert int(totals[0]) == 4 and int(totals[1]) == 3
        for expert in model.expert_modules():
            expert.next_task()
        path = Path(directory) / "latest_adapter.pt"
        save_checkpoint(path, model, dict(task=1, phase="dense", epoch=0), optimizer, scheduler,
                        [row], prototypes, {"test": True}, rank, 2)
        saved = torch.load(path, weights_only=False)
        restore_rng(saved["rng"][rank])
        model.load_lightweight_state(saved["adapter"])
        assert all("expert" in k or k.startswith("classifier") for k in saved["adapter"])
        # Task 2 verifies old-expert regularization participates in DDP backward.
        train_epoch(network, model, loader, optimizer, scheduler, args, torch.device("cpu"),
                    0, 2, False, 1, 0, "train", rank, 2)
    finally:
        dist.destroy_process_group()


class DistributedTests(unittest.TestCase):
    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo not available")
    def test_two_rank_train_scan_evaluate_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(worker, args=(directory,), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
