import os
import math
import numpy as np
import torch
from typing import List, Generator, Union, Optional, cast, Tuple, Dict
from config import config

class DataLoader:
    """
    DataLoader for SAE training. Loads token shards, splits them into sequences,
    and yields batches across shards. When a shard doesn't split evenly by
    batch_size, the remainder is carried over and filled from the next shard.
    """
    def __init__(self, device: torch.device, skip_first_token: bool = True, pin_memory: bool = False):
        self.data_path = cast(str, config.data.dataset_path)
        self.batch_size = cast(int, config.data.batch_size)
        self.skip_first_token = skip_first_token
        self.pin_memory = pin_memory and device.type == "cuda"
        self.device = device
        all_shards = self._get_shard_files()
        n_shards = cast(int, config.data.n_shards)
        self.shard_files = all_shards[:n_shards]
        self.shard_id_ranges: List[Tuple[int, int]] = []
        self._shard_indices: List[np.ndarray] = []
        self._shard_sequence_counts = self._load_sequence_counts()

    def _get_shard_files(self) -> List[str]:
        """Lists and sorts all .npy shard files in the data directory."""
        if not os.path.exists(self.data_path):
            print(f"Warning: Data path {self.data_path} does not exist.")
            return []
        shards = [f for f in os.listdir(self.data_path) if f.endswith(".npy")]
        shards.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        return shards

    def _load_sequence_counts(self) -> List[int]:
        """Load or build per-shard sequence indices and store global ID ranges."""
        counts: List[int] = []
        self.shard_id_ranges = []
        needs_build = [
            i for i in range(len(self.shard_files))
            if not os.path.exists(self._get_index_path(i))
        ]
        if needs_build:
            print(f"Building shard indices ({len(needs_build)} shard(s))...")
        current_id = 1
        for shard_index in range(len(self.shard_files)):
            index = self._load_or_build_index(shard_index)
            self._shard_indices.append(index)
            count = len(index)
            counts.append(count)
            if count > 0:
                self.shard_id_ranges.append((current_id, current_id + count - 1))
                current_id += count
            else:
                self.shard_id_ranges.append((-1, -1))
        return counts

    def _get_index_path(self, shard_index: int) -> str:
        """Returns the path to the cached index file for a given shard."""
        index_dir = os.path.join(self.data_path, ".shard_indices")
        shard_file = self.shard_files[shard_index]
        suffix = f"_sft{int(self.skip_first_token)}.idx.npy"
        return os.path.join(index_dir, shard_file + suffix)

    def _build_shard_index(self, shard_index: int) -> np.ndarray:
        """
        Scans a shard once to record the (start, end) byte positions of every
        valid cleaned sequence, then saves the result as a .idx.npy cache file.

        The stored positions are ready-to-use: shard[start:end] gives the final
        token array with no further processing needed (separators and the optional
        first token are already excluded). Shape: (n_valid_seqs, 2), dtype int64.
        """
        shard_path = os.path.join(self.data_path, self.shard_files[shard_index])
        shard = np.load(shard_path, mmap_mode="r")

        if len(shard) == 0:
            index = np.zeros((0, 2), dtype=np.int64)
        else:
            sep_positions = np.where(shard == -1)[0]
            skip = 1 if self.skip_first_token else 0

            # Segment boundaries: each segment lives between two -1 separators.
            # seg_starts_raw already skips the -1 itself (+1); skip_first_token
            # advances start by one more.
            seg_starts_raw = np.concatenate([[0], sep_positions + 1])
            seg_ends_raw   = np.concatenate([sep_positions, [len(shard)]])

            starts = seg_starts_raw + skip
            ends   = seg_ends_raw
            valid  = (ends - starts) > 1
            index  = np.stack([starts[valid], ends[valid]], axis=1).astype(np.int64)

        index_dir = os.path.dirname(self._get_index_path(shard_index))
        os.makedirs(index_dir, exist_ok=True)
        np.save(self._get_index_path(shard_index), index)
        return index

    def _load_or_build_index(self, shard_index: int) -> np.ndarray:
        """Returns the cached index, rebuilding it if missing or stale."""
        index_path = self._get_index_path(shard_index)
        shard_path = os.path.join(self.data_path, self.shard_files[shard_index])
        if (
            os.path.exists(index_path)
            and os.path.getmtime(index_path) >= os.path.getmtime(shard_path)
        ):
            return np.load(index_path)
        return self._build_shard_index(shard_index)

    def __len__(self) -> int:
        """Returns the total number of batches across all (limited) shards."""
        total_sequences = sum(self._shard_sequence_counts)
        return math.ceil(total_sequences / self.batch_size) if total_sequences else 0

    def num_batches_for_shards(self, shard_indices: List[int]) -> int:
        """Returns the total number of batches for a subset of canonical shards."""
        total_sequences = sum(
            self._shard_sequence_counts[shard_index]
            for shard_index in self._validate_shard_indices(shard_indices)
        )
        return math.ceil(total_sequences / self.batch_size) if total_sequences else 0

    def _load_shard(self, shard_index: int) -> List[np.ndarray]:
        """
        Loads a specific shard by index using the pre-built position index.
        """
        if shard_index < 0 or shard_index >= len(self.shard_files):
            raise IndexError("Shard index out of range")

        shard_path = os.path.join(self.data_path, self.shard_files[shard_index])
        shard = np.load(shard_path, mmap_mode="r")
        index = self._shard_indices[shard_index]
        return [shard[start:end].copy() for start, end in index]

    def load_shard(self, shard_index: int) -> List[np.ndarray]:
        """Public API for loading a shard (delegates to _load_shard)."""
        return self._load_shard(shard_index)

    def load_shard_sequences(
        self, shard_index: int, local_indices: List[int]
    ) -> Dict[int, np.ndarray]:
        """
        Loads specific sequences from a shard by their local (within-shard) indices
        using the pre-built position index. Only the memory-mapped pages covering
        the requested sequences are read from disk — no full shard parse needed.

        Returns a dict mapping local_index -> token array.
        """
        if not local_indices:
            return {}
        index = self._shard_indices[shard_index]
        if len(index) == 0:
            return {}
        shard_path = os.path.join(self.data_path, self.shard_files[shard_index])
        shard = np.load(shard_path, mmap_mode="r")
        result: Dict[int, np.ndarray] = {}
        for local_idx in local_indices:
            if 0 <= local_idx < len(index):
                start, end = index[local_idx]
                result[local_idx] = shard[start:end].copy()
        return result

    def get_sequence(self, sequence_id: int) -> np.ndarray:
        """
        Retrieves a specific sequence (row) by its global ID.
        """
        for shard_idx, (start_id, end_id) in enumerate(self.shard_id_ranges):
            if start_id <= sequence_id <= end_id:
                local_idx = sequence_id - start_id
                seq_map = self.load_shard_sequences(shard_idx, [local_idx])
                if local_idx in seq_map:
                    return seq_map[local_idx]
                break
        
        raise IndexError(f"Sequence ID {sequence_id} not found.")

    def get_batches(
        self,
        pad_to_max: bool = True,
        max_length: int = 64,
        device: Optional[torch.device] = None,
    ) -> Generator[Tuple[torch.Tensor, Union[torch.Tensor, List[torch.Tensor]]], None, None]:
        """
        Yields batches of sequences across all shards using indexed mmap reads.
        """
        yield from self.get_batches_for_shards(
            list(range(len(self.shard_files))),
            pad_to_max=pad_to_max,
            max_length=max_length,
            device=device,
        )

    def get_batches_for_shards(
        self,
        shard_indices: List[int],
        pad_to_max: bool = True,
        max_length: int = 64,
        device: Optional[torch.device] = None,
    ) -> Generator[Tuple[torch.Tensor, Union[torch.Tensor, List[torch.Tensor]]], None, None]:
        """
        Yields batches from selected canonical shards while preserving global IDs.
        """
        batch_tokens_list: List[np.ndarray] = []
        batch_ids_list: List[int] = []

        for shard_index in self._validate_shard_indices(shard_indices):
            start_id, _end_id = self.shard_id_ranges[shard_index]
            if start_id == -1:
                continue

            shard_path = os.path.join(self.data_path, self.shard_files[shard_index])
            shard = np.load(shard_path, mmap_mode="r")
            index = self._shard_indices[shard_index]

            for local_idx, (start_pos, end_pos) in enumerate(index):
                seq = shard[start_pos:end_pos].copy()
                batch_tokens_list.append(seq)
                batch_ids_list.append(start_id + local_idx)

                if len(batch_tokens_list) == self.batch_size:
                    batch_tokens = self._batch_to_tensor(batch_tokens_list, pad_to_max, max_length, device=(self.device if device is None else device))
                    batch_ids = torch.tensor(batch_ids_list, device=self.device, dtype=torch.int32)
                    yield batch_ids, batch_tokens
                    batch_tokens_list = []
                    batch_ids_list = []

        if batch_tokens_list:
            batch_tokens = self._batch_to_tensor(batch_tokens_list, pad_to_max, max_length, device=(self.device if device is None else device))
            batch_ids = torch.tensor(batch_ids_list, device=self.device, dtype=torch.int32)
            yield batch_ids, batch_tokens

    def _validate_shard_indices(self, shard_indices: List[int]) -> List[int]:
        """Validate selected shard IDs against the loaded canonical shard list."""
        for shard_index in shard_indices:
            if shard_index < 0 or shard_index >= len(self.shard_files):
                raise IndexError(f"Shard index out of range: {shard_index}")
        return list(shard_indices)

    def get_batches_by_ids(
        self,
        sequence_ids: list[int],
        pad_to_max: bool = True,
        max_length: int = 64,
        device: Optional[torch.device] = None,
    ) -> Generator[Tuple[torch.Tensor, Union[torch.Tensor, List[torch.Tensor]]], None, None]:
        """
        Optimized batching for a specific set of IDs.
        Uses the shard index for direct random-access reads — only the memory-mapped
        pages covering each requested sequence are loaded from disk.
        """
        current_shard_idx = -1
        current_shard_mmap: Optional[np.ndarray] = None

        batch_tokens_list: List[np.ndarray] = []
        batch_ids_list: List[int] = []

        for seq_id in sequence_ids:
            target_shard_idx = -1
            start_id_in_shard = 0
            for i, (start, end) in enumerate(self.shard_id_ranges):
                if start <= seq_id <= end:
                    target_shard_idx = i
                    start_id_in_shard = start
                    break

            if target_shard_idx == -1:
                continue

            # Open a new mmap handle only when the shard changes
            if target_shard_idx != current_shard_idx:
                shard_path = os.path.join(self.data_path, self.shard_files[target_shard_idx])
                current_shard_mmap = np.load(shard_path, mmap_mode="r")
                current_shard_idx = target_shard_idx

            local_idx = seq_id - start_id_in_shard
            shard_index = self._shard_indices[target_shard_idx]
            if 0 <= local_idx < len(shard_index):
                start_pos, end_pos = shard_index[local_idx]
                seq = current_shard_mmap[start_pos:end_pos].copy()  # type: ignore[index]
                batch_tokens_list.append(seq)
                batch_ids_list.append(seq_id)

            if len(batch_tokens_list) == self.batch_size:
                batch_tokens = self._batch_to_tensor(batch_tokens_list, pad_to_max, max_length, device=(self.device if device is None else device))
                batch_ids = torch.tensor(batch_ids_list, device=self.device, dtype=torch.int32)
                yield batch_ids, batch_tokens
                batch_tokens_list, batch_ids_list = [], []

        if batch_tokens_list:
            batch_tokens = self._batch_to_tensor(batch_tokens_list, pad_to_max, max_length, device=(self.device if device is None else device))
            batch_ids = torch.tensor(batch_ids_list, device=self.device, dtype=torch.int32)
            yield batch_ids, batch_tokens

    def _batch_to_tensor(
        self,
        batch: List[np.ndarray],
        pad_to_max: bool,
        max_length: int,
        device: Optional[torch.device],
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Convert a list of sequences to a padded tensor or list of tensors."""
        if pad_to_max:
            tensor_batch = torch.zeros((len(batch), max_length), dtype=torch.long)
            for idx, seq in enumerate(batch):
                length = min(len(seq), max_length)
                tensor_batch[idx, :length] = torch.from_numpy(seq[:length].astype(np.int64))
            if self.pin_memory:
                tensor_batch = tensor_batch.pin_memory()
            if device is not None:
                tensor_batch = tensor_batch.to(device, non_blocking=self.pin_memory)
            return tensor_batch
        tensor_batch = [torch.from_numpy(seq.astype(np.int64)) for seq in batch]
        if device is not None:
            tensor_batch = [tensor.to(device) for tensor in tensor_batch]
        return tensor_batch
