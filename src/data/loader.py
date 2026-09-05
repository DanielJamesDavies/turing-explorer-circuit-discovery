import os
import math
import time
import numpy as np
import torch
from typing import List, Generator, Union, Optional, cast, Tuple, Dict, Any
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
        self._token_cache_ids: Optional[torch.Tensor] = None
        self._token_cache_tokens: Optional[torch.Tensor] = None
        self._token_cache_id_to_row: Dict[int, int] = {}
        self._token_cache_max_length: Optional[int] = None
        self._token_cache_metadata: Dict[str, Any] = {}

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

        index_path = self._get_index_path(shard_index)
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        # Atomic publish: write to a per-process temp file, then rename.
        # Concurrent shard processes (one per GPU x co-tenants) all build
        # these lazily on first use; a plain np.save let readers see a
        # half-written file (EOFError, 2026-09-01 H100 launch).
        tmp_path = "%s.tmp-%d" % (index_path, os.getpid())
        with open(tmp_path, "wb") as fh:
            np.save(fh, index)
        os.replace(tmp_path, index_path)
        return index

    def _load_or_build_index(self, shard_index: int) -> np.ndarray:
        """Returns the cached index, rebuilding it if missing or stale."""
        index_path = self._get_index_path(shard_index)
        shard_path = os.path.join(self.data_path, self.shard_files[shard_index])
        if (
            os.path.exists(index_path)
            and os.path.getmtime(index_path) >= os.path.getmtime(shard_path)
        ):
            try:
                return np.load(index_path)
            except (EOFError, ValueError, OSError):
                # Truncated/corrupt cache (e.g. left by a pre-atomic-write
                # race): rebuild rather than die.
                pass
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

    def get_batches_by_ids_grouped(
        self,
        sequence_ids: list[int],
        pad_to_max: bool = True,
        max_length: int = 64,
        device: Optional[torch.device] = None,
        restore_order: bool = True,
    ) -> Generator[Tuple[torch.Tensor, Union[torch.Tensor, List[torch.Tensor]]], None, None]:
        """
        Load specific IDs grouped by shard to reduce mmap churn.

        When restore_order is true, returned batches preserve the original input
        order after the shard-local reads complete. This lets callers optimize IO
        without changing ranked-selection semantics.
        """
        grouped: Dict[int, List[Tuple[int, int, int]]] = {}
        for rank, seq_id in enumerate(sequence_ids):
            located = self._locate_sequence_id(seq_id)
            if located is None:
                continue
            shard_idx, start_id_in_shard = located
            grouped.setdefault(shard_idx, []).append((rank, seq_id, seq_id - start_id_in_shard))

        loaded: List[Tuple[int, int, np.ndarray]] = []
        for shard_idx in sorted(grouped):
            shard_path = os.path.join(self.data_path, self.shard_files[shard_idx])
            shard = np.load(shard_path, mmap_mode="r")
            shard_index = self._shard_indices[shard_idx]
            for rank, seq_id, local_idx in grouped[shard_idx]:
                if 0 <= local_idx < len(shard_index):
                    start_pos, end_pos = shard_index[local_idx]
                    loaded.append((rank, seq_id, shard[start_pos:end_pos].copy()))

        if restore_order:
            loaded.sort(key=lambda item: item[0])

        batch_tokens_list: List[np.ndarray] = []
        batch_ids_list: List[int] = []
        for _rank, seq_id, seq in loaded:
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

    def preload_sequence_tokens(
        self,
        sequence_ids: list[int],
        max_length: int = 64,
        dtype: torch.dtype = torch.int32,
        max_bytes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Preload selected sequence tokens into a CPU RAM cache.

        The cache is intentionally CPU-resident. Callers gather from it and only
        move the requested window to the accelerator, avoiding repeated scattered
        mmap reads during negctx filtering.
        """
        started = time.perf_counter()
        requested_ids = [int(seq_id) for seq_id in sequence_ids]
        requested_count = len(requested_ids)
        estimated_bytes = requested_count * int(max_length) * torch.empty((), dtype=dtype).element_size()
        if max_bytes is not None and estimated_bytes > int(max_bytes):
            raise MemoryError(
                f"Requested token cache would use about {estimated_bytes} bytes, "
                f"exceeding limit {int(max_bytes)} bytes"
            )

        grouped: Dict[int, List[Tuple[int, int, int]]] = {}
        sorted_requests = sorted((seq_id, rank) for rank, seq_id in enumerate(requested_ids))
        shard_idx = 0
        for seq_id, rank in sorted_requests:
            while shard_idx < len(self.shard_id_ranges):
                start_id, end_id = self.shard_id_ranges[shard_idx]
                if start_id == -1 or seq_id > end_id:
                    shard_idx += 1
                    continue
                break
            if shard_idx >= len(self.shard_id_ranges):
                break
            start_id, end_id = self.shard_id_ranges[shard_idx]
            if start_id <= seq_id <= end_id:
                grouped.setdefault(shard_idx, []).append((rank, seq_id, seq_id - start_id))

        located_count = sum(len(entries) for entries in grouped.values())
        cached_tokens = torch.zeros((located_count, max_length), dtype=dtype)
        loaded_by_rank: List[Tuple[int, int, int]] = []
        output_row = 0
        for shard_idx in sorted(grouped):
            shard_path = os.path.join(self.data_path, self.shard_files[shard_idx])
            # Preload is a one-time cache build; reading each small shard fully is
            # much faster than many tiny mmap-backed copies on mounted filesystems.
            shard = np.load(shard_path, mmap_mode=None)
            shard_index = self._shard_indices[shard_idx]
            for rank, seq_id, local_idx in grouped[shard_idx]:
                if 0 <= local_idx < len(shard_index):
                    start_pos, end_pos = shard_index[local_idx]
                    seq = shard[start_pos:end_pos]
                    length = min(len(seq), int(max_length))
                    if length > 0:
                        cached_tokens[output_row, :length] = torch.as_tensor(
                            seq[:length],
                            dtype=dtype,
                        )
                    loaded_by_rank.append((rank, int(seq_id), output_row))
                    output_row += 1

        cached_tokens = cached_tokens[:output_row].contiguous()
        loaded_by_rank.sort(key=lambda item: item[0])
        if output_row:
            ordered_rows = torch.tensor([row for _rank, _seq_id, row in loaded_by_rank], dtype=torch.long)
            cached_tokens = cached_tokens.index_select(0, ordered_rows).contiguous()
            loaded_ids = [seq_id for _rank, seq_id, _row in loaded_by_rank]
            cached_ids = torch.tensor(loaded_ids, dtype=torch.int64)
        else:
            loaded_ids = []
            cached_ids = torch.zeros((0,), dtype=torch.int64)

        actual_bytes = int(cached_tokens.numel() * cached_tokens.element_size())
        if max_bytes is not None and actual_bytes > int(max_bytes):
            raise MemoryError(
                f"Loaded token cache uses {actual_bytes} bytes, exceeding limit {int(max_bytes)} bytes"
            )

        self._token_cache_ids = cached_ids
        self._token_cache_tokens = cached_tokens
        self._token_cache_id_to_row = {int(seq_id): row for row, seq_id in enumerate(loaded_ids)}
        self._token_cache_max_length = int(max_length)
        self._token_cache_metadata = {
            "requested_count": requested_count,
            "loaded_count": len(loaded_ids),
            "max_length": int(max_length),
            "dtype": str(dtype).replace("torch.", ""),
            "bytes": actual_bytes,
            "estimated_bytes": int(estimated_bytes),
            "duration_s": time.perf_counter() - started,
        }
        return dict(self._token_cache_metadata)

    def has_token_cache(self, max_length: int = 64) -> bool:
        return (
            self._token_cache_tokens is not None
            and self._token_cache_ids is not None
            and self._token_cache_max_length is not None
            and int(self._token_cache_max_length) >= int(max_length)
        )

    def token_cache_metadata(self) -> Dict[str, Any]:
        return dict(self._token_cache_metadata)

    def get_cached_tokens_by_ids(
        self,
        sequence_ids: list[int],
        max_length: int = 64,
        device: Optional[torch.device] = None,
    ) -> Tuple[List[int], torch.Tensor, List[int]]:
        """
        Return cached tokens for requested IDs in requested order plus cache misses.
        """
        if not self.has_token_cache(max_length=max_length):
            return [], torch.zeros((0, max_length), dtype=torch.long, device=(self.device if device is None else device)), [int(seq_id) for seq_id in sequence_ids]

        assert self._token_cache_tokens is not None
        hit_ids: List[int] = []
        miss_ids: List[int] = []
        row_indices: List[int] = []
        for seq_id in sequence_ids:
            seq_id_int = int(seq_id)
            row = self._token_cache_id_to_row.get(seq_id_int)
            if row is None:
                miss_ids.append(seq_id_int)
            else:
                hit_ids.append(seq_id_int)
                row_indices.append(int(row))

        target_device = self.device if device is None else device
        if not row_indices:
            return hit_ids, torch.zeros((0, max_length), dtype=torch.long, device=target_device), miss_ids

        rows = torch.tensor(row_indices, dtype=torch.long)
        tokens = self._token_cache_tokens.index_select(0, rows)
        tokens = self._format_cached_tokens(tokens, max_length=max_length, device=target_device)
        return hit_ids, tokens, miss_ids

    def get_cached_batches_by_ids(
        self,
        sequence_ids: list[int],
        max_length: int = 64,
        device: Optional[torch.device] = None,
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        hit_ids, tokens, _miss_ids = self.get_cached_tokens_by_ids(
            sequence_ids,
            max_length=max_length,
            device=device,
        )
        if not hit_ids:
            return
        for start in range(0, len(hit_ids), max(1, int(self.batch_size))):
            batch_ids = hit_ids[start : start + max(1, int(self.batch_size))]
            batch_tokens = tokens[start : start + len(batch_ids)]
            yield torch.tensor(batch_ids, device=self.device, dtype=torch.int32), batch_tokens

    def clear_token_cache(self) -> None:
        self._token_cache_ids = None
        self._token_cache_tokens = None
        self._token_cache_id_to_row = {}
        self._token_cache_max_length = None
        self._token_cache_metadata = {}

    def _format_cached_tokens(
        self,
        tokens: torch.Tensor,
        *,
        max_length: int,
        device: Optional[torch.device],
    ) -> torch.Tensor:
        if tokens.shape[1] > int(max_length):
            tokens = tokens[:, : int(max_length)]
        elif tokens.shape[1] < int(max_length):
            padded = torch.zeros((tokens.shape[0], int(max_length)), dtype=tokens.dtype)
            padded[:, : tokens.shape[1]] = tokens
            tokens = padded
        tokens = tokens.to(dtype=torch.long)
        if device is not None:
            tokens = tokens.to(device, non_blocking=self.pin_memory)
        return tokens

    def _locate_sequence_id(self, sequence_id: int) -> Optional[Tuple[int, int]]:
        for shard_idx, (start_id, end_id) in enumerate(self.shard_id_ranges):
            if start_id <= sequence_id <= end_id:
                return shard_idx, start_id
        return None

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
