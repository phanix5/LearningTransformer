from collections import defaultdict
import json
import os
from typing import BinaryIO
import multiprocessing as mp
from functools import partial
from datetime import datetime

import regex as re

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pre_tokenize(chunk: str, special_tokens: list[str]) -> dict[bytes, int]:
    delimiter = "|".join(re.escape(token) for token in special_tokens)
    mini_chunks = re.split(delimiter, chunk)
    splitter_regex = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    splitter = re.compile(splitter_regex)
    pre_token_counts = defaultdict(int)
    for mini_chunk in mini_chunks:
        for match in splitter.finditer(mini_chunk):
            pre_token_counts[match.group(0).encode("utf-8")] += 1
    return pre_token_counts

def initialize_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    # Combine both vocab initializations into dictionary comprehensions
    vocab = {i: token.encode("utf-8") for i, token in enumerate(special_tokens)}
    vocab.update({len(special_tokens) + i: bytes([i]) for i in range(256)})
    return vocab

def merge_bytes_pair_in_pre_token(
        pre_token: tuple[bytes, ...],
        pair: tuple[bytes, bytes]
        ) -> tuple[bytes, ...]:
    merged_token = pair[0] + pair[1]
    result = []
    i = 0
    
    while i < len(pre_token):
        if (i < len(pre_token) - 1 and 
            pre_token[i] == pair[0] and 
            pre_token[i + 1] == pair[1]):
            result.append(merged_token)
            i += 2
        else:
            result.append(pre_token[i])
            i += 1
    
    return tuple(result)


def merge(
        pre_token_counts: dict[tuple[bytes, ...], int],
        vocab: dict[int, bytes],
        vocab_size: int
        ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    merges = []

    # Initialize pair counts
    pair_counts = defaultdict(int)
    for pre_token, count in pre_token_counts.items():
        for pair in zip(pre_token, pre_token[1:]):
            pair_counts[pair] += count

    while len(vocab) < vocab_size:
        # Find best pair more concisely
        # Select pair with the highest count; among ties, pick lexicographically the largest
        # Key puts count first (x[1]) so max() compares by count, then by pair (x[0])
        best_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
        merges.append(best_pair)

        new_pre_token_counts = {}

        for pre_token, count in pre_token_counts.items():
            # Check if this pre-token has the best pair and is a candidate for the merge
            has_consecutive_pair = any(
                pre_token[i] == best_pair[0] and pre_token[i + 1] == best_pair[1]
                for i in range(len(pre_token) - 1)
            )

            if has_consecutive_pair:
                # Decrement old pair counts. Basically remove the contribution this pre_token has to the counts.
                # we'll add them back in after merging
                for pair in zip(pre_token, pre_token[1:]):
                    pair_counts[pair] -= count
                    if pair_counts[pair] == 0:
                        del pair_counts[pair]

                # Merge and add new pair counts
                merged_pre_token = merge_bytes_pair_in_pre_token(pre_token, best_pair)
                new_pre_token_counts[merged_pre_token] = count

                # Add back the contributions from this merged pre_token
                for pair in zip(merged_pre_token, merged_pre_token[1:]):
                    pair_counts[pair] += count
            else:
                # No consecutive pair - pre_token unchanged
                new_pre_token_counts[pre_token] = count

        # Update pre_token_counts and vocab
        pre_token_counts = new_pre_token_counts

        # Add the new merged token to the vocabulory
        vocab[max(vocab.keys()) + 1] = best_pair[0] + best_pair[1]

    return vocab, merges


def process_chunk_worker(start_end: tuple[int, int], input_path: str, special_tokens: list[str]) -> dict[bytes, int]:
    """Worker function to process a single chunk of the file."""
    start, end = start_end
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        return pre_tokenize(chunk, special_tokens)


def bpe_tokenizer(
        input_path: str,
        vocab_size: int,
        special_tokens: list[str],
        num_processes: int | None = None
        ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    if num_processes is None:
        cpu_count = mp.cpu_count()
        num_processes = cpu_count if cpu_count is not None else 4
    
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, 100, special_tokens[0].encode("utf-8"))
        
        # Prepare chunk boundaries for parallel processing
        chunk_ranges = list(zip(boundaries[:-1], boundaries[1:]))
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processing {len(chunk_ranges)} chunks using {num_processes} processes...")
        
        # Use multiprocessing to process chunks in parallel
        with mp.Pool(processes=num_processes) as pool:
            # Create partial function with fixed arguments
            worker_func = partial(process_chunk_worker, 
                                input_path=input_path, 
                                special_tokens=special_tokens)
            
            # Process all chunks in parallel
            chunk_results = pool.map(worker_func, chunk_ranges)
        
        # Merge results from all chunks
        global_counts = defaultdict(int)
        for pre_token_counts in chunk_results:
            for pre_token, count in pre_token_counts.items():
                global_counts[pre_token] += count

        # Convert each Pre-Token into a tuple of single byte bytes
        global_counts = {
            tuple(bytes([b]) for b in pre_token): count 
            for pre_token, count in global_counts.items()
        }

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Finished Pre-Tokenizing, generating vocabulary...")
        vocab = initialize_vocab(special_tokens)
        return merge(global_counts, vocab, vocab_size)

def serialize_vocab(vocab: dict[int, bytes], output_path: str) -> None:
    """Serialize vocabulary to a JSON file."""
    # Convert bytes to strings for JSON serialization
    vocab_str = {}
    for token_id, token_bytes in vocab.items():
        try:
            # Try to decode as UTF-8
            token_str = token_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # Fall back to hex representation for non-UTF-8 bytes
            token_str = f"<hex:{token_bytes.hex()}>"
        vocab_str[token_str] = token_id
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_str, f, indent=2, ensure_ascii=False)


def serialize_merges(merges: list[tuple[bytes, bytes]], output_path: str) -> None:
    """Serialize merge rules to a text file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for merge_pair in merges:
            left_bytes, right_bytes = merge_pair
            try:
                # Try to decode as UTF-8
                left_str = left_bytes.decode('utf-8')
                right_str = right_bytes.decode('utf-8')
            except UnicodeDecodeError:
                # Fall back to hex representation for non-UTF-8 bytes
                left_str = f"<hex:{left_bytes.hex()}>"
                right_str = f"<hex:{right_bytes.hex()}>"
            
            f.write(f"{left_str} {right_str}\n")


if __name__ == "__main__":
    # Train the BPE tokenizer and serialize the results
    vocab, merges = bpe_tokenizer('../TinyStoriesV2-GPT4-train.txt', 10000, ['<|endoftext|>'])

    # Serialize to files
    serialize_vocab(vocab, 'vocab.json')
    serialize_merges(merges, 'merges.txt')

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Vocabulary saved to vocab.json ({len(vocab)} tokens)")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Merge rules saved to merges.txt ({len(merges)} merges)")