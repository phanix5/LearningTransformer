from collections import defaultdict
import os
from typing import BinaryIO

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


def bpe_tokenizer(
        input_path: str,
        vocab_size: int,
        special_tokens: list[str]
        ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, 100, special_tokens[0].encode("utf-8"))
        global_counts = defaultdict(int)

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            pre_token_counts = pre_tokenize(chunk, special_tokens)

            # Merge counts from all the chunks
            for pre_token, count in pre_token_counts.items():
                global_counts[pre_token] += count

        # Convert each Pre-Token into a tuple of single byte bytes
        global_counts = {
            tuple(bytes([b]) for b in pre_token): count 
            for pre_token, count in global_counts.items()
        }

        vocab = initialize_vocab(special_tokens)
        return merge(global_counts, vocab, vocab_size)

# bpe_tokenizer('tokenizer_sample_text.txt', 260, ['<|endoftext|>'])