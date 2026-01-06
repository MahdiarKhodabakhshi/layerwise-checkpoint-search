#!/usr/bin/env python3
"""
Filter checkpoints for a specific array task.

This script takes a list of checkpoints and splits them into chunks,
then outputs the checkpoints for a specific chunk index.
Used by job array scripts to distribute checkpoints across array tasks.
"""
import sys
from pathlib import Path

# All checkpoints (excluding main)
ALL_CHECKPOINTS = [
    "step0", "step1", "step2", "step4", "step8", "step16", "step32", "step64", "step128", "step256", "step512",
    "step1000", "step2000", "step3000", "step4000", "step5000", "step6000", "step7000", "step8000", "step9000",
    "step10000", "step11000", "step12000", "step13000", "step14000", "step15000", "step16000", "step17000", "step18000", "step19000",
    "step20000", "step21000", "step22000", "step23000", "step24000", "step25000", "step26000", "step27000", "step28000", "step29000",
    "step30000", "step31000", "step32000", "step33000", "step34000", "step35000", "step36000", "step37000", "step38000", "step39000",
    "step40000", "step41000", "step42000", "step43000", "step44000", "step45000", "step46000", "step47000", "step48000", "step49000",
    "step50000", "step51000", "step52000", "step53000", "step54000", "step55000", "step56000", "step57000", "step58000", "step59000",
    "step60000", "step61000", "step62000", "step63000", "step64000", "step65000", "step66000", "step67000", "step68000", "step69000",
    "step70000", "step71000", "step72000", "step73000", "step74000", "step75000", "step76000", "step77000", "step78000", "step79000",
    "step80000", "step81000", "step82000", "step83000", "step84000", "step85000", "step86000", "step87000", "step88000", "step89000",
    "step90000", "step91000", "step92000", "step93000", "step94000", "step95000", "step96000", "step97000", "step98000", "step99000",
    "step100000", "step101000", "step102000", "step103000", "step104000", "step105000", "step106000", "step107000", "step108000", "step109000",
    "step110000", "step111000", "step112000", "step113000", "step114000", "step115000", "step116000", "step117000", "step118000", "step119000",
    "step120000", "step121000", "step122000", "step123000", "step124000", "step125000", "step126000", "step127000", "step128000", "step129000",
    "step130000", "step131000", "step132000", "step133000", "step134000", "step135000", "step136000", "step137000", "step138000", "step139000",
    "step140000", "step141000", "step142000", "step143000"
]


def split_checkpoints(checkpoints, num_chunks):
    """Split checkpoints into approximately equal chunks."""
    chunk_size = len(checkpoints) // num_chunks
    remainder = len(checkpoints) % num_chunks
    
    chunks = []
    start = 0
    for i in range(num_chunks):
        # Distribute remainder across first chunks
        size = chunk_size + (1 if i < remainder else 0)
        end = start + size
        chunks.append(checkpoints[start:end])
        start = end
    
    return chunks


def main():
    if len(sys.argv) != 3:
        print("Usage: filter_checkpoints_for_array.py <num_chunks> <chunk_index>", file=sys.stderr)
        sys.exit(1)
    
    num_chunks = int(sys.argv[1])
    chunk_index = int(sys.argv[2])
    
    if chunk_index < 0 or chunk_index >= num_chunks:
        print(f"Error: chunk_index must be in [0, {num_chunks-1}], got {chunk_index}", file=sys.stderr)
        sys.exit(1)
    
    chunks = split_checkpoints(ALL_CHECKPOINTS, num_chunks)
    checkpoints_for_this_chunk = chunks[chunk_index]
    
    # Output as space-separated list (for bash array)
    print(" ".join(checkpoints_for_this_chunk))


if __name__ == "__main__":
    main()
