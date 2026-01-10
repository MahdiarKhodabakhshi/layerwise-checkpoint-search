import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from transformers import AutoModel
    from huggingface_hub import HfApi
except ImportError:
    print("ERROR: transformers or huggingface_hub not installed")
    print("Install with: pip install transformers huggingface_hub")
    sys.exit(1)


def check_checkpoint(model_id: str, revision: str) -> bool:
    try:
        api = HfApi()
        # Try to get model info for this revision
        model_info = api.model_info(model_id, revision=revision)
        return True
    except Exception as e:
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify Pythia checkpoint availability")
    parser.add_argument(
        "--model-size",
        choices=["14m", "70m", "410m"],
        default="410m",
        help="Model size to check (default: 410m)"
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=["step-1000", "step-2000", "step-5000", "step-10000", "main"],
        help="Checkpoints to verify"
    )
    args = parser.parse_args()
    
    model_id = f"EleutherAI/pythia-{args.model_size}"
    
    print("=" * 80)
    print(f"Verifying checkpoints for {model_id}")
    print("=" * 80)
    print()
    
    available = []
    unavailable = []
    
    for checkpoint in args.checkpoints:
        print(f"Checking {checkpoint}...", end=" ", flush=True)
        if check_checkpoint(model_id, checkpoint):
            print("✅ Available")
            available.append(checkpoint)
        else:
            print("❌ Not found")
            unavailable.append(checkpoint)
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Available checkpoints: {available}")
    if unavailable:
        print(f"Unavailable checkpoints: {unavailable}")
        print()
        print("⚠️  Update configs/mteb_layersweep.yaml to remove unavailable checkpoints")
    else:
        print("✅ All checkpoints are available!")
    print("=" * 80)
    
    if available:
        print()
        print("Suggested config (copy to mteb_layersweep.yaml):")
        print(f'  revisions: {available}')
        print()


if __name__ == "__main__":
    main()

