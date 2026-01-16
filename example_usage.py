from pathlib import Path
from kernels.kernel_card_utils import (
    _update_kernel_card_available_funcs,
    _update_kernel_card_license,
    _load_or_create_kernel_card,
    _update_kernel_card_usage,
    _update_kernel_card_backends,
)
import argparse


def main(args):
    kernel_dir = Path(args.kernels_dir)

    kernel_card = _load_or_create_kernel_card(kernel_description=args.description, license="apache-2.0")

    updated_card = _update_kernel_card_usage(kernel_card=kernel_card, local_path=kernel_dir)
    updated_card = _update_kernel_card_available_funcs(kernel_card=kernel_card, local_path=kernel_dir)
    updated_card = _update_kernel_card_backends(kernel_card=updated_card, local_path=kernel_dir)
    updated_card = _update_kernel_card_license(kernel_card, kernel_dir)

    card_path = args.card_path or "README.md"
    updated_card.save(card_path)
    print("Kernel card updated successfully!")
    print("\nUpdated content preview:")
    print(updated_card.content[:500] + "...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernels_dir", type=str, required=True, help="Path to the kernels source.")
    parser.add_argument("--card_path", type=str, default=None, help="Path to save the card to.")
    parser.add_argument("--description", type=str, default=None)
    args = parser.parse_args()

    main(args)
