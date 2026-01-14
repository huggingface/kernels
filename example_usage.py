from pathlib import Path
from kernels.kernel_card_utils import _load_or_create_model_card, _update_model_card_usage
import argparse


def main(args):
    kernel_dir = Path(args.kernels_dir)

    model_card = _load_or_create_model_card(
        kernel_description=args.description, license="apache-2.0"
    )

    updated_card = _update_model_card_usage(model_card=model_card, local_path=kernel_dir)

    card_path = args.card_path or "README.md"
    updated_card.save(card_path)
    print("Model card updated successfully!")
    print("\nUpdated content preview:")
    print(updated_card.content[:500] + "...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernels_dir", type=str, required=True, help="Path to the kernels source.")
    parser.add_argument("--card_path", type=str, default=None, help="Path to save the card to.")
    parser.add_argument("--description", type=str, default=None)
    args = parser.parse_args()

    main(args)
