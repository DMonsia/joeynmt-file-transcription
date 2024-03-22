import argparse
from pathlib import Path
from typing import Dict

import pandas as pd
from joeynmt.config import BaseConfig, load_config, parse_global_args
from joeynmt.datasets import StreamDataset
from joeynmt.helpers_for_ddp import get_logger, use_ddp
from joeynmt.prediction import predict, prepare
from torch.utils.data import Dataset

logger = get_logger()


def translate_file(cfg: Dict, file_to_translate: str, output_path: str) -> None:
    """
    File translation function.
    Loads the template from the control point and translates all sentences in the input file. Translations are saved in the output file.
    Note: The input sentences don't have to be pre-tokenized.

    :param cfg: configuration dict
    :file_to_translate: Path to file to translate
    :param output_path: path to output file
    """

    # parse args
    args = parse_global_args(cfg, rank=0, mode="translate")

    # load the model
    model, _, _, test_data = prepare(args, rank=0, mode="translate")
    assert isinstance(test_data, StreamDataset)

    logger.info(
        "Ready to decode. (device: %s, n_gpu: %s, use_ddp: %r, fp16: %r)",
        args.device.type,
        args.n_gpu,
        use_ddp(),
        args.autocast["enabled"],
    )

    def _translate_data(test_data: Dataset, args: BaseConfig):
        """Translates given dataset, using parameters from outer scope."""
        _, _, hypotheses, trg_tokens, trg_scores, _ = predict(
            model=model,
            data=test_data,
            compute_loss=False,
            device=args.device,
            rank=0,
            n_gpu=args.n_gpu,
            normalization="none",
            num_workers=args.num_workers,
            args=args.test,
            autocast=args.autocast,
        )
        return hypotheses, trg_tokens, trg_scores

    with open(file_to_translate) as f:
        # we remove the header
        rows = f.read().strip().split("\n")[1:]
        row_ids = []
        for row in rows:
            _id, line = row.split(",")
            if not line.strip():
                # skip empty lines and print warning
                logger.warning("The sentence in line %s is empty. Skip to load.", _id)
                continue
            row_ids.append(_id)
            test_data.set_item(line.rstrip())
        all_hypotheses, _, _ = _translate_data(test_data, args)
        assert len(all_hypotheses) == len(test_data) * args.test.n_best

    if args.test.n_best > 1:
        all_hypotheses = [
            all_hypotheses[i] for i in range(0, len(all_hypotheses), args.test.n_best)
        ]
    if output_path is not None:
        # write to outputfile if given
        out_file = Path(output_path).expanduser()
        pd.DataFrame({"id": row_ids, args.data["src"]["lang"]: all_hypotheses}).to_csv(
            out_file, index=False
        )
        logger.info("Translations saved to: %s.", out_file)
    else:
        raise FileNotFoundError("No output file found.")


def main():
    ap = argparse.ArgumentParser("joeynmt")

    ap.add_argument(
        "file_to_translate",
        metavar="file-to-translate",
        type=str,
        help="Path to files to be translated.",
    )

    ap.add_argument(
        "config_path", metavar="config-path", type=str, help="Path to YAML config file"
    )

    ap.add_argument(
        "-o", "--output-path", type=str, help="Path for saving translation output"
    )

    ap.add_argument(
        "-a",
        "--save-attention",
        action="store_true",
        help="Save attention visualizations",
    )

    ap.add_argument("-s", "--save-scores", action="store_true", help="Save scores")

    ap.add_argument(
        "-t", "--skip-test", action="store_true", help="Skip test after training"
    )

    ap.add_argument(
        "-d", "--use-ddp", action="store_true", help="Invoke DDP environment"
    )

    args = ap.parse_args()

    # read config file
    cfg = load_config(Path(args.config_path))

    translate_file(
        cfg=cfg, file_to_translate=args.file_to_translate, output_path=args.output_path
    )


if __name__ == "__main__":
    main()
