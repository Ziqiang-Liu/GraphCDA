import argparse


def parameter_parser():

    parser = argparse.ArgumentParser(description="Run GraphCDA.")

    parser.add_argument("--dataset-path",
                        nargs="?",
                        default="./datasets",
                        help="Training datasets.")

    parser.add_argument("--epoch",
                        type=int,
                        default=400,
                        help="Number of training epochs. Default is 400.")

    parser.add_argument("--gcn-layers",
                        type=int,
                        default=2,
                        help="Number of Graph Convolutional Layers. Default is 2.")

    parser.add_argument("--out-channels",
                        type=int,
                        default=256,
                        help="out-channels of cnn. Default is 256.")

    parser.add_argument("--circRNA-number",
                        type=int,
                        default=585,
                        help="circRNA number. Default is 585.")

    parser.add_argument("--fcir",
                        type=int,
                        default=128,
                        help="circRNA feature dimensions. Default is 128.")

    parser.add_argument("--disease-number",
                        type=int,
                        default=88,
                        help="disease number. Default is 88.")

    parser.add_argument("--fdis",
                        type=int,
                        default=128,
                        help="disease feature dimensions. Default is 128.")


    return parser.parse_args()