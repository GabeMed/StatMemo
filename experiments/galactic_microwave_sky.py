import argparse
import numpy as np
import pysm3
import pysm3.units as u


def main():
    parser = argparse.ArgumentParser()
    # Required args from ${ARGS}
    parser.add_argument("frequency", type=float, help="Frequency in GHz")
    parser.add_argument("nside", type=int, help="HEALPix NSIDE resolution")
    parser.add_argument("seed", type=int, help="Random seed")
    parser.add_argument(
        "models", type=str, help="Comma-separated PySM models (e.g. d1,s1)"
    )

    # Framework args (ignored logic-wise)
    parser.add_argument("--exec-mode", type=str, default="no-cache")
    parser.add_argument("-s", type=str, default=None)

    args = parser.parse_args()

    # Parse model list
    model_list = args.models.split(",")

    # Set seed
    np.random.seed(args.seed)

    # Run PySM
    sky = pysm3.Sky(nside=args.nside, preset_strings=model_list)
    sky_map = sky.get_emission(args.frequency * u.GHz)

    # Compute result (mean I)
    mean_I = np.mean(sky_map[0])

    # Final output line required by your automation
    print(f"Result: {mean_I}")


if __name__ == "__main__":
    main()
