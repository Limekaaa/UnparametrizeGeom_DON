import json 
import os


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Create a dataset of square meshes with circular holes.")
    parser.add_argument(
        "--experiment_directory",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include specifications file 'specs.json' and 'specs_data.json'."
    )
    args = parser.parse_args()

    
    specs_filename = os.path.join(args.experiment_directory, "specs.json")
    specs_data_filename = os.path.join(args.experiment_directory, "specs_data.json")

    if not os.path.isfile(specs_filename):
        raise Exception(
            f'The file: {specs_filename} does not exist"'
        )

    if not os.path.isfile(specs_data_filename):
        raise Exception(
            f'The file: {specs_data_filename} does not exist"'
        )

    specs = json.load(open(specs_filename))
    specs_data = json.load(open(specs_data_filename))
    
    sdf_generator = __import__("data_generators."+specs_data["SDFDataGenerator"], fromlist="sdf_data_generator")

    sdf_generator.sdf_data_generator(specs, specs_data)