from PIL import Image
import sys
import os

IMG_DIR = "data/silhouettes"
NAMES_FILE = "data/pokemon_names.txt"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def check_image_resolution(image_path):
    # Validate file path
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' does not exist.")
        return

    # Open image
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            print(f"Image: {os.path.basename(image_path)}")
            print(f"Resolution: {width} x {height} pixels")
    except Exception as e:
        print(f"Error reading image: {e}")


def list_immediate_subfolders(root_dir):
    l = sorted(
        [
            name
            for name in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, name))
        ]
    )
    print(f"Found {len(l)} Pokemon classes")
    return l


def write_subfolders_to_file(subfolders, output_path):
    print(f"Writing {len(subfolders)} subfolder names to '{output_path}'...")

    with open(output_path, "w") as f:
        for name in subfolders:
            f.write(name + "\n")

    print(f"Done writing {len(subfolders)} subfolder names to '{output_path}'")


def is_image_file(filename):
    return os.path.splitext(filename)[1].lower() in SUPPORTED_EXTS


def find_largest_image_recursive(root_dir):
    max_area = -1
    max_info = None  # (path, width, height)

    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if not is_image_file(fn):
                continue
            path = os.path.join(dirpath, fn)
            try:
                # PIL usually gets size without decoding full image
                with Image.open(path) as img:
                    w, h = img.size
                area = w * h
                if area > max_area:
                    max_area = area
                    max_info = (path, w, h)
            except Exception as e:
                # Skip unreadable/corrupt files
                print(f"Warning: could not read '{path}': {e}")
                continue

    return max_info  # or None if no images found


def main():
    # if len(sys.argv) < 2:
    #     print("Usage: python scan_images.py <main_folder>")
    #     sys.exit(1)

    # main_folder = sys.argv[1]
    assert os.path.exists(IMG_DIR), f"Folder '{IMG_DIR}' does not exist."

    subfolders = list_immediate_subfolders(IMG_DIR)
    # print("Subfolders:", subfolders)
    write_subfolders_to_file(subfolders, NAMES_FILE)

    # # 2) Largest image resolution
    largest = find_largest_image_recursive(IMG_DIR)
    if largest is None:
        print("No images found.")
    else:
        path, w, h = largest
        print(f"Largest image: {path}")
        print(f"Resolution: {w} x {h} (pixels)")


if __name__ == "__main__":
    main()
