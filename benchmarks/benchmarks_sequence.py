import asyncio
import collections
import tempfile
import uuid
from pathlib import Path

import numpy as np
import tifffile as tf

from tiled.catalog import from_uri
from tiled.client import Context, from_context
from tiled.client.register import (
    IMG_SEQUENCE_EMPTY_NAME_ROOT,
    IMG_SEQUENCE_STEM_PATTERNS,
    register,
)
from tiled.server.app import build_app


class TiffSequenceRegistrationSuite:
    """
    Benchmark suite for registering TIFF files in a Tiled catalog.
    Run in isolation and within the current environment with:
        `asv run --environment existing:same --bench TiffSequenceRegistrationSuite`

    Purpose:
    In machine learning applications, large datasets of images are often stored with randomized filenames.
    When users explored using Tiled to manage such datasets, they observed significant performance differences
    during registration (with `tiled serve directory`) based on filename scheme.
    In particular, files with sequential names (e.g., image0001.tif) were registered very fast,
    while files with randomized names took much longer to register.
    With default walkers, files with sequential names are grouped and registered as a sequence,
    whereas files with randomized names are registered one by one.
    This benchmark suite measures the impact of filename scheme on registration performance
    based on the initial hypothesis that aside from registration overhead,
    the filename grouping logic might be the cause of the performance differences.

    Key Finding:
    Grouping logic introduces minimal overhead (on the order of milliseconds).
    Image size is not a significant factor in registration performance.
    The performance difference is due to the number of registration calls:
    one for grouped files vs. one per file when grouping cannot be applied.

    """

    param_names = ["num_files", "image_size", "filename_scheme"]
    # params = [[100, 500, 1000], [128, 256, 512, 1024], ["sequential", "random"]]
    # Since image size does not affect performance significantly,
    # the smaller scale test below is sufficient to show the differences.
    params = [[100], [128], ["sequential", "random"]]

    def setup(self, num_files, image_size, filename_scheme):
        self.num_files = num_files
        self.image_size = image_size
        self.directory = tempfile.TemporaryDirectory()
        self.data = np.ones((image_size, image_size))
        self.filename_scheme = filename_scheme

        catalog = from_uri(
            f"sqlite:///{self.directory.name}/catalog.db",
            init_if_not_exists=True,
            writable_storage=self.directory.name,
        )
        self.context = Context.from_app(build_app(catalog))
        self.client = from_context(self.context)

        self._write_tiffs()

    def teardown(self, num_files, image_size, filename_scheme):
        self.context.close()
        self.directory.cleanup()

    def _generate_filename(self, i):
        """Generate a filename based on the file index and filename scheme."""
        if self.filename_scheme == "sequential":
            # Use zero-padded filenames with just enough digits to cover the range
            # (e.g., image000.tif to image999.tif)
            num_zeros = len(str(self.num_files - 1))
            return f"image{i:0{num_zeros}d}.tif"
        else:
            # Use random UUIDs for filenames
            return f"{uuid.uuid4().hex}.tif"

    def _write_tiffs(self):
        for i in range(self.num_files):
            filename = self._generate_filename(i)
            tf.imwrite(Path(self.directory.name) / filename, self.data * i)

    def time_register_tiffs_default(self, num_files, image_size, filename_scheme):
        """
        Register TIFF files in the directory using the default walkers:
        first group_image_sequences, then one_node_per_item.
        This is the default behavior of tiled.client.register.register.
        """
        asyncio.run(register(self.client, self.directory.name))

    def time_register_tiffs_one_by_one(self, num_files, image_size, filename_scheme):
        """Register TIFF files in the directory one by one using the one_node_per_item walker."""
        asyncio.run(
            register(
                self.client,
                self.directory.name,
                walkers=["tiled.client.register:one_node_per_item"],
            )
        )

    def time_group_image_sequences_without_register(
        self, num_files, image_size, filename_scheme
    ):
        """
        Group image sequences without registering them in the client.
        This isolates the grouping logic in tiled.client.register.group_image_sequences
        from the registration process.
        """
        path = Path(self.directory.name)
        # Portion from tiled.client.register._walk
        files = []
        directories = []
        for item in path.iterdir():
            if item.is_dir():
                directories.append(item)
            else:
                files.append(item)
        # Portion from tiled.client.register.group_image_sequences
        unhandled_files = []
        sequences = collections.defaultdict(list)
        for file in files:
            file_ext = Path(file).suffixes[-1] if len(Path(file).suffixes) > 0 else None
            if file.is_file() and file_ext and file_ext in IMG_SEQUENCE_STEM_PATTERNS:
                match = IMG_SEQUENCE_STEM_PATTERNS[file_ext].match(file.name)
                if match:
                    sequence_name, _sequence_number = match.groups()
                    if sequence_name == "":
                        sequence_name = IMG_SEQUENCE_EMPTY_NAME_ROOT
                    sequences[sequence_name].append(file)
                    continue
            unhandled_files.append(file)
        # Unlike the original group_image_sequences function, this version stops before registering.


if __name__ == "__main__":
    # Manually run individual benchmark methods for debugging.
    # Note: ASV suppresses stdout/stderr during benchmark runs,
    # so use this for development or validation only.
    # Alternatively the `asv run ...` command with `--quick --show-stderr --dry-run` does show stderr
    suite = TiffSequenceRegistrationSuite()
    num_files = 100
    image_size = 128
    filename_scheme = "sequential"  # or "random"
    suite.setup(num_files, image_size, filename_scheme)
    suite.time_register_tiffs_default(num_files, image_size, filename_scheme)
    print(f"Client size: {len(suite.client)}")
    suite.time_register_tiffs_one_by_one(num_files, image_size, filename_scheme)
    print(f"Client size after one by one: {len(suite.client)}")
    suite.teardown(num_files, image_size, filename_scheme)
