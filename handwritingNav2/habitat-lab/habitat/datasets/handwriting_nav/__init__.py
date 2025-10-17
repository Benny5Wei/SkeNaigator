#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.core.dataset import Dataset
from habitat.core.registry import registry

def _try_register_handwritingnavdataset():
    try:
        from habitat.datasets.handwriting_nav.handwriting_nav_dataset import (  # noqa: F401
            HandWritingNavDataset,
        )

    except ImportError as e:
        pointnav_import_error = e
        @registry.register_dataset(name="HandWritingNav")
        class HandWritingNavDatasetImportError(Dataset):
            def __init__(self, *args, **kwargs):
                raise pointnav_import_error