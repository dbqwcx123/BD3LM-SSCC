# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Defines project-wide constants."""

BATCH_SIZE = 4

NUM_IMAGE_TRAIN = 785 * (2048//128) * (1024//128)  # 785 张 2048x1024 图像
NUM_IMAGE_VALID = 99 * (2048//128) * (1024//128)  # 99 张 2048x1024 图像
NUM_IMAGE_TEST = 10
IMAGE_SHAPE_TRAIN = (256, 256, 3)
IMAGE_SHAPE_TEST = (32, 32, 3)

IS_CHANNEL_WISED = True  # RGB False; Gray True
CHUNK_SHAPE_2D = (4, 4)  # 图像块的高度和宽度

CHUNK_SIZE_BYTES = CHUNK_SHAPE_2D[0] * CHUNK_SHAPE_2D[1]
CHUNK_SIZE_BYTES *= 1 if IS_CHANNEL_WISED else 3
PATCHES_PER_IMAGE_TRAIN = (IMAGE_SHAPE_TRAIN[0] // CHUNK_SHAPE_2D[0]) * (IMAGE_SHAPE_TRAIN[1] // CHUNK_SHAPE_2D[1])
PATCHES_PER_IMAGE_TRAIN *= 3 if IS_CHANNEL_WISED else 1
PATCHES_PER_IMAGE_TEST = (IMAGE_SHAPE_TEST[0] // CHUNK_SHAPE_2D[0]) * (IMAGE_SHAPE_TEST[1] // CHUNK_SHAPE_2D[1])
PATCHES_PER_IMAGE_TEST *= 3 if IS_CHANNEL_WISED else 1

NUM_CHUNKS_TRAIN = PATCHES_PER_IMAGE_TRAIN * NUM_IMAGE_TRAIN
NUM_CHUNKS_VALID = PATCHES_PER_IMAGE_TRAIN * NUM_IMAGE_VALID
NUM_CHUNKS_TEST = PATCHES_PER_IMAGE_TEST * NUM_IMAGE_TEST

ALPHABET_SIZE = 256
ARITHMETIC_CODER_BASE = 2
ARITHMETIC_CODER_PRECISION = 32
