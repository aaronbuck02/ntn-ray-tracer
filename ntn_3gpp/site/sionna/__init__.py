#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Sionna Library with Sionna RT"""

from . import rt

from .rt import __version__  # sionna-rt-only install lacks the top-level attribute
