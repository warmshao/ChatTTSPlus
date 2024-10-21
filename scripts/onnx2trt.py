#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import pdb
import sys
import logging
import argparse

import tensorrt as trt
import ctypes
import numpy as np
import onnx
from polygraphy.backend.trt import ModifyNetworkOutputs, Profile
from polygraphy.backend.trt import (
    engine_from_bytes,
    engine_from_network,
    network_from_onnx_path,
    save_engine,
)

logging.basicConfig(level=logging.INFO)
logging.getLogger("EngineBuilder").setLevel(logging.INFO)
log = logging.getLogger("EngineBuilder")


class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    """

    def __init__(self, verbose=False):
        """
        :param verbose: If enabled, a higher verbosity level will be set on the TensorRT logger.
        """
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE

        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        # self.config.max_workspace_size = 24 * (2 ** 30)  # 12 GB

        profile = self.builder.create_optimization_profile()

        # GPT
        profile.set_shape("inputs_embeds", min=[1, 1, 768], opt=[2, 1, 768], max=[4, 2048, 768])
        profile.set_shape("attention_mask", min=[1, 1], opt=[2, 256], max=[4, 2048])
        profile.set_shape("position_ids", min=[1, 1], opt=[2, 1], max=[4, 2048])
        profile.set_shape("past_key_values", min=[20, 2, 1, 12, 1, 64], opt=[20, 2, 2, 12, 256, 64],
                          max=[20, 2, 4, 12, 2048, 64])

        self.config.add_optimization_profile(profile)

        self.batch_size = None
        self.network = None
        self.parser = None

    def set_mixed_precision(self):
        """
        Experimental precision mode.
        Enable mixed-precision mode. When set, the layers defined here will be forced to FP16 to maximize
        INT8 inference accuracy, while having minimal impact on latency.
        """
        self.config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        self.config.set_flag(trt.BuilderFlag.DIRECT_IO)
        self.config.set_flag(trt.BuilderFlag.REJECT_EMPTY_ALGORITHMS)

        # add or remove blocks.
        for i in range(self.network.num_layers):
            layer = self.network.get_layer(i)
            if "norm" in layer.name:
                layer.precision = trt.DataType.FLOAT
                log.info(
                    "Mixed-Precision Layer {} set to FLOAT STRICT data type".format(
                        layer.name
                    )
                )

    def create_network(self, onnx_path):
        """
        Parse the ONNX graph and create the corresponding TensorRT network definition.
        :param onnx_path: The path to the ONNX graph to load.
        """

        onnx_path = os.path.realpath(onnx_path)
        self.network_infos = self.builder, self.network, _ = network_from_onnx_path(
            onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM]
        )
        # self.set_mixed_precision()

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]
        log.info("Network Description")
        for input in inputs:
            self.batch_size = input.shape[0]
            log.info("Input '{}' with shape {} and dtype {}".format(input.name, input.shape, input.dtype))
        for output in outputs:
            log.info("Output '{}' with shape {} and dtype {}".format(output.name, output.shape, output.dtype))

    def create_engine(
            self,
            engine_path,
            precision
    ):
        """
        Build the TensorRT engine and serialize it to disk.
        :param engine_path: The path where to serialize the engine to.
        :param precision: The datatype to use for the engine, either 'fp32', 'fp16' or 'int8'.
        """
        engine_path = os.path.realpath(engine_path)
        engine_dir = os.path.dirname(engine_path)
        os.makedirs(engine_dir, exist_ok=True)
        log.info("Building {} Engine in {}".format(precision, engine_path))

        if precision == "fp16":
            if not self.builder.platform_has_fast_fp16:
                log.warning("FP16 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "int8":
            if not self.builder.platform_has_fast_int8:
                log.warning("Int8 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.INT8)

        try:
            engine = engine_from_network(
                self.network_infos,
                self.config
            )
        except Exception as e:
            print(f"Failed to build engine: {e}")
            return 1
        try:
            save_engine(engine, path=engine_path)
        except Exception as e:
            print(f"Failed to save engine: {e}")
            return 1
        return 0


def main(args):
    builder = EngineBuilder(args.verbose)
    builder.create_network(args.onnx)
    builder.create_engine(
        args.engine,
        args.precision
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx", required=True, help="The input ONNX model file to load")
    parser.add_argument("-e", "--engine", help="The output path for the TRT engine")
    parser.add_argument(
        "-p",
        "--precision",
        default="fp16",
        choices=["fp32", "fp16", "int8"],
        help="The precision mode to build in, either 'fp32', 'fp16' or 'int8', default: 'fp16'",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable more verbose log output")
    args = parser.parse_args()
    if args.engine is None:
        args.engine = args.onnx.replace(".onnx", ".trt")
    main(args)
