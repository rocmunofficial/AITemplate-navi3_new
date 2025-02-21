#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import unittest

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import IntImm, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    filter_test_cases_by_params,
    get_random_torch_tensor,
    TestEnv,
)

from parameterized import parameterized


class ConvBiasTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(1)

    def _test_conv_bias(
        self,
        batch=1,
        input_dim_x=64,
        input_dim_y=64,
        weight_dim_x=3,
        weight_dim_y=3,
        input_channels=320,
        output_channels=4,
        copy_op=False,
        test_name="conv2d_bias",
        dtype="float16",
    ):
        target = detect_target()
        X = Tensor(
            shape=[IntImm(batch), IntImm(input_dim_x), IntImm(input_dim_y), IntImm(input_channels)],
            dtype=dtype,
            name="input_0",
            is_input=True,
        )
        W = Tensor(
            shape=[IntImm(output_channels), IntImm(weight_dim_x), IntImm(weight_dim_y), IntImm(input_channels)],
            dtype=dtype,
            name="input_1",
            is_input=True,
        )
        B = Tensor(
            shape=[IntImm(output_channels)],
            dtype=dtype,
            name="input_2",
            is_input=True,
        )
        OP = ops.conv2d_bias(stride=1, pad=1, dilate=1)
        if copy_op:
            OP = ops.conv2d_bias(**OP._get_op_attributes())
        Y = OP(X, W, B)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", test_name)

        X_pt = get_random_torch_tensor([batch, input_channels, input_dim_x, input_dim_y], dtype=dtype)
        W_pt = get_random_torch_tensor([output_channels, input_channels, weight_dim_x, weight_dim_y], dtype=dtype)
        B_pt = get_random_torch_tensor([1, output_channels, 1, 1], dtype=dtype)
        Y_pt = torch.nn.functional.conv2d(X_pt.float(), W_pt.float(), padding=1).to(
            dtype=X_pt.dtype
        )
        Y_pt = Y_pt + B_pt
        x = X_pt.permute((0, 2, 3, 1)).contiguous()
        w = W_pt.permute((0, 2, 3, 1)).contiguous()
        inputs = {"input_0": x, "input_1": w, "input_2": B_pt.squeeze()}
        y = torch.empty_like(Y_pt).permute((0, 2, 3, 1)).contiguous()
        module.run_with_tensors(inputs, [y])
        y_transpose = y.permute((0, 3, 1, 2))
        if target.name() == "cuda":
            if dtype == "float32":
                torch.testing.assert_close(Y_pt, y_transpose, atol=5e-2, rtol=1e-2)
            else:
                torch.testing.assert_close(Y_pt, y_transpose, atol=1e-2, rtol=1e-2)
        else:
            torch.testing.assert_close(Y_pt, y_transpose, atol=1.25e-1, rtol=1e-1)

    @parameterized.expand(
        **filter_test_cases_by_params(
            {
                # TestEnv.CUDA_LESS_THAN_SM80: [("float16")],
                # TestEnv.CUDA_SM80: [("float32"), ("bfloat16")],
                TestEnv.ROCM: [("float16")],
            }
        )
    )
    def test_conv2d_bias(self, dtype):
        # default
        self._test_conv_bias(
            test_name=f"conv2d_bias_{dtype}",
            dtype=dtype,
        )
        self._test_conv_bias(
            copy_op=True,
            test_name=f"conv2d_bias_{dtype}_copy_op",
            dtype=dtype,
        )
        # unet model test
        # Not implemented yet
        # vae_model_conv = [
        #     [64 ,64 ,1, 1, 4, 4],
        #     [64 ,64 ,3, 3, 512, 4],
        #     [64 ,64 ,3, 3, 512, 512],
        #     [128 ,128 ,3, 3, 512, 512],
        #     [256 ,256 ,3, 3, 512, 512],
        #     [256 ,256 ,3, 3, 256, 512],
        #     [256 ,256 ,3, 3, 256, 256],
        #     [256 ,256 ,3, 3, 512, 256],
        #     [512 ,512 ,3, 3, 256, 256],
        #     [512 ,512 ,3, 3, 128, 256],
        #     [512 ,512 ,3, 3, 128, 128],
        #     [512 ,512 ,3, 3, 128, 3],
        # ]
        # test_conv_cnt = 0
        # for configs in vae_model_conv:
        #     self._test_conv_bias(
        #         input_dim_x=configs[0],
        #         input_dim_y=configs[1],
        #         weight_dim_x=configs[2],
        #         weight_dim_y=configs[3],
        #         input_channels=configs[4],
        #         output_channels=configs[5],
        #         copy_op=False,
        #         test_name=f"conv2d_bias_{dtype}_{test_conv_cnt}",
        #         dtype=dtype,
        #     )
            
        #     self._test_conv_bias(
        #         input_dim_x=configs[0],
        #         input_dim_y=configs[1],
        #         weight_dim_x=configs[2],
        #         weight_dim_y=configs[3],
        #         input_channels=configs[4],
        #         output_channels=configs[5],
        #         copy_op=True,
        #         test_name=f"conv2d_bias_{dtype}_{test_conv_cnt}_copy_op",
        #         dtype=dtype,
        #     )
            
        # vae model test
        vae_model_conv = [
            [64 ,64 ,1, 1, 4, 4],
            [64 ,64 ,3, 3, 512, 4],
            [64 ,64 ,3, 3, 512, 512],
            [128 ,128 ,3, 3, 512, 512],
            [256 ,256 ,3, 3, 512, 512],
            [256 ,256 ,3, 3, 256, 512],
            [256 ,256 ,3, 3, 256, 256],
            [256 ,256 ,3, 3, 512, 256],
            [512 ,512 ,3, 3, 256, 256],
            [512 ,512 ,3, 3, 128, 256],
            [512 ,512 ,3, 3, 128, 128],
            [512 ,512 ,3, 3, 128, 3],
        ]
        test_conv_cnt = 0
        for configs in vae_model_conv:
            self._test_conv_bias(
                input_dim_x=configs[0],
                input_dim_y=configs[1],
                weight_dim_x=configs[2],
                weight_dim_y=configs[3],
                input_channels=configs[4],
                output_channels=configs[5],
                copy_op=False,
                test_name=f"conv2d_bias_{dtype}_{test_conv_cnt}",
                dtype=dtype,
            )
            
            self._test_conv_bias(
                input_dim_x=configs[0],
                input_dim_y=configs[1],
                weight_dim_x=configs[2],
                weight_dim_y=configs[3],
                input_channels=configs[4],
                output_channels=configs[5],
                copy_op=True,
                test_name=f"conv2d_bias_{dtype}_{test_conv_cnt}_copy_op",
                dtype=dtype,
            )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
