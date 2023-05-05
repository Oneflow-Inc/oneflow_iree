"""
Copyright 2020 The OneFlow Authors. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import absolute_import
import setuptools

reqs = [
        'protobuf',
        'iree-compiler==20230404.479',
        'iree-runtime==20230404.479'
    ]

long_description = (
    "oneflow_iree is a toolkit for converting nn.graph of OneFlow to tosa and running on iree backend.\n\n"
)

long_description += "GitHub: https://github.com/Oneflow-Inc/oneflow_iree\n"

setuptools.setup(
    name="oneflow_iree",
    version="0.0.1",
    description="a toolkit for converting nn.graph of OneFlow to tosa and running on iree backend.",
    long_description=long_description,
    long_description_content_type="text/plain",
    packages=setuptools.find_packages(),
    url="https://github.com/Oneflow-Inc/oneflow_iree",
    install_requires=reqs,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    license="Apache 2.0",
)
