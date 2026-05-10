# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""pytest config for EP example tests."""
import pytest


def pytest_addoption(parser):
    parser.addoption("--coordinator-address", action="store", default="localhost:12345")
    parser.addoption("--num-processes", action="store", default=1)
    parser.addoption("--process-id", action="store", default=0)
    parser.addoption("--local-device-ids", action="store", default=None)


@pytest.fixture(autouse=True)
def distributed_args(request):
    if request.cls:
        request.cls.coordinator_address = request.config.getoption("--coordinator-address")
        request.cls.num_processes = int(request.config.getoption("--num-processes"))
        request.cls.process_id = int(request.config.getoption("--process-id"))
        request.cls.local_device_ids = request.config.getoption("--local-device-ids")
