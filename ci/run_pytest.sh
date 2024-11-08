#!/bin/bash
# Copyright 2024 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Source JAXCI environment variables.
source "ci/utilities/setup_jaxci_envs.sh" "$1"
# Set up the build environment.
source "ci/utilities/setup_build_environment.sh"

export PY_COLORS=1
export JAX_SKIP_SLOW_TESTS=true

"$JAXCI_PYTHON" -c "import jax; print(jax.default_backend()); print(jax.devices()); print(len(jax.devices()))"

if [[ $JAXCI_RUN_PYTEST_CPU == 1 ]]; then
  export TF_CPP_MIN_LOG_LEVEL=0
  echo "Running CPU tests..."
  "$JAXCI_PYTHON" -m pytest -n auto --tb=short --maxfail=20 tests examples
fi

if [[ $JAXCI_RUN_PYTEST_GPU == 1 ]]; then
  nvidia-smi
  export NCCL_DEBUG=WARN
  export TF_CPP_MIN_LOG_LEVEL=0

  echo "Running GPU tests..."
  export XLA_PYTHON_CLIENT_ALLOCATOR=platform
  export XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1
  "$JAXCI_PYTHON" -m pytest -n 8 --tb=short --maxfail=20 \
  tests examples \
  --deselect=tests/multi_device_test.py::MultiDeviceTest::test_computation_follows_data \
  --deselect=tests/xmap_test.py::XMapTest::testCollectivePermute2D \
  --deselect=tests/multiprocess_gpu_test.py::MultiProcessGpuTest::test_distributed_jax_visible_devices \
  --deselect=tests/compilation_cache_test.py::CompilationCacheTest::test_task_using_cache_metric
fi

if [[ $JAXCI_RUN_PYTEST_TPU == 1 ]]; then
  "$JAXCI_PYTHON" -c 'import sys; print("python version:", sys.version)'
  "$JAXCI_PYTHON" -c 'import jax; print("jax version:", jax.__version__)'
  "$JAXCI_PYTHON" -c 'import jaxlib; print("jaxlib version:", jaxlib.__version__)'
  strings /usr/local/lib/"$JAXCI_PYTHON"/site-packages/libtpu/libtpu.so | grep 'Built on'
  "$JAXCI_PYTHON" -c 'import jax; print("libtpu version:",jax.lib.xla_bridge.get_backend().platform_version)'

  echo "Running TPU tests..."
  export JAX_PLATFORMS=tpu,cpu
  # Run single-accelerator tests in parallel
  export JAX_ENABLE_TPU_XDIST=true

  "$JAXCI_PYTHON" -m pytest -n="$JAXCI_TPU_CORES" --tb=short \
    --deselect=tests/pallas/tpu_pallas_test.py::PallasCallPrintTest \
    --maxfail=20 -m "not multiaccelerator" tests examples

  # Run Pallas printing tests, which need to run with I/O capturing disabled.
  export TPU_STDERR_LOG_LEVEL=0
  "$JAXCI_PYTHON" -m pytest -s tests/pallas/tpu_pallas_test.py::PallasCallPrintTest

  # Run multi-accelerator across all chips
  "$JAXCI_PYTHON" -m pytest --tb=short --maxfail=20 -m "multiaccelerator" tests
fi