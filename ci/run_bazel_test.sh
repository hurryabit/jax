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
# Source "JAXCI_" environment variables.
source "ci/utilities/source_jaxci_envs.sh" "$1"
# Set up the build environment.
source "ci/utilities/setup_build_environment.sh"

# Run Bazel CPU tests with RBE.
if [[ $JAXCI_RUN_BAZEL_TEST_CPU_RBE == 1 ]]; then
      os=$(uname -s | awk '{print tolower($0)}')
      arch=$(uname -m)

      # When running on Mac or Linux Aarch64, we only build the test targets and
      # not run them. These platforms do not have native RBE support so we
      # RBE cross-compile them on remote Linux x86 machines. As the tests still
      # need to be run on the host machine and because running the tests on a
      # single machine can take a long time, we skip running them on these
      # platforms.
      if [[ $os == "darwin" ]] || ( [[ $os == "linux" ]] && [[ $arch == "aarch64" ]] ); then
            echo "Building RBE CPU tests..."
            bazel build --config=rbe_cross_compile_${os}_${arch} \
                  --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION" \
                  --override_repository=xla="${JAXCI_XLA_GIT_DIR}" \
                  --test_env=JAX_NUM_GENERATED_CASES=25 \
                  --test_env=JAX_SKIP_SLOW_TESTS=true \
                  --action_env=JAX_ENABLE_X64=0 \
                  --test_output=errors \
                  --color=yes \
                  //tests:cpu_tests //tests:backend_independent_tests
      else
            echo "Running RBE CPU tests..."
            bazel test --config=rbe_${os}_${arch} \
                  --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION" \
                  --override_repository=xla="${JAXCI_XLA_GIT_DIR}" \
                  --test_env=JAX_NUM_GENERATED_CASES=25 \
                  --test_env=JAX_SKIP_SLOW_TESTS=true \
                  --action_env=JAX_ENABLE_X64=0 \
                  --test_output=errors \
                  --color=yes \
                  //tests:cpu_tests //tests:backend_independent_tests
      fi
fi

# Run Bazel GPU tests with RBE.
if [[ $JAXCI_RUN_BAZEL_TEST_GPU_RBE == 1 ]]; then
      nvidia-smi
      echo "Running RBE GPU tests..."

      # Only Linux x86 builds run GPU tests
      # Runs single accelerator tests with one GPU apiece.
      bazel test --config=rbe_linux_x86_64_cuda \
            --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION" \
            --override_repository=xla="${JAXCI_XLA_GIT_DIR}" \
            --test_env=XLA_PYTHON_CLIENT_ALLOCATOR=platform \
            --test_output=errors \
            --test_env=TF_CPP_MIN_LOG_LEVEL=0 \
            --test_env=JAX_EXCLUDE_TEST_TARGETS=PmapTest.testSizeOverflow \
            --test_tag_filters=-multiaccelerator \
            --test_env=JAX_SKIP_SLOW_TESTS=true \
            --action_env=JAX_ENABLE_X64=0 \
            --color=yes \
            //tests:gpu_tests //tests:backend_independent_tests //tests/pallas:gpu_tests //tests/pallas:backend_independent_tests
fi

# Run Non-RBE Bazel GPU tests (single accelerator and multiaccelerator tests).
if [[ $JAXCI_RUN_BAZEL_TEST_GPU_NON_RBE == 1 ]]; then
      nvidia-smi
      echo "Running single accelerator tests (no RBE)..."

      # Runs single accelerator tests with one GPU apiece.
      # It appears --run_under needs an absolute path.
      # The product of the `JAX_ACCELERATOR_COUNT`` and `JAX_TESTS_PER_ACCELERATOR`
      # should match the VM's CPU core count (set in `--local_test_jobs`).
      bazel test --config=ci_linux_x86_64_cuda \
            --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION" \
            --//jax:build_jaxlib=false \
            --test_env=XLA_PYTHON_CLIENT_ALLOCATOR=platform \
            --run_under "$(pwd)/build/parallel_accelerator_execute.sh" \
            --test_output=errors \
            --test_env=JAX_ACCELERATOR_COUNT=4 \
            --test_env=JAX_TESTS_PER_ACCELERATOR=12 \
            --local_test_jobs=48 \
            --test_env=JAX_EXCLUDE_TEST_TARGETS=PmapTest.testSizeOverflow \
            --test_tag_filters=-multiaccelerator \
            --test_env=TF_CPP_MIN_LOG_LEVEL=0 \
            --test_env=JAX_SKIP_SLOW_TESTS=true \
            --action_env=JAX_ENABLE_X64=0 \
            --action_env=NCCL_DEBUG=WARN \
            --color=yes \
            //tests:gpu_tests //tests:backend_independent_tests \
            //tests/pallas:gpu_tests //tests/pallas:backend_independent_tests

      echo "Running multi-accelerator tests (no RBE)..."
      # Runs multiaccelerator tests with all GPUs.
      bazel test --config=ci_linux_x86_64_cuda \
            --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION" \
            --//jax:build_jaxlib=false \
            --test_env=XLA_PYTHON_CLIENT_ALLOCATOR=platform \
            --test_output=errors \
            --jobs=8 \
            --test_tag_filters=multiaccelerator \
            --test_env=TF_CPP_MIN_LOG_LEVEL=0 \
            --test_env=JAX_SKIP_SLOW_TESTS=true \
            --action_env=JAX_ENABLE_X64=0 \
            --action_env=NCCL_DEBUG=WARN \
            --color=yes \
            //tests:gpu_tests //tests/pallas:gpu_tests
fi