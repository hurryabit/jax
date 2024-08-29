# Copyright 2021 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Serializations routines for pytrees including array and non-array serialization.
"""

from __future__ import annotations

from os import PathLike
import os
from types import ModuleType
import re
from typing import Any, Iterable
from uuid import uuid4, UUID
import json
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, Future
import shutil
import logging

import jax
from jax.tree_util import PyTreeDef
from jax.util import safe_zip
from jax._src import distributed
from jax._src.layout import Layout
from jax.experimental import multihost_utils
from jax.experimental.array_serialization import asyncio_utils
from jax.experimental.array_serialization import tensorstore_impl as ts_impl
import jax.experimental.array_serialization.pytree_serialization_utils as utils
from jax.sharding import SingleDeviceSharding
from jax._src.path import epath_installed, Path
import numpy as np

serialization_registry = utils.default_serialization_registry

_THREADING_SAVE_LOCK = threading.Lock()

_REMOTE_URL_PREFIXES = ['gs://', 's3://']
_PYTREEDEF_FILE = "pytreedef.json"
_TENSORSTORE_SUFFIX = ".tensorstore"
_ARCHIVE_NAME = "archive.zip"
_OBJ_DATA_PREFIX = "obj_data"
_NODE_AUX_DATA_PREFIX = "node_aux_data"
_TYPE_ID_LEAF_DELIMITER = " -> "
_USE_OCDBT = True  # a lot of the code relies on this being True
_MAX_PATH_LENGTH = 4096
_ARRAY_STORE_DIRNAME = f"array_store{_TENSORSTORE_SUFFIX}"
_ARRAY_TYPE_NAME = "Array"
_ARRAY_TYPE_REGEX = r"Array\[\[([0-9, ]*)\],\s*([a-zA-Z0-9_]+)\]"
_DOT_REPLACEMENT = ":"

__all__ = ["save", "load", "load_pytreedef", "nonblocking_load",
           "nonblocking_save", "serialization_registry"]

PyTreeT = Any
PickleModule = ModuleType

logger = logging.getLogger(__name__)

def _get_sync_client() -> distributed.xla_extension.DistributedRuntimeClient:
  assert jax.process_count() > 1, (
    "You are attempting to wait for other hosts, but there is only 1 host"
  )
  assert distributed.global_state.client is not None, (
    "The distributed runtime is not initialized. You likely need to call"
    " `jax.distributed.initialize()` first."
  )
  return distributed.global_state.client

def _get_unique_sync_key() -> str | None:
  """Generate a thread-local key for ensuring all host finish (de)serializing"""
  if jax.process_count() == 1:
    return None
  # broadcast a thread-local unique barrier name
  sync_key_id = UUID(bytes=np.array(multihost_utils.broadcast_one_to_all(
    np.frombuffer(uuid4().bytes, dtype=np.int32))).tobytes())
  sync_key = f"jax_sync_key_{str(sync_key_id)}"
  return sync_key

def _is_str_same_on_all_hosts(path: str | PathLike[str]) -> bool:
  """All-gather the location of the checkpoint and check if it's the same."""
  if jax.process_count() <= 1:
    return False
  path_b = str(path).encode("utf-8")
  assert len(path_b) <= _MAX_PATH_LENGTH, (
      f"Path exceeds maximum length of {_MAX_PATH_LENGTH} in multiprocess"
      " case.")
  path_array = np.concatenate([
      np.frombuffer(path_b, dtype=np.uint8), np.zeros(
          _MAX_PATH_LENGTH - len(path_b), dtype=np.uint8)])
  all_path_arrays = multihost_utils.process_allgather(path_array)
  return bool(np.all(all_path_arrays == all_path_arrays[:1, ...]))

def _sync_on_key(key: str | None, extra_tag: str = "") -> None:
  if key is None:
    return
  full_key = key if not extra_tag else f"{key}-{extra_tag}"
  multihost_utils.sync_global_devices(full_key)

def _is_array_like(x):
  return isinstance(x, (jax.Array, np.ndarray))

def _leaf_to_type_desc(leaf) -> str:
  if leaf is None:
    return "null"
  elif isinstance(leaf, (np.ndarray, jax.Array)):
    return (f"{_ARRAY_TYPE_NAME}[[{', '.join(map(str, leaf.shape))}]," +
            f" {leaf.dtype.name}]")
  else:
    return type(leaf).__name__

def _leaf_desc_to_leaf(leaf_desc: str) -> str | jax.ShapeDtypeStruct:
  leaf_type: str = (leaf_desc.split(_TYPE_ID_LEAF_DELIMITER, 1)[0]
                    if _TYPE_ID_LEAF_DELIMITER in leaf_desc else leaf_desc)
  if not leaf_type.startswith(_ARRAY_TYPE_NAME):
    return leaf_type
  shape_dtype_match = re.match(_ARRAY_TYPE_REGEX, leaf_type)
  assert shape_dtype_match is not None, (
      f"Failed to parse array descriptor: {leaf_type} with pattern:"
      f" {_ARRAY_TYPE_REGEX}")
  shape_str, dtype_str = shape_dtype_match.groups()
  shape = [int(x.strip()) for x in shape_str.strip("]").strip().split(",")
            if len(x.strip()) > 0]
  dtype = jax.numpy.dtype(dtype_str)
  return jax.ShapeDtypeStruct(shape, dtype)

def _join_leaf_type_and_id(leaf_type: str, leaf_id: str) -> str:
  return f"{leaf_type}{_TYPE_ID_LEAF_DELIMITER}{leaf_id}"

def _inscribe_leaf_types(pytree_repr: dict[str, Any],
                         leaf_id_type_map: dict[str, str]):
  """Rewrite a JSON PyTree representation by adding type to leaf_id."""
  if pytree_repr["node_type"] == "leaf":
    leaf_id = pytree_repr["leaf_id"]
    if leaf_id is None:
      return
    pytree_repr["leaf_id"] = _join_leaf_type_and_id(leaf_id_type_map[leaf_id],
                                                    leaf_id)
  else:
    _ = [_inscribe_leaf_types(child, leaf_id_type_map)
         for child in pytree_repr["children"]]

def _inplace_add_types_to_pytree_repr(pytree_repr, leaf_ids_flat, data_flat):
  # inscribe types into leaf ids in-place
  leaf_ids_type_map = {leaf_id: _leaf_to_type_desc(leaf) for (leaf_id, leaf)
                       in safe_zip(leaf_ids_flat, data_flat)}
  _inscribe_leaf_types(pytree_repr[utils._TREE_REPR_KEY], leaf_ids_type_map)
  pytree_repr[utils._LEAF_IDS_KEY] = [
    _join_leaf_type_and_id(leaf_ids_type_map[leaf_id], leaf_id)
    for leaf_id in leaf_ids_flat]

def _combine_two_pytrees(dest: dict[str, Any], source: dict[str, Any]):
  """Combine two pytrees in JSON format, in-place into dest."""
  assert dest["node_type"] == source["node_type"]
  if dest["node_type"] == "leaf":
    _, id1 = dest["leaf_id"].split(_TYPE_ID_LEAF_DELIMITER, 1)
    type2, id2 = source["leaf_id"].split(_TYPE_ID_LEAF_DELIMITER, 1)
    assert id1 == id2
    if type2 != "null":
      dest["leaf_id"] = source["leaf_id"]
    return dest
  else:
    dest["children"] = [
      _combine_two_pytrees(child1, child2) for (child1, child2)
      in safe_zip(dest["children"], source["children"])]
    return dest

def _is_remote_path(path: str | PathLike[str]):
  """Check whether a path is remote by examining the prefix."""
  # we need to truncate e.g., gs:// to gs:/ because pathlib.Path collapses //
  return any(str(path).startswith(prefix[:-1])
             for prefix in _REMOTE_URL_PREFIXES)

def _rm_dir(root: Path) -> None:
  if _is_remote_path(root):
    root.rmtree()  # pytype: disable=attribute-error
  else:
    shutil.rmtree(root)

def _set_up_destination(root: Path, overwrite: bool, partial_write: bool,
                              pytree_repr: dict[str, Any],
                              distinct_locations: bool, sync_key: str | None
                              ) -> dict[str, Any]:
  """Inspect the destination, set it up for writing, potentially read existing data."""
  if overwrite:
    if root.exists() and len(list(root.iterdir())) > 0:
      # check that we're only deleting things that come from JAX
      # refuse to rm directories containing additional entries
      paths_present = list(root.iterdir())
      extra_member_paths = [path for path in paths_present if path.name not in
                            (_PYTREEDEF_FILE, _ARCHIVE_NAME,
                             _ARRAY_STORE_DIRNAME)]

      assert len(extra_member_paths) == 0, (
        "Refusing to work on a directory that is not a previous checkpoint."
        f" Unrecognized paths: {extra_member_paths}. Remove them manually if"
        f" you're sure you want to use {root} as the checkpoint directory.")

      if partial_write and Path(_PYTREEDEF_FILE) in [path.relative_to(root)
                                                     for path in paths_present]:
        try:
          other_pytree = json.loads((root / _PYTREEDEF_FILE).read_text())
          assert other_pytree[utils._LEAF_COUNT_KEY] == pytree_repr[
              utils._LEAF_COUNT_KEY]
          tree_repr = _combine_two_pytrees(other_pytree[utils._TREE_REPR_KEY],
                                           pytree_repr[utils._TREE_REPR_KEY])
          new_leaf_ids = [y if not y.startswith("null") else x
                          for x, y in safe_zip(
                              other_pytree[utils._LEAF_IDS_KEY],
                              pytree_repr[utils._LEAF_IDS_KEY])]
          return {utils._TREE_REPR_KEY: tree_repr,
                  utils._LEAF_COUNT_KEY: pytree_repr[utils._LEAF_COUNT_KEY],
                  utils._LEAF_IDS_KEY: new_leaf_ids}
        except AssertionError:
          logger.warning("The previous pytree does not match, overwritting"
                         " existing data.")
      if ((jax.process_index() == 0 or distinct_locations) and root.exists()
          and not partial_write):
        _rm_dir(root)
    _sync_on_key(sync_key, "overwrite")
    return pytree_repr
  else:
    if (root.exists() and len(list(root.iterdir())) > 0):  # not empty
      raise ValueError(f"Files already exist at path: `{root}`, but you"
                       f" specified `{overwrite=}`")
    return pytree_repr

def _prepare_directory(directory: Path, overwrite: bool, partial_write: bool,
                       pytreedef_repr: dict[str, Any],
                       distinct_locations: bool, sync_key: str | None):
  """Prepare the directory: check destination, potentially read existing data
  and overwrite.
  """
  directory = Path(directory)
  # prepare the destination directory, overwrite destination directory or error
  root = Path(directory).resolve()
  pytreedef_repr = _set_up_destination(root, overwrite, partial_write,
                                       pytreedef_repr, distinct_locations,
                                       sync_key)
  if not _is_remote_path(directory):
    if distinct_locations or jax.process_index() == 0:
      root.mkdir(exist_ok=True)  # do not make parents, that's too much
      assert root.exists() and root.is_dir()
  _sync_on_key(sync_key, "mkdir")
  return pytreedef_repr

def _obj_serialize(archive: utils.MemKVStore, filename_id: str | int, x: Any,
                   registry: utils.SerializationRegistry) -> None:
  """Serialization method for NOT-array objects."""
  # we're only interested in name and suffix
  filename = Path(str(filename_id)).with_suffix("")
  serialization_registry = registry
  if _is_array_like(x):
    raise ValueError(
        "Arrays cannot be serialized using this method for non-arrays.")
  # dispatch the serialization method in a thread to yield async control
  payload, method_name = serialization_registry.serialize(x)
  suffix = "." + method_name.lstrip(".").replace(".", _DOT_REPLACEMENT)
  archive.write(filename.with_suffix(suffix), payload)

def _obj_deserialize(archive: utils.MemKVStore, filename: str,
                     registry: utils.SerializationRegistry,
                     best_effort: bool = False) -> Any:
  """Deserialization method for NON-array objects."""
  path = Path(filename)
  payload = archive.read(path)
  method_name = str(path.suffix).lstrip(".").replace(_DOT_REPLACEMENT, ".")
  try:
    return registry.deserialize(payload, method_name)
  except ValueError as exc:
    if best_effort:
      logging.warning("Unrecognized data type `%s` we'll do our best and just"
                      " return the raw bytes", method_name)
      return payload
    else:
      raise exc

async def serialize_array(arr, path, extra_config, distinct_locations: bool
                          ) -> None:
  arr = jax.numpy.asarray(arr, dtype=arr.dtype)
  extra_ts_spec = extra_config
  process_num = (jax.process_index() if (
      jax.process_count() > 1 and not distinct_locations) else None)
  default_ts_spec = ts_impl.get_tensorstore_spec(
      path, ocdbt=_USE_OCDBT, process_num=process_num, arr=arr)
  expected_path = default_ts_spec['kvstore']['base']['path']
  ts_spec = ts_impl.merge_nested_specs(default_ts_spec, extra_ts_spec)
  ts_impl.verify_tensorstore_spec(ts_spec, arr, expected_path,
                                  check_metadata=True)
  # all hosts write because they're writing to different storage locations (to
  # be combined later) -> `primary_host=None`
  await ts_impl.async_serialize(arr, ts_spec, primary_host=None)

def _finalize_array_store(kvstore_path, extra_config, distinct_locations: bool
                         ) -> None:
  """When multiple processes are writing, they must write to a per-processlocation
  followed by combining them via no-copy links to the final location.
  """
  # only in multiprocess case and only process 0
  if distinct_locations or jax.process_count() <= 1 or jax.process_index() != 0:
    return
  extra_ts_spec = extra_config
  dummy_key_path = os.path.join(kvstore_path, "dummy_key")
  combined_ts_spec = ts_impl.merge_nested_specs(ts_impl.get_tensorstore_spec(
      dummy_key_path, ocdbt=_USE_OCDBT, process_num=None), extra_ts_spec)
  children_ts_spec = [ts_impl.merge_nested_specs(ts_impl.get_tensorstore_spec(
      dummy_key_path, ocdbt=_USE_OCDBT, process_num=i), extra_ts_spec)
      for i in range(jax.process_count())]
  combined_kvstore = combined_ts_spec["kvstore"]
  children_kvstores = [ts_spec["kvstore"] for ts_spec in children_ts_spec]
  _ = combined_kvstore.pop("path")
  _ = [kvstore.pop("path") for kvstore in children_kvstores]
  asyncio.run(ts_impl.combine_kvstores(combined_kvstore, children_kvstores))

async def deserialize_array(
    path: str | PathLike[str], sharding: jax.sharding.Sharding | Layout,
    ts_spec: dict[str, Any],
    byte_limiter: asyncio_utils._LimitInFlightBytes | None = None) -> jax.Array:
  """Deserialize an array from a given path with an optional extra tensorstore spec."""
  # every process reads from the central location
  default_ts_spec = ts_impl.get_tensorstore_spec(
      path, ocdbt=_USE_OCDBT, process_num=None)
  expected_path = default_ts_spec['kvstore']['base']['path']
  ts_spec = ts_impl.merge_nested_specs(default_ts_spec, ts_spec)
  ts_impl.verify_tensorstore_spec(ts_spec, arr=None, path=expected_path,
                                  check_metadata=False)
  return await ts_impl.async_deserialize(sharding, ts_spec,
                                         byte_limiter=byte_limiter)

def _write_pytreedef(directory: os.PathLike[str], archive: utils.MemKVStore,
                     pytree_repr: dict[str, Any],
                     node_aux_data_store: dict[str, Any],
                     distinct_locations: bool):
  """Write the pytreedef to the desitination directory and aux data to the archive."""
  if not (jax.process_index() == 0 or distinct_locations):
    return
  root = Path(directory)
  pytree_repr_json = json.dumps(pytree_repr, indent=2)
  (root / _PYTREEDEF_FILE).write_text(pytree_repr_json)
  for node_data_key, node_data_bytes in node_aux_data_store.items():
    # this KV storage only admits sequential writes, no need for threads
    archive.write(f"{_NODE_AUX_DATA_PREFIX}/{node_data_key}", node_data_bytes)

def _write_objects(archive: utils.MemKVStore,
                   objs_and_ids: list[tuple[Any, int]],
                   serialization_registry: utils.SerializationRegistry,
                   distinct_locations: bool):
  """Write objects to the in-memory archive."""
  if not (jax.process_index() == 0 or distinct_locations):
    return
  for obj, leaf_id in objs_and_ids:
    _obj_serialize(archive, f"{_OBJ_DATA_PREFIX}/{leaf_id}", obj,
                   serialization_registry)

def _write_arrays(arrs_and_paths: list[tuple[Any, Path]],
                  full_ts_specs: list[Any | None],
                  distinct_locations: bool):

  async def _serialize_arrays():
    await asyncio.gather(*[serialize_array(
      arr, path, extra_ts_spec, distinct_locations)
      for ((arr, path), extra_ts_spec)
      in safe_zip(arrs_and_paths, full_ts_specs)])

  asyncio.run(_serialize_arrays())

def _finalize_archive(root: Path, archive: utils.MemKVStore,
                      distinct_locations: bool,
                      depends_futs: Iterable[Future[Any]]) -> None:
  """Write the archive to disk after node aux data and objects are written to it."""
  if not (jax.process_index() == 0 or distinct_locations):
    return
  _ = [fut.result() for fut in depends_futs]
  archive_path = root / _ARCHIVE_NAME
  archive_path.write_bytes(archive.tobytes())

def save(data: PyTreeT, directory: str | PathLike[str], overwrite: bool = True,
         partial_write: bool = False, ts_specs: PyTreeT | None = None) -> None:
  """Saves the given data structure to the provided directory path.

  This function provides functionality to serialize and save a data structure
  comprising JAX arrays, NumPy arrays, Python objects, etc., along with its
  structure to a given directory. It leverages `PyTree` for flattening and
  reconstructing the data structure.

  .. code-block:: python

    data = {'a': jnp.array([1, 3]), 'b': {'c': [jnp.array([4, 5]), "hello"]}}
    pytree_serialization.save(data, directory)

  Args:
    data: The data structure to be saved. Arbitrary composition of JAX arrays,
      NumPy arrays, and Python objects, including nested structures.
    directory: The directory path where the data will be saved. A local path or
      a remote URL (e.g., gs://, s3://). For remote URLs, `etils` is required.
    overwrite: If True, any existing directory with the same name will be
      overwritten.

  .. code-block:: python
    data = {"a": jnp.array([1, 2]), "b": None}
    save(data, directory)

    # partial write is also supported
    data = {"a": None, "b": None}
    save(data, directory, partial_write=True)
  """
  with _THREADING_SAVE_LOCK:
    return _save(data, directory, overwrite, partial_write, ts_specs)

def _save(data: PyTreeT, directory: str | PathLike[str], overwrite: bool = True,
         partial_write: bool = False, ts_specs: PyTreeT | None = None) -> None:
  sync_key = _get_unique_sync_key()  # get a synchronization key for multi-host

  assert not _is_remote_path(directory) or epath_installed, (
    "For saving to remote URLs (e.g., gs, s3) you need the `etils` module"
    "installed. You can install it using `pip install etils`.")
  data_flat, pytreedef = jax.tree.flatten(data, is_leaf=lambda x: x is None)
  distinct_locations = not _is_str_same_on_all_hosts(directory)
  if jax.process_count() > 1 and distinct_locations:
    logger.warning("Saving to different locations on different hosts is"
                   " supported, but extremely fragile. Consider using a single"
                   " location.")
  root = Path(directory).resolve()

  # start serialization ##################################
  futures, executor = [], ThreadPoolExecutor(max_workers=utils._MAX_CONCURRENCY)
  serialization_registry = utils.default_serialization_registry.copy()
  archive_path, archive_data = root / _ARCHIVE_NAME, None
  if (jax.process_index() == 0 or distinct_locations) and partial_write:
    if archive_path.exists():
      archive_data = (root / _ARCHIVE_NAME).read_bytes()
  archive = utils.MemKVStore(archive_data)

  # 0. serialize the pytree
  pytreedef_repr, leaf_count, node_aux_data_store = utils.serialize_pytreedef(
    pytreedef, registry=serialization_registry)
  leaf_ids_flat = list(range(leaf_count))
  # augment the pytree representation with leaf types
  _inplace_add_types_to_pytree_repr(pytreedef_repr, leaf_ids_flat, data_flat)

  pytreedef_repr = _prepare_directory(root, overwrite, partial_write,
                                      pytreedef_repr, distinct_locations,
                                      sync_key)
  pytreedef_fut = executor.submit(_write_pytreedef, root, archive,
                                 pytreedef_repr, node_aux_data_store,
                                 distinct_locations)
  futures.append(pytreedef_fut)

  # 1. serialize non-array (objects) in the pytree
  objs_and_ids = [(data, leaf_id)
                  for data, leaf_id in safe_zip(data_flat, leaf_ids_flat)
                  if not _is_array_like(data) and data is not None]
  objs_fut = executor.submit(_write_objects, archive, objs_and_ids,
                                 serialization_registry,
                                 distinct_locations)
  futures.append(objs_fut)

  # write the archive to disk
  futures.append(executor.submit(_finalize_archive, root, archive,
                                 distinct_locations, (pytreedef_fut, objs_fut)))

  # 2. serialize arrays
  array_store_path = root / _ARRAY_STORE_DIRNAME
  arrs_and_paths = [(data, array_store_path / str(leaf_id)) for data, leaf_id in
                    safe_zip(data_flat, leaf_ids_flat) if _is_array_like(data)]
  full_ts_specs = (([None] * len(arrs_and_paths)) if ts_specs is None else
                   jax.tree.leaves(ts_specs, ts_impl.is_tensorstore_spec_leaf))
  futures.append(executor.submit(_write_arrays, arrs_and_paths, full_ts_specs,
                                 distinct_locations))

  _ = [fut.result() for fut in futures]
  _sync_on_key(sync_key, "array_serialization")
  if len(arrs_and_paths) > 0:
    _finalize_array_store(array_store_path, full_ts_specs[0],
                          distinct_locations)
  # we are done with all async ops here, we can block
  _sync_on_key(sync_key, "end")

def _read_objects(archive: utils.MemKVStore, obj_leaf_ids: list[int],
                 serialization_registry: utils.SerializationRegistry,
                 best_effort: bool = False):
  _key2id = lambda x: int(Path(x).stem)
  obj_keys = [key for key in archive.keys() if key.startswith(_OBJ_DATA_PREFIX)]
  missing_leaf_ids = set(obj_leaf_ids) - set(map(_key2id, obj_keys))
  requested_obj_keys = [obj_key for obj_key in obj_keys
                        if _key2id(obj_key) in obj_leaf_ids]
  if len(missing_leaf_ids) > 0:
    raise ValueError(
      f"Values {missing_leaf_ids} are missing from the checkpoint directory."
      f" Existing keys: {obj_keys}, requested keys: {obj_leaf_ids}, all keys: {list(archive.keys())}")
  obj_values = [_obj_deserialize(
      archive, obj_key, registry=serialization_registry,
      best_effort=best_effort) for obj_key in requested_obj_keys]
  return dict(safe_zip(map(_key2id, requested_obj_keys), obj_values))

def _read_arrays(array_store_path: Path, arr_leaf_ids: list[int],
                 ts_specs: Any | None,
                 shardings: PyTreeT | utils._MISSING_TYPE):
  # array_store_path = root / _LEAF_DATA_DIR / _ARRAY_STORE_DIRNAME
  arr_store_path = Path(array_store_path)
  arr_paths = [arr_store_path / str(leaf_id) for leaf_id in arr_leaf_ids]
  # missing sharding assumes we want to deserialize on default device
  if shardings is utils.MISSING:
    device = jax.devices()[0]  # default device
    shardings = [SingleDeviceSharding(device) for _ in arr_paths]
  else:
    shardings = jax.tree.flatten(shardings, is_leaf=lambda x: x is None)[0]
    assert len(shardings) == len(arr_paths), (
      "The sharding leaves must match the load arrays requested.")
  full_ts_specs = (([None] * len(arr_paths))
                   if ts_specs is None else jax.tree.leaves(
                       ts_specs, is_leaf=ts_impl.is_tensorstore_spec_leaf))
  byte_limiter = asyncio_utils._LimitInFlightBytes(100 * 1024 ** 3)  # 100 GB
  async def _deserialize_arrays():
    return await asyncio.gather(*[
        deserialize_array(path, sharding, ts_spec, byte_limiter)
        for (path, sharding, ts_spec)
        in safe_zip(arr_paths, shardings, full_ts_specs)])

  arr_keys = [int(path.stem) for path in arr_paths]

  # finally, collect the results
  arrs = dict(zip(arr_keys, asyncio.run(_deserialize_arrays())))
  return arrs


def load(directory: str | PathLike[str],
         shardings: PyTreeT | utils._MISSING_TYPE = utils.MISSING,
         pytree: PyTreeT | utils._MISSING_TYPE = utils.MISSING,
         ts_specs: PyTreeT | None = None, best_effort: bool = False) -> PyTreeT:
  """Loads and reconstructs a data structure from a directory.

  Args:
    directory: Directory path where the data is stored.
    shardings: Sharding strategy for array objects. If None, defaults to
      single device sharding on the default device.
    pytree: Optional pre-populated PyTree for structure. If provided, must
      specify a pytree with string object ids. Useful for partial reads.
    best_effort: Proceed with deserialization even in the face of partial
      failures. Return custom nodes as a list of children.
  Returns:
    Reconstructed data structure.

  .. code-block:: python
    data = load(directory)
    # deserialize ignoring unknonwn nodes (return custom nodes as a list of children)
    data = load(directory, best_effort=True)
  """
  assert not _is_remote_path(directory) or epath_installed, (
    "For checkpointing using remote URLs (e.g., gs, s3) you need `etils`"
    " module installed. You can install it using `pip install etils`.")

  root = Path(directory).resolve()
  assert root.is_dir(), f"Checkpoint directory {root} does not exist"
  if not _is_remote_path(root):
    root = root.resolve()
  archive = utils.MemKVStore(data=Path(root / _ARCHIVE_NAME).read_bytes())

  # deserialize in 3 stages

  # 1. deserialize PyTreeDef
  if pytree is utils.MISSING:
    pytree = load_pytreedef(directory, archive=archive, best_effort=best_effort)
    # in pytreedef, None leafs indicate StaticNodes (without leaves)
    # so we CANNOT flatten with is_leaf=lambda x: x is None
    leaf_ids_flat, pytree = jax.tree.flatten(pytree)
  else:
    leaf_ids_flat, pytree = jax.tree.flatten(pytree,
                                                is_leaf=lambda x: x is None)
  serialization_registry = utils.default_serialization_registry.copy()
  executor = ThreadPoolExecutor(max_workers=utils._MAX_CONCURRENCY)

  # 2. deserialize non-array objects
  obj_leaf_ids = [int(leaf_id.split(_TYPE_ID_LEAF_DELIMITER, 1)[1]) for leaf_id
                  in leaf_ids_flat if leaf_id is not None
                  and not leaf_id.startswith(_ARRAY_TYPE_NAME)
                  and not leaf_id.startswith("null")]
  objs_fut = executor.submit(_read_objects, archive, obj_leaf_ids,
                             serialization_registry, best_effort=best_effort)

  # 3. deserialize array objects
  arr_leaf_ids = [int(leaf_id.split(_TYPE_ID_LEAF_DELIMITER, 1)[1]) for leaf_id
                  in leaf_ids_flat if leaf_id is not None
                  and leaf_id.startswith(_ARRAY_TYPE_NAME)]
  arrs_fut = executor.submit(_read_arrays, root / _ARRAY_STORE_DIRNAME,
                            arr_leaf_ids, ts_specs, shardings)

  objs, arrs = objs_fut.result(), arrs_fut.result()
  arr_and_objs = arrs | objs
  filled_values = [arr_and_objs.get(leaf_id, None)
                   for leaf_id in range(len(leaf_ids_flat))]
  return jax.tree.unflatten(pytree, filled_values)

def load_pytreedef(directory: str | PathLike[str],
                   archive: utils.MemKVStore | None = None,
                   best_effort: bool = False) -> PyTreeDef:
  """Loads a pytree from the given directory.
  Args:
    directory: Directory path to load from.
    best_effort: Proceed with deserialization even in the face of partial
      failures. Return custom nodes as a list of children.
  Returns:
    The loaded pytree.
  """
  assert not _is_remote_path(directory) or epath_installed, (
    "For checkpointing using remote URLs (e.g., gs, s3) you need `etils`"
    " module installed. You can install it using `pip install etils`.")
  root = Path(directory).resolve()
  if archive is None:
    archive = utils.MemKVStore(data=Path(root / _ARCHIVE_NAME).read_bytes())
  json_content = (root / _PYTREEDEF_FILE).read_text()
  raw_tree = json.loads(json_content)
  serialization_registry = utils.default_serialization_registry.copy()
  executor = ThreadPoolExecutor(max_workers=utils._MAX_CONCURRENCY)
  node_aux_data_futs = {Path(k).name: executor.submit(archive.read, k)
                        for k in archive.keys()
                        if k.startswith(_NODE_AUX_DATA_PREFIX)}
  node_aux_data_store = {k: v.result() for k, v in node_aux_data_futs.items()}
  return utils.deserialize_pytreedef(raw_tree, node_aux_data_store,
                                     registry=serialization_registry,
                                     best_effort=best_effort)

def _pytree_leaf_desc(leaf):
  if isinstance(leaf, (np.ndarray, jax.Array)):
    return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype)
  else:
    return leaf

nonblocking_executor = ThreadPoolExecutor(max_workers=utils._MAX_CONCURRENCY)

def nonblocking_save(data: PyTreeT, directory: str | PathLike[str],
                     overwrite: bool = True, partial_write: bool = False,
                     tensorstore_specs: PyTreeT | None = None,
                     ) -> asyncio_utils.AwaitableFuture:
  """Start the serialization without blocking, return a
  concurrent.futures.Future future to the result.

  .. code-block:: python
    fut = nonblocking_save(data, directory)
    print(fut.pytree)  # a pytree of jax.ShapeDtypeStruct
    print(fut.result())  # None, blocking until the serialization is done
  """
  # start serialization immediately
  fut = asyncio_utils.AwaitableFuture(nonblocking_executor.submit(
      save, data, directory, overwrite, partial_write, tensorstore_specs))
  # construct a nice looking pytree representing the nodes being read
  fut.pytree = jax.tree.map(_pytree_leaf_desc, data)
  return fut


_is_desc_array = lambda x: (re.match(_ARRAY_TYPE_REGEX,
                                     x.split(_TYPE_ID_LEAF_DELIMITER, 1)[0])
                            is not None)
_none_is_leaf = lambda x: x is None

def nonblocking_load(directory: str | PathLike[str],
                     shardings: PyTreeT | utils._MISSING_TYPE = utils.MISSING,
                     pytreedef: PyTreeT | utils._MISSING_TYPE = utils.MISSING,
                     tensorstore_specs: PyTreeT | None = None,
                     best_effort: bool = False
                     ) -> asyncio_utils.AwaitableFuture:
  """Start deserialization without blocking, return a
  concurrent.futures.Future future to the result with a pytree stub

  .. code-block:: python
    fut = nonblocking_load(directory)
    print(fut.pytree)  # a pytree of jax.ShapeDtypeStruct
    print(fut.result())  # the fully populated pytree
  """
  if pytreedef is utils.MISSING:
    pytreedef = load_pytreedef(directory, best_effort=best_effort)

  # read in all the objects synchronously, but arrays asynchronously
  arr_pytree = jax.tree.map(lambda x: x if _is_desc_array(x) else None,
                            pytreedef)
  obj_pytree = jax.tree.map(lambda x: None if _is_desc_array(x) else x,
                            pytreedef)
  arr_shapes = jax.tree.map(_leaf_desc_to_leaf, arr_pytree)  # skip None-s here
  obj_data = load(directory, pytree=obj_pytree, best_effort=best_effort)
  pytree_stub = jax.tree.map(lambda x, y: x if x is not None else y,
                             arr_shapes, obj_data, is_leaf=_none_is_leaf)

  # TODO(rdyro): the awaitable future output is a workaround
  # it should return the fully populated pytree instead of just
  # jax.ShapeDtypeStruct for arrays by constructing them asynchronously
  fut = asyncio_utils.AwaitableFuture(nonblocking_executor.submit(
      load, directory, shardings, pytreedef, tensorstore_specs, best_effort))
  fut.pytree = pytree_stub
  return fut
