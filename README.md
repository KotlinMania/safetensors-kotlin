# safetensors-kotlin in Kotlin

[![GitHub link](https://img.shields.io/badge/GitHub-KotlinMania%2Fsafetensors-kotlin-blue.svg)](https://github.com/KotlinMania/safetensors-kotlin)
[![Maven Central](https://img.shields.io/maven-central/v/io.github.kotlinmania/safetensors-kotlin)](https://central.sonatype.com/artifact/io.github.kotlinmania/safetensors-kotlin)
[![Build status](https://img.shields.io/github/actions/workflow/status/KotlinMania/safetensors-kotlin/ci.yml?branch=main)](https://github.com/KotlinMania/safetensors-kotlin/actions)

This is a Kotlin Multiplatform line-by-line transliteration port of [`safetensors`](https://github.com/huggingface/safetensors).

**Original Project:** This port is based on [`safetensors`](https://github.com/huggingface/safetensors). All design credit and project intent belong to the upstream authors; this repository is a faithful port to Kotlin Multiplatform with no behavioural changes intended.

### Porting status

This is an **in-progress port**. The goal is feature parity with the upstream Rust crate while providing a native Kotlin Multiplatform API. Every Kotlin file carries a `// port-lint: source <path>` header naming its upstream Rust counterpart so the AST-distance tool can track provenance.

---

## Upstream README — `safetensors`

> The text below is reproduced from [`https://github.com/huggingface/safetensors`](https://github.com/huggingface/safetensors). It is the upstream project's own description and remains under the upstream authors' authorship; links have been rewritten to absolute upstream URLs where appropriate so they continue to resolve from this repository.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/safetensors/assets/raw/main/banner-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/datasets/safetensors/assets/raw/main/banner-light.svg">
    <img alt="Hugging Face Safetensors Library" src="https://huggingface.co/datasets/safetensors/assets/raw/main/banner-light.svg" style="max-width: 100%;">
  </picture>
  <br/>
  <br/>
</p>

Python
[![Pypi](https://img.shields.io/pypi/v/safetensors.svg)](https://pypi.org/pypi/safetensors/)
[![Documentation](https://img.shields.io/website/http/huggingface.co/docs/safetensors/index.svg?label=docs)](https://huggingface.co/docs/safetensors/index)
[![Downloads](https://static.pepy.tech/badge/safetensors/month)](https://pepy.tech/project/safetensors)

Rust
[![Crates.io](https://img.shields.io/crates/v/safetensors.svg)](https://crates.io/crates/safetensors)
[![Documentation](https://docs.rs/safetensors/badge.svg)](https://docs.rs/safetensors/)
[![Dependency status](https://deps.rs/repo/github/huggingface/safetensors/status.svg?path=safetensors)](https://deps.rs/repo/github/huggingface/safetensors?path=safetensors)

# safetensors

## Safetensors

This repository implements a new simple format for storing tensors
safely (as opposed to pickle) and that is still fast (zero-copy).

### Installation
#### Pip

You can install safetensors via the pip manager:

```bash
pip install safetensors
```

#### From source

For the sources, you need Rust

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# Make sure it's up to date and using stable channel
rustup update
git clone https://github.com/huggingface/safetensors
cd safetensors/bindings/python
pip install setuptools_rust
pip install -e .
```

### Getting started

```python
import torch
from safetensors import safe_open
from safetensors.torch import save_file

tensors = {
   "weight1": torch.zeros((1024, 1024)),
   "weight2": torch.zeros((1024, 1024))
}
save_file(tensors, "model.safetensors")

tensors = {}
with safe_open("model.safetensors", framework="pt", device="cpu") as f:
   for key in f.keys():
       tensors[key] = f.get_tensor(key)
```

[Python documentation](https://huggingface.co/docs/safetensors/index)


### Format

- 8 bytes: `N`, an unsigned little-endian 64-bit integer, containing the size of the header
- N bytes: a JSON UTF-8 string representing the header.
  - The header data MUST begin with a `{` character (0x7B).
  - The header data MAY be trailing padded with whitespace (0x20).
  - The header is a dict like `{"TENSOR_NAME": {"dtype": "F16", "shape": [1, 16, 256], "data_offsets": [BEGIN, END]}, "NEXT_TENSOR_NAME": {...}, ...}`,
    - `data_offsets` point to the tensor data relative to the beginning of the byte buffer (i.e. not an absolute position in the file),
      with `BEGIN` as the starting offset and `END` as the one-past offset (so total tensor byte size = `END - BEGIN`).
  - A special key `__metadata__` is allowed to contain free form string-to-string map. Arbitrary JSON is not allowed, all values must be strings.
- Rest of the file: byte-buffer.

Notes:
 - Duplicate keys are disallowed. Not all parsers may respect this.
 - In general the subset of JSON is implicitly decided by `serde_json` for
   this library. Anything obscure might be modified at a later time, that odd ways
   to represent integer, newlines and escapes in utf-8 strings. This would only
   be done for safety concerns
 - Tensor values are not checked against, in particular NaN and +/-Inf could
   be in the file
 - Empty tensors (tensors with 1 dimension being 0) are allowed.
   They are not storing any data in the databuffer, yet retaining size in the header.
   They don't really bring a lot of values but are accepted since they are valid tensors
   from traditional tensor libraries perspective (torch, tensorflow, numpy, ..).
 - 0-rank Tensors (tensors with shape `[]`) are allowed, they are merely a scalar.
 - The byte buffer needs to be entirely indexed, and cannot contain holes. This prevents
   the creation of polyglot files.
 - Endianness: Little-endian.
   moment.
 - Order: 'C' or row-major.
 - Notes: Some smaller than 1 byte dtypes appeared, which make alignment tricky. Non traditional APIs might be required for those.


### Yet another format ?

The main rationale for this crate is to remove the need to use
`pickle` on `PyTorch` which is used by default.
There are other formats out there used by machine learning and more general
formats.


Let's take a look at alternatives and why this format is deemed interesting.
This is my very personal and probably biased view:

| Format                  | Safe | Zero-copy | Lazy loading | No file size limit | Layout control | Flexibility | Bfloat16/Fp8
| ----------------------- | --- | --- | --- | --- | --- | --- | --- |
| pickle (PyTorch)        | ✗ | ✗ | ✗ | 🗸 | ✗ | 🗸 | 🗸 |
| H5 (Tensorflow)         | 🗸 | ✗ | 🗸 | 🗸 | ~ | ~ | ✗ |
| SavedModel (Tensorflow) | 🗸 | ✗ | ✗ | 🗸 | 🗸 | ✗ | 🗸 |
| MsgPack (flax)          | 🗸 | 🗸 | ✗ | 🗸 | ✗ | ✗ | 🗸 |
| Protobuf (ONNX)         | 🗸 | ✗ | ✗ | ✗ | ✗ | ✗ | 🗸 |
| Cap'n'Proto             | 🗸 | 🗸 | ~ | 🗸 | 🗸 | ~ | ✗ |
| Arrow                   | ? | ? | ? | ? | ? | ? | ✗ |
| Numpy (npy,npz)         | 🗸 | ? | ? | ✗ | 🗸 | ✗ | ✗ |
| pdparams (Paddle)       | ✗ | ✗ | ✗ | 🗸 | ✗ | 🗸 | 🗸 |
| SafeTensors             | 🗸 | 🗸 | 🗸 | 🗸 | 🗸 | ✗ | 🗸 |

- Safe: Can I use a file randomly downloaded and expect not to run arbitrary code ?
- Zero-copy: Does reading the file require more memory than the original file ?
- Lazy loading: Can I inspect the file without loading everything ? And loading only
  some tensors in it without scanning the whole file (distributed setting) ?
- Layout control: Lazy loading, is not necessarily enough since if the information about tensors is spread out in your file, then even if the information is lazily accessible you might have to access most of your file to read the available tensors (incurring many DISK -> RAM copies). Controlling the layout to keep fast access to single tensors is important.
- No file size limit: Is there a limit to the file size ?
- Flexibility: Can I save custom code in the format and be able to use it later with zero extra code ? (~ means we can store more than pure tensors, but no custom code)
- Bfloat16/Fp8: Does the format support native bfloat16/fp8 (meaning no weird workarounds are
  necessary)? This is becoming increasingly important in the ML world.


### Main oppositions

- Pickle: Unsafe, runs arbitrary code
- H5: Apparently now discouraged for TF/Keras. Seems like a great fit otherwise actually. Some classic use after free issues: <https://www.cvedetails.com/vulnerability-list/vendor_id-15991/product_id-35054/Hdfgroup-Hdf5.html>. On a very different level than pickle security-wise. Also 210k lines of code vs ~400 lines for this lib currently.
- SavedModel: Tensorflow specific (it contains TF graph information).
- MsgPack: No layout control to enable lazy loading (important for loading specific parts in distributed setting)
- Protobuf: Hard 2Go max file size limit
- Cap'n'proto: Float16 support is not present [link](https://capnproto.org/language.html#built-in-types) so using a manual wrapper over a byte-buffer would be necessary. Layout control seems possible but not trivial as buffers have limitations [link](https://stackoverflow.com/questions/48458839/capnproto-maximum-filesize).
- Numpy (npz): No `bfloat16` support. Vulnerable to zip bombs (DOS). Not zero-copy.
- Arrow: No `bfloat16` support.

### Notes

- Zero-copy: No format is really zero-copy in ML, it needs to go from disk to RAM/GPU RAM (that takes time). On CPU, if the file is already in cache, then it can
  truly be zero-copy, whereas on GPU there is not such disk cache, so a copy is always required
  but you can bypass allocating all the tensors on CPU at any given point.
  SafeTensors is not zero-copy for the header. The choice of JSON is pretty arbitrary, but since deserialization is <<< of the time required to load the actual tensor data and is readable I went that way, (also space is <<< to the tensor data).

- Endianness: Little-endian. This can be modified later, but it feels really unnecessary at the
  moment.
- Order: 'C' or row-major. This seems to have won. We can add that information later if needed.
- Stride: No striding, all tensors need to be packed before being serialized. I have yet to see a case where it seems useful to have a strided tensor stored in serialized format.
 - Sub 1 bytes dtypes: Dtypes can now have lower than 1 byte size, this makes alignment&adressing tricky. For now, the library will simply error out whenever an operation triggers an non aligned read. Trickier API may be created later for those non standard ops. 

### Benefits

Since we can invent a new format we can propose additional benefits:

- Prevent DOS attacks: We can craft the format in such a way that it's almost
  impossible to use malicious files to DOS attack a user. Currently, there's a limit
  on the size of the header of 100MB to prevent parsing extremely large JSON.
  Also when reading the file, there's a guarantee that addresses in the file
  do not overlap in any way, meaning when you're loading a file you should never
  exceed the size of the file in memory

- Faster load: PyTorch seems to be the fastest file to load out in the major
  ML formats. However, it does seem to have an extra copy on CPU, which we
  can bypass in this lib by using `torch.UntypedStorage.from_file`.
  Currently, CPU loading times are extremely fast with this lib compared to pickle.
  GPU loading times are as fast or faster than PyTorch equivalent.
  Loading first on CPU with memmapping with torch, and then moving all tensors to GPU seems
  to be faster too somehow (similar behavior in torch pickle)

- Lazy loading: in distributed (multi-node or multi-gpu) settings, it's nice to be able to
  load only part of the tensors on the various models. For
  [BLOOM](https://huggingface.co/bigscience/bloom) using this format enabled
  to load the model on 8 GPUs from 10mn with regular PyTorch weights down to 45s.
  This really speeds up feedbacks loops when developing on the model. For instance
  you don't have to have separate copies of the weights when changing the distribution
  strategy (for instance Pipeline Parallelism vs Tensor Parallelism).

License: Apache-2.0

---

## About this Kotlin port

### Installation

```kotlin
dependencies {
    implementation("io.github.kotlinmania:safetensors-kotlin:0.1.0")
}
```

### Building

```bash
./gradlew build
./gradlew test
```

### Targets

- macOS arm64
- Linux x64
- Windows mingw-x64
- iOS arm64 / simulator-arm64 (Swift export + XCFramework)
- JS (browser + Node.js)
- Wasm-JS (browser + Node.js)
- Android (API 24+)

### Porting guidelines

See [AGENTS.md](AGENTS.md) and [CLAUDE.md](CLAUDE.md) for translator discipline, port-lint header convention, and Rust → Kotlin idiom mapping.

### License

This Kotlin port is distributed under the same license terms as the upstream [`safetensors`](https://github.com/huggingface/safetensors). See [LICENSE](LICENSE) (and any sibling `LICENSE-*` / `NOTICE` files mirrored from upstream) for the full text.

Original work copyrighted by the safetensors authors.  
Kotlin port: Copyright (c) 2026 Sydney Renee and The Solace Project.

### Acknowledgments

Thanks to the [`safetensors`](https://github.com/huggingface/safetensors) maintainers and contributors for the original Rust implementation. This port reproduces their work in Kotlin Multiplatform; bug reports about upstream design or behavior should go to the upstream repository.
