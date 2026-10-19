// port-lint: tests tensor.rs
package io.github.kotlinmania.safetensors

import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

private const val MAX_DIMENSION: Int = 8
private const val MAX_SIZE: Int = 8
private const val MAX_TENSORS: Int = 8

private fun floatToLe(value: Float): ByteArray {
    val bits = value.toRawBits()
    return byteArrayOf(
        (bits and 0xFF).toByte(),
        ((bits ushr 8) and 0xFF).toByte(),
        ((bits ushr 16) and 0xFF).toByte(),
        ((bits ushr 24) and 0xFF).toByte(),
    )
}

private fun floatsToLe(vararg values: Float): ByteArray {
    val out = ByteArray(values.size * 4)
    for (i in values.indices) {
        val b = floatToLe(values[i])
        b.copyInto(out, i * 4)
    }
    return out
}

private fun bytesFromCsv(csv: String): ByteArray =
    csv.split(',').map { it.trim().toInt().toByte() }.toByteArray()

internal fun arbitraryDtype(rng: Random = Random): Dtype {
    val choices =
        listOf(
            Dtype.BOOL,
            Dtype.F4,
            Dtype.F6_E3M2,
            Dtype.F6_E2M3,
            Dtype.F8_E8M0,
            Dtype.F8_E5M2,
            Dtype.F8_E4M3,
            Dtype.I8,
            Dtype.U8,
            Dtype.I16,
            Dtype.U16,
            Dtype.I32,
            Dtype.U32,
            Dtype.I64,
            Dtype.U64,
            Dtype.F16,
            Dtype.BF16,
            Dtype.F32,
            Dtype.F64,
            Dtype.C64,
        )
    return choices[rng.nextInt(choices.size)]
}

internal fun arbitraryShape(rng: Random = Random): List<Int> {
    val length = rng.nextInt(1, MAX_DIMENSION)
    return List(length) { rng.nextInt(1, MAX_SIZE) }
}

internal fun arbitraryMetadata(rng: Random = Random): Metadata {
    while (true) {
        val size = rng.nextInt(1, MAX_TENSORS)
        val dtypes = List(size) { arbitraryDtype(rng) }
        val shapes = List(size) { arbitraryShape(rng) }
        var start = 0
        val namedTensors = ArrayList<NamedTensorInfo>()
        for (i in 0 until size) {
            val dtype = dtypes[i]
            val shape = shapes[i]
            var prod = 1
            for (d in shape) prod *= d
            val bitlength = prod * dtype.bitsize()
            if (bitlength % 8 != 0) {
                continue
            }
            val length = (bitlength + 7) / 8
            val end = start + length
            val info =
                TensorInfo(
                    dtype = dtype,
                    shape = shape,
                    dataOffsets = ByteSpan(start, end),
                )
            namedTensors.add(NamedTensorInfo("t.${namedTensors.size}", info))
            start = end
        }
        if (namedTensors.isNotEmpty()) {
            return Metadata.new(null, namedTensors)
        }
    }
}

internal fun dataSize(metadata: Metadata): Int = metadata.dataLen()

internal fun asciiWithEscapes(
    prefix: ByteArray,
    json: String,
    suffix: ByteArray,
): ByteArray {
    val jsonBytes = json.encodeToByteArray()
    val out = ByteArray(prefix.size + jsonBytes.size + suffix.size)
    prefix.copyInto(out, 0)
    jsonBytes.copyInto(out, prefix.size)
    suffix.copyInto(out, prefix.size + jsonBytes.size)
    return out
}

internal fun gpt2Like(
    nHeads: Int,
    modelId: String,
) {
    val tensorsDesc = ArrayList<Pair<String, List<Int>>>()
    tensorsDesc.add("wte" to listOf(50, 76))
    tensorsDesc.add("wpe" to listOf(10, 76))
    for (i in 0 until nHeads) {
        tensorsDesc.add("h.$i.ln_1.weight" to listOf(76))
        tensorsDesc.add("h.$i.ln_1.bias" to listOf(76))
        tensorsDesc.add("h.$i.attn.bias" to listOf(1, 1, 10, 10))
        tensorsDesc.add("h.$i.attn.c_attn.weight" to listOf(76, 230))
        tensorsDesc.add("h.$i.attn.c_attn.bias" to listOf(230))
        tensorsDesc.add("h.$i.attn.c_proj.weight" to listOf(76, 76))
        tensorsDesc.add("h.$i.attn.c_proj.bias" to listOf(76))
        tensorsDesc.add("h.$i.ln_2.weight" to listOf(76))
        tensorsDesc.add("h.$i.ln_2.bias" to listOf(76))
        tensorsDesc.add("h.$i.mlp.c_fc.weight" to listOf(76, 307))
        tensorsDesc.add("h.$i.mlp.c_fc.bias" to listOf(307))
        tensorsDesc.add("h.$i.mlp.c_proj.weight" to listOf(307, 76))
        tensorsDesc.add("h.$i.mlp.c_proj.bias" to listOf(76))
    }
    tensorsDesc.add("ln_f.weight" to listOf(76))
    tensorsDesc.add("ln_f.bias" to listOf(76))

    val dtype = Dtype.F32
    var totalElements = 0
    for ((_, shape) in tensorsDesc) {
        var prod = 1
        for (d in shape) prod *= d
        totalElements += prod
    }
    val nbits = totalElements * dtype.bitsize()
    if (nbits % 8 != 0) {
        error("Misaligned slice")
    }
    val n = nbits / 8
    val allData = ByteArray(n)
    val metadata = LinkedHashMap<String, TensorView>(tensorsDesc.size)
    var offset = 0
    for ((name, shape) in tensorsDesc) {
        var prod = 1
        for (d in shape) prod *= d
        val byteLen = (prod * dtype.bitsize()) / 8
        val buffer = allData.copyOfRange(offset, offset + byteLen)
        val tensor = TensorView.new(dtype, shape, buffer)
        metadata[name] = tensor
        offset += prod
    }

    val filename = "./out_$modelId.safetensors"
    val out = serialize(metadata, null)
    val deserialized1 = SafeTensors.deserialize(out)
    assertEquals(tensorsDesc.size, deserialized1.len())

    serializeToFile(metadata, null, filename)
    val raw = readSerializedFile(filename)
    val deserialized2 = SafeTensors.deserialize(raw)
    assertEquals(tensorsDesc.size, deserialized2.len())
    removeSerializedFile(filename)
}

class TensorTest {
    @Test
    fun testIndexing() {
        val rng = Random(42)
        repeat(20) {
            val metadata = arbitraryMetadata(rng)
            val data = ByteArray(dataSize(metadata))
            val tensors = SafeTensors(metadata, data)
            for (name in tensors.names()) {
                val view = tensors.tensor(name)
                assertTrue(view.data().size >= 0)
            }
        }
    }

    @Test
    fun testRoundtrip() {
        val rng = Random(123)
        repeat(20) {
            val metadata = arbitraryMetadata(rng)
            val data = ByteArray(dataSize(metadata)) { idx -> idx.toByte() }
            val before = SafeTensors(metadata, data)
            val tensors = before.tensors()
            val bytes = serializeNamed(tensors, null)

            val after = SafeTensors.deserialize(bytes)

            assertEquals(before.names().size, after.names().size)
            for (name in before.names()) {
                val tensorBefore = before.tensor(name)
                val tensorAfter = after.tensor(name)
                assertEquals(tensorBefore, tensorAfter)
            }
        }
    }

    @Test
    fun testSerialization() {
        val data = floatsToLe(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f)
        val shape = listOf(1, 2, 3)
        val attn0 = TensorView.new(Dtype.F32, shape, data)
        val metadata = mapOf("attn.0" to attn0)

        val out = serialize(metadata, null)
        val expected =
            bytesFromCsv(
                "64, 0, 0, 0, 0, 0, 0, 0, 123, 34, 97, 116, 116, 110, 46, 48, 34, 58, 123, 34, 100, " +
                    "116, 121, 112, 101, 34, 58, 34, 70, 51, 50, 34, 44, 34, 115, 104, 97, 112, 101, 34, " +
                    "58, 91, 49, 44, 50, 44, 51, 93, 44, 34, 100, 97, 116, 97, 95, 111, 102, 102, 115, " +
                    "101, 116, 115, 34, 58, 91, 48, 44, 50, 52, 93, 125, 125, 0, 0, 0, 0, 0, 0, 128, 63, " +
                    "0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64, 0, 0, 160, 64",
            )
        assertContentEquals(expected, out)
        val parsed = SafeTensors.deserialize(out)
        assertEquals(1, parsed.len())
    }

    @Test
    fun testSerializationFp4() {
        val data = byteArrayOf(0)
        val shape = listOf(1, 2)
        val attn0 = TensorView.new(Dtype.F4, shape, data)
        val metadata = mapOf("attn.0" to attn0)

        val out = serialize(metadata, null)
        val expected =
            bytesFromCsv(
                "64, 0, 0, 0, 0, 0, 0, 0, 123, 34, 97, 116, 116, 110, 46, 48, 34, 58, 123, 34, 100, " +
                    "116, 121, 112, 101, 34, 58, 34, 70, 52, 34, 44, 34, 115, 104, 97, 112, 101, 34, 58, " +
                    "91, 49, 44, 50, 93, 44, 34, 100, 97, 116, 97, 95, 111, 102, 102, 115, 101, 116, " +
                    "115, 34, 58, 91, 48, 44, 49, 93, 125, 125, 32, 32, 32, 32, 0",
            )
        assertContentEquals(expected, out)
        val parsed = SafeTensors.deserialize(out)
        val tensors = parsed.tensors().associate { it.name to it.view }
        assertEquals(metadata, tensors)
    }

    @Test
    fun testSerializationFp4Misaligned() {
        val data = byteArrayOf(0, 1)
        val shape = listOf(1, 3)
        assertFailsWith<SafeTensorError.MisalignedSlice> {
            TensorView.new(Dtype.F4, shape, data)
        }
    }

    @Test
    fun testSerializationFp4Invalid() {
        val data = byteArrayOf(0, 1)
        val shape = listOf(1, 2)
        val err =
            assertFailsWith<SafeTensorError.InvalidTensorView> {
                TensorView.new(Dtype.F4, shape, data)
            }
        assertEquals(Dtype.F4, err.dtype)
    }

    @Test
    fun testEmpty() {
        val tensors = emptyMap<String, TensorView>()

        val out1 = serialize(tensors, null)
        val expected1 =
            bytesFromCsv(
                "8, 0, 0, 0, 0, 0, 0, 0, 123, 125, 32, 32, 32, 32, 32, 32",
            )
        assertContentEquals(expected1, out1)
        val parsed1 = SafeTensors.deserialize(out1)
        assertTrue(parsed1.isEmpty())

        val metadata = mapOf("framework" to "pt")
        val out2 = serialize(tensors, metadata)
        val expected2 =
            bytesFromCsv(
                "40, 0, 0, 0, 0, 0, 0, 0, 123, 34, 95, 95, 109, 101, 116, 97, 100, 97, 116, 97, 95, " +
                    "95, 34, 58, 123, 34, 102, 114, 97, 109, 101, 119, 111, 114, 107, 34, 58, 34, 112, " +
                    "116, 34, 125, 125, 32, 32, 32, 32, 32",
            )
        assertContentEquals(expected2, out2)
        val parsed2 = SafeTensors.deserialize(out2)
        assertTrue(parsed2.isEmpty())
    }

    @Test
    fun testSerializationForcedAlignement() {
        val data = floatsToLe(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f)
        val shape = listOf(1, 1, 2, 3)
        val attn0 = TensorView.new(Dtype.F32, shape, data)
        val metadata = mapOf("attn0" to attn0)

        val out = serialize(metadata, null)
        val expected =
            bytesFromCsv(
                "72, 0, 0, 0, 0, 0, 0, 0, 123, 34, 97, 116, 116, 110, 48, 34, 58, 123, 34, 100, 116, " +
                    "121, 112, 101, 34, 58, 34, 70, 51, 50, 34, 44, 34, 115, 104, 97, 112, 101, 34, 58, " +
                    "91, 49, 44, 49, 44, 50, 44, 51, 93, 44, 34, 100, 97, 116, 97, 95, 111, 102, 102, " +
                    "115, 101, 116, 115, 34, 58, 91, 48, 44, 50, 52, 93, 125, 125, 32, 32, 32, 32, 32, " +
                    "32, 32, 0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64, 0, 0, " +
                    "160, 64",
            )
        assertContentEquals(expected, out)
        val parsed = SafeTensors.deserialize(out)
        val tensor = parsed.tensor("attn0")
        assertContentEquals(data, tensor.data())
    }

    @Test
    fun testSlicing() {
        val data = floatsToLe(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f)
        val attn0 = TensorView.new(Dtype.F32, listOf(1, 2, 3), data)
        val metadata = mapOf("attn.0" to attn0)

        val out = serialize(metadata, null)
        val parsed = SafeTensors.deserialize(out)

        val sliceIter1 =
            parsed.tensor("attn.0").slice(
                TensorIndexer.full(),
                TensorIndexer.toExclusive(1),
            )
        val outBuffer1 = ArrayList<Byte>()
        for (chunk in sliceIter1) {
            for (b in chunk) outBuffer1.add(b)
        }
        val outBytes1 = outBuffer1.toByteArray()
        assertContentEquals(
            byteArrayOf(0, 0, 0, 0, 0, 0, 128.toByte(), 63, 0, 0, 0, 64),
            outBytes1,
        )
        assertContentEquals(floatsToLe(0.0f, 1.0f, 2.0f), outBytes1)

        val sliceIter2 =
            parsed.tensor("attn.0").slice(
                TensorIndexer.full(),
                TensorIndexer.full(),
                TensorIndexer.toExclusive(1),
            )
        val outBuffer2 = ArrayList<Byte>()
        for (chunk in sliceIter2) {
            for (b in chunk) outBuffer2.add(b)
        }
        val outBytes2 = outBuffer2.toByteArray()
        assertContentEquals(
            byteArrayOf(0, 0, 0, 0, 0, 0, 64, 64),
            outBytes2,
        )
        assertContentEquals(floatsToLe(0.0f, 3.0f), outBytes2)
    }

    @Test
    fun testGpt2() {
        gpt2Like(12, "gpt2")
    }

    @Test
    fun testGpt2Tiny() {
        gpt2Like(6, "gpt2_tiny")
    }

    @Test
    fun testSerializeToFileSamePathWithActiveMmap() {
        val filename = "safetensors_test_762.safetensors"
        val bytes = floatsToLe(1.0f, 2.0f, 3.0f)
        val view = TensorView.new(Dtype.F32, listOf(3), bytes)
        val metadata = mapOf("w" to view)
        serializeToFile(metadata, null, filename)

        val mmapBytes = readSerializedFile(filename)
        val safetensors = SafeTensors.deserialize(mmapBytes)
        val mmapTensors = safetensors.tensors()
        serializeNamedToFile(mmapTensors, null, filename)

        val raw = readSerializedFile(filename)
        val reloaded = SafeTensors.deserialize(raw)
        assertContentEquals(bytes, reloaded.tensor("w").data())
        removeSerializedFile(filename)
    }

    @Test
    fun testEmptyShapesAllowed() {
        val serialized =
            asciiWithEscapes(
                byteArrayOf(56, 0, 0, 0, 0, 0, 0, 0),
                "{\"test\":{\"dtype\":\"I32\",\"shape\":[],\"data_offsets\":[0,4]}}",
                byteArrayOf(0, 0, 0, 0),
            )

        val loaded = SafeTensors.deserialize(serialized)
        assertEquals(listOf("test"), loaded.names())
        val tensor = loaded.tensor("test")
        assertTrue(tensor.shape().isEmpty())
        assertEquals(Dtype.I32, tensor.dtype())
        assertContentEquals(byteArrayOf(0, 0, 0, 0), tensor.data())
    }

    @Test
    fun testDeserialization() {
        val serialized =
            asciiWithEscapes(
                byteArrayOf(60, 0, 0, 0, 0, 0, 0, 0),
                "{\"test\":{\"dtype\":\"I32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}",
                ByteArray(16),
            )

        val loaded = SafeTensors.deserialize(serialized)
        assertEquals(1, loaded.len())
        assertEquals(listOf("test"), loaded.names())
        val tensor = loaded.tensor("test")
        assertEquals(listOf(2, 2), tensor.shape())
        assertEquals(Dtype.I32, tensor.dtype())
        assertContentEquals(ByteArray(16), tensor.data())
    }

    @Test
    fun testLifetimes() {
        val serialized =
            asciiWithEscapes(
                byteArrayOf(60, 0, 0, 0, 0, 0, 0, 0),
                "{\"test\":{\"dtype\":\"I32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}",
                ByteArray(16),
            )

        val tensor =
            run {
                val loaded = SafeTensors.deserialize(serialized)
                loaded.tensor("test")
            }

        assertEquals(listOf(2, 2), tensor.shape())
        assertEquals(Dtype.I32, tensor.dtype())
        assertContentEquals(ByteArray(16), tensor.data())
    }

    @Test
    fun testJsonAttack() {
        val sb = StringBuilder()
        sb.append('{')
        for (i in 0 until 10) {
            if (i > 0) sb.append(',')
            sb.append("\"weight_$i\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]}")
        }
        sb.append('}')
        val jsonStr = sb.toString()
        val jsonBytes = jsonStr.encodeToByteArray()
        val n = jsonBytes.size
        val nBytes =
            byteArrayOf(
                (n and 0xFF).toByte(),
                ((n ushr 8) and 0xFF).toByte(),
                ((n ushr 16) and 0xFF).toByte(),
                ((n ushr 24) and 0xFF).toByte(),
                0,
                0,
                0,
                0,
            )
        val reloaded = asciiWithEscapes(nBytes, jsonStr, ByteArray(16))
        assertFailsWith<SafeTensorError.InvalidOffset> {
            SafeTensors.deserialize(reloaded)
        }
    }

    @Test
    fun testMetadataIncompleteBuffer() {
        val serializedExtra =
            asciiWithEscapes(
                byteArrayOf(60, 0, 0, 0, 0, 0, 0, 0),
                "{\"test\":{\"dtype\":\"I32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}",
                ByteArray(16) + "extra_bogus_data_for_polyglot_file".encodeToByteArray(),
            )
        assertFailsWith<SafeTensorError.MetadataIncompleteBuffer> {
            SafeTensors.deserialize(serializedExtra)
        }

        val serializedMissing =
            asciiWithEscapes(
                byteArrayOf(60, 0, 0, 0, 0, 0, 0, 0),
                "{\"test\":{\"dtype\":\"I32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}",
                ByteArray(14),
            )
        assertFailsWith<SafeTensorError.MetadataIncompleteBuffer> {
            SafeTensors.deserialize(serializedMissing)
        }
    }

    @Test
    fun testHeaderTooLarge() {
        val serialized =
            asciiWithEscapes(
                byteArrayOf(60, 0, 0, 0, 0, 0xFF.toByte(), 0xFF.toByte(), 0xFF.toByte()),
                "{\"test\":{\"dtype\":\"I32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}",
                ByteArray(16),
            )
        assertFailsWith<SafeTensorError.HeaderTooLarge> {
            SafeTensors.deserialize(serialized)
        }
    }

    @Test
    fun testHeaderTooSmall() {
        val serialized = ByteArray(0)
        assertFailsWith<SafeTensorError.HeaderTooSmall> {
            SafeTensors.deserialize(serialized)
        }
    }

    @Test
    fun testInvalidHeaderLength() {
        val serialized = byteArrayOf(60, 0, 0, 0, 0, 0, 0, 0)
        assertFailsWith<SafeTensorError.InvalidHeaderLength> {
            SafeTensors.deserialize(serialized)
        }
    }

    @Test
    fun testInvalidHeaderNonUtf8() {
        val serialized = byteArrayOf(1, 0, 0, 0, 0, 0, 0, 0, 0xFF.toByte())
        assertFailsWith<SafeTensorError.InvalidHeader> {
            SafeTensors.deserialize(serialized)
        }
    }

    @Test
    fun testInvalidHeaderNotJson() {
        val serialized = byteArrayOf(1, 0, 0, 0, 0, 0, 0, 0, '{'.code.toByte())
        assertFailsWith<SafeTensorError.InvalidHeaderDeserialization> {
            SafeTensors.deserialize(serialized)
        }
    }

    @Test
    fun testWhitespacePaddedHeader() {
        val serialized =
            byteArrayOf(6, 0, 0, 0, 0, 0, 0, 0) +
                byteArrayOf('{'.code.toByte(), '}'.code.toByte(), 0x0D, 0x20, 0x09, 0x0A)
        val loaded = SafeTensors.deserialize(serialized)
        assertEquals(0, loaded.len())
    }

    @Test
    fun testWhitespaceLeadingPaddedHeader() {
        val serialized =
            byteArrayOf(6, 0, 0, 0, 0, 0, 0, 0) +
                byteArrayOf(0x09, 0x0A, '{'.code.toByte(), '}'.code.toByte(), 0x0D, 0x20)
        val loaded = SafeTensors.deserialize(serialized)
        assertEquals(0, loaded.len())
    }

    @Test
    fun testZeroSizedTensor() {
        val serialized =
            asciiWithEscapes(
                byteArrayOf(60, 0, 0, 0, 0, 0, 0, 0),
                "{\"test\":{\"dtype\":\"I32\",\"shape\":[2,0],\"data_offsets\":[0, 0]}}",
                ByteArray(0),
            )
        val loaded = SafeTensors.deserialize(serialized)
        assertEquals(listOf("test"), loaded.names())
        val tensor = loaded.tensor("test")
        assertEquals(listOf(2, 0), tensor.shape())
        assertEquals(Dtype.I32, tensor.dtype())
        assertContentEquals(ByteArray(0), tensor.data())
    }

    @Test
    fun testInvalidInfo() {
        val serialized =
            asciiWithEscapes(
                byteArrayOf(60, 0, 0, 0, 0, 0, 0, 0),
                "{\"test\":{\"dtype\":\"I32\",\"shape\":[2,2],\"data_offsets\":[0, 4]}}",
                ByteArray(0),
            )
        assertFailsWith<SafeTensorError.TensorInvalidInfo> {
            SafeTensors.deserialize(serialized)
        }
    }

    @Test
    fun testValidationOverflow() {
        val serialized1 =
            asciiWithEscapes(
                byteArrayOf('O'.code.toByte(), 0, 0, 0, 0, 0, 0, 0),
                "{\"test\":{\"dtype\":\"I32\",\"shape\":[2,18446744073709551614],\"data_offsets\":[0,16]}}",
                ByteArray(16),
            )
        assertFailsWith<SafeTensorError.ValidationOverflow> {
            SafeTensors.deserialize(serialized1)
        }

        val serialized2 =
            asciiWithEscapes(
                byteArrayOf('N'.code.toByte(), 0, 0, 0, 0, 0, 0, 0),
                "{\"test\":{\"dtype\":\"I32\",\"shape\":[2,9223372036854775807],\"data_offsets\":[0,16]}}",
                ByteArray(16),
            )
        assertFailsWith<SafeTensorError.ValidationOverflow> {
            SafeTensors.deserialize(serialized2)
        }
    }

    @Test
    fun testInvalidHeaderSizeSerialization() {
        val dataInfo = HashMap<String, String>()
        val tensors = emptyMap<String, TensorView>()
        val veryLargeMetadata = "a".repeat(MAX_HEADER_SIZE / 100)
        for (i in 0 until 100) {
            dataInfo["very_large_metadata_$i"] = veryLargeMetadata
        }
        assertFailsWith<SafeTensorError.HeaderTooLarge> {
            serialize(tensors, dataInfo)
        }
    }
}
