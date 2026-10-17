// port-lint: source tensor.rs
package io.github.kotlinmania.safetensors

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive

internal const val MAX_HEADER_SIZE: Int = 100_000_000
internal const val N_LEN: Int = 8

/**
 * Possible errors that could occur while reading or writing a Safetensor file.
 */
public sealed class SafeTensorError(
    message: String,
    cause: Throwable? = null,
) : Exception(message, cause) {
    /** The header is an invalid UTF-8 string and cannot be read. */
    public class InvalidHeader(
        public val error: String,
    ) : SafeTensorError("invalid UTF-8 in header: $error") {
        override fun equals(other: Any?): Boolean = other is InvalidHeader && error == other.error

        override fun hashCode(): Int = error.hashCode()
    }

    /** The header does contain a valid string, but it is not valid JSON. */
    public class InvalidHeaderDeserialization(
        public val error: String,
    ) : SafeTensorError("invalid JSON in header: $error") {
        override fun equals(other: Any?): Boolean = other is InvalidHeaderDeserialization && error == other.error

        override fun hashCode(): Int = error.hashCode()
    }

    /** The header is larger than 100MB which is considered too large. */
    public class HeaderTooLarge : SafeTensorError("header too large") {
        override fun equals(other: Any?): Boolean = other is HeaderTooLarge

        override fun hashCode(): Int = 1
    }

    /** The header is smaller than 8 bytes. */
    public class HeaderTooSmall : SafeTensorError("header too small") {
        override fun equals(other: Any?): Boolean = other is HeaderTooSmall

        override fun hashCode(): Int = 2
    }

    /** The header length is invalid. */
    public class InvalidHeaderLength : SafeTensorError("invalid header length") {
        override fun equals(other: Any?): Boolean = other is InvalidHeaderLength

        override fun hashCode(): Int = 3
    }

    /** The tensor name was not found in the archive. */
    public data class TensorNotFound(
        public val name: String,
    ) : SafeTensorError("tensor `$name` not found")

    /** Invalid information between shape, dtype and the proposed offsets in the file. */
    public class TensorInvalidInfo : SafeTensorError("invalid shape, data type, or offset for tensor") {
        override fun equals(other: Any?): Boolean = other is TensorInvalidInfo

        override fun hashCode(): Int = 4
    }

    /** The offsets declared for tensor with name [name] in the header are invalid. */
    public data class InvalidOffset(
        public val name: String,
    ) : SafeTensorError("invalid offset for tensor `$name`")

    /** I/O error. */
    public data class IoError(
        public val error: String,
    ) : SafeTensorError("I/O error: $error")

    /** JSON error. */
    public data class JsonError(
        public val error: String,
    ) : SafeTensorError("JSON error: $error")

    /** The tensor cannot be created because the buffer size doesn't match shape + dtype. */
    public data class InvalidTensorView(
        public val dtype: Dtype,
        public val shape: List<Int>,
        public val nBytes: Int,
    ) : SafeTensorError(
            "tensor of type $dtype and shape (${shape.joinToString(", ")}) can't be created from $nBytes bytes",
        )

    /**
     * The metadata is invalid because the data offsets of the tensor do not
     * fully cover the buffer part of the file. The last offset must be
     * the end of the file.
     */
    public class MetadataIncompleteBuffer : SafeTensorError("incomplete metadata, file not fully covered") {
        override fun equals(other: Any?): Boolean = other is MetadataIncompleteBuffer

        override fun hashCode(): Int = 5
    }

    /**
     * The metadata contains information (shape or shape * dtype size) which lead to an
     * arithmetic overflow.
     */
    public class ValidationOverflow : SafeTensorError("overflow computing buffer size from shape and/or element type") {
        override fun equals(other: Any?): Boolean = other is ValidationOverflow

        override fun hashCode(): Int = 6
    }

    /**
     * For smaller than 1 byte dtypes, some slices will happen outside of the byte boundary.
     */
    public class MisalignedSlice :
        SafeTensorError(
            "The slice is slicing for subbytes dtypes, and the slice does not end up at a byte boundary, this is invalid.",
        ) {
        override fun equals(other: Any?): Boolean = other is MisalignedSlice

        override fun hashCode(): Int = 7
    }

    override fun toString(): String = message.orEmpty()
}

private class PreparedData(
    val n: ULong,
    val headerBytes: ByteArray,
    val offset: Int,
)

/**
 * The interface necessary to enable safetensors to serialize a tensor.
 */
public interface View {
    /** The [Dtype] of the tensor. */
    public fun dtype(): Dtype

    /** The shape of the tensor. */
    public fun shape(): List<Int>

    /** The data of the tensor. */
    public fun data(): ByteArray

    /**
     * The length of the data, in bytes.
     * This is necessary as this might be faster to get than `data().size`,
     * for instance for tensors residing in GPU.
     */
    public fun dataLen(): Int = data().size
}

/**
 * Named tensor entry pairing a tensor [name] with its [info].
 */
public data class NamedTensorInfo(
    public val name: String,
    public val info: TensorInfo,
)

/**
 * Named tensor view pairing a tensor [name] with its [view].
 */
public data class NamedTensorView(
    public val name: String,
    public val view: TensorView,
)

/**
 * Header size and parsed [Metadata] returned by [SafeTensors.readMetadata].
 */
public data class ReadMetadataOutput(
    public val headerSize: Int,
    public val metadata: Metadata,
)

private fun escapeJsonString(value: String): String {
    val sb = StringBuilder(value.length + 2)
    sb.append('"')
    for (ch in value) {
        when (ch) {
            '"' -> sb.append("\\\"")
            '\\' -> sb.append("\\\\")
            '\b' -> sb.append("\\b")
            '\u000C' -> sb.append("\\f")
            '\n' -> sb.append("\\n")
            '\r' -> sb.append("\\r")
            '\t' -> sb.append("\\t")
            else -> if (ch.code < 0x20) sb.append("\\u").append(ch.code.toString(16).padStart(4, '0')) else sb.append(ch)
        }
    }
    sb.append('"')
    return sb.toString()
}

private fun serializeMetadataToJsonBytes(metadata: Metadata): ByteArray {
    val sb = StringBuilder()
    sb.append('{')
    var firstEntry = true
    val metaMap = metadata.metadata()
    if (metaMap != null) {
        sb.append("\"__metadata__\":{")
        var firstMeta = true
        for ((k, v) in metaMap) {
            if (!firstMeta) sb.append(',')
            firstMeta = false
            sb.append(escapeJsonString(k)).append(':').append(escapeJsonString(v))
        }
        sb.append('}')
        firstEntry = false
    }

    val names = Array(metadata.indexMap.size) { "" }
    for ((name, index) in metadata.indexMap) {
        names[index] = name
    }

    for (i in names.indices) {
        if (!firstEntry) sb.append(',')
        firstEntry = false
        val name = names[i]
        val info = metadata.tensorsList[i]
        sb.append(escapeJsonString(name)).append(":{")
        sb.append("\"dtype\":").append(escapeJsonString(info.dtype.toString())).append(',')
        sb.append("\"shape\":[")
        for (j in info.shape.indices) {
            if (j > 0) sb.append(',')
            sb.append(info.shape[j])
        }
        sb.append("],\"data_offsets\":[")
        sb.append(info.dataOffsets.start).append(',').append(info.dataOffsets.stop)
        sb.append("]}")
    }
    sb.append('}')
    return sb.toString().encodeToByteArray()
}

private fun nextMultipleOf(value: Int, multiple: Int): Int {
    val rem = value % multiple
    return if (rem == 0) value else value + (multiple - rem)
}

private fun prepareEntries(
    entries: List<Pair<String, View>>,
    dataInfo: Map<String, String>?,
): Pair<PreparedData, List<View>> {
    val sorted =
        entries.sortedWith { a, b ->
            val dtypeCmp = b.second.dtype().compareTo(a.second.dtype())
            if (dtypeCmp != 0) dtypeCmp else a.first.compareTo(b.first)
        }

    val tensors = ArrayList<View>(sorted.size)
    val hmetadata = ArrayList<NamedTensorInfo>(sorted.size)
    var offset = 0

    for ((name, tensor) in sorted) {
        val n = tensor.dataLen()
        val tensorInfo =
            TensorInfo(
                dtype = tensor.dtype(),
                shape = tensor.shape().toList(),
                dataOffsets = ByteSpan(offset, offset + n),
            )
        offset += n
        hmetadata.add(NamedTensorInfo(name, tensorInfo))
        tensors.add(tensor)
    }

    val metadata = Metadata.new(dataInfo, hmetadata)
    val rawMetadataBytes = serializeMetadataToJsonBytes(metadata)
    val alignedMetadataLen = nextMultipleOf(rawMetadataBytes.size, N_LEN)
    val metadataBuf =
        ByteArray(alignedMetadataLen) { idx ->
            if (idx < rawMetadataBytes.size) rawMetadataBytes[idx] else ' '.code.toByte()
        }

    return Pair(
        PreparedData(
            n = alignedMetadataLen.toULong(),
            headerBytes = metadataBuf,
            offset = offset,
        ),
        tensors,
    )
}

/**
 * Serialize a map of tensors to an owned byte buffer.
 */
public fun serialize(
    data: Map<String, View>,
    dataInfo: Map<String, String>? = null,
): ByteArray = serializeEntries(data.entries.map { it.key to it.value }, dataInfo)

/**
 * Serialize a list of named tensor views to an owned byte buffer.
 */
public fun serializeNamed(
    data: List<NamedTensorView>,
    dataInfo: Map<String, String>? = null,
): ByteArray = serializeEntries(data.map { it.name to it.view }, dataInfo)

internal fun serializeEntries(
    entries: List<Pair<String, View>>,
    dataInfo: Map<String, String>? = null,
): ByteArray {
    if (dataInfo != null) {
        var minHeaderBytes = 2L
        for ((k, v) in dataInfo) {
            minHeaderBytes += k.length.toLong() + v.length.toLong() + 6L
        }
        if (minHeaderBytes > MAX_HEADER_SIZE.toLong()) {
            throw SafeTensorError.HeaderTooLarge()
        }
    }
    val (prepared, tensors) = prepareEntries(entries, dataInfo)
    if (prepared.n > MAX_HEADER_SIZE.toULong()) {
        throw SafeTensorError.HeaderTooLarge()
    }
    val headerBytes = prepared.headerBytes
    val expectedSize = N_LEN + headerBytes.size + prepared.offset
    val buffer = ByteArray(expectedSize)
    val n = prepared.n
    for (i in 0 until N_LEN) {
        buffer[i] = ((n shr (i * 8)) and 0xFFuL).toByte()
    }
    headerBytes.copyInto(buffer, destinationOffset = N_LEN)
    var writePos = N_LEN + headerBytes.size
    for (tensor in tensors) {
        val bytes = tensor.data()
        bytes.copyInto(buffer, destinationOffset = writePos)
        writePos += bytes.size
    }
    return buffer
}

private val virtualFileStore: MutableMap<String, ByteArray> = mutableMapOf()

/**
 * Serialize a map of tensors to a file at [filename].
 */
public fun serializeToFile(
    data: Map<String, View>,
    dataInfo: Map<String, String>?,
    filename: String,
) {
    val bytes = serialize(data, dataInfo)
    virtualFileStore[filename] = bytes
}

/**
 * Serialize a list of named tensor views to a file at [filename].
 */
public fun serializeNamedToFile(
    data: List<NamedTensorView>,
    dataInfo: Map<String, String>?,
    filename: String,
) {
    val bytes = serializeNamed(data, dataInfo)
    virtualFileStore[filename] = bytes
}

/**
 * Read a serialized safetensors buffer previously written via [serializeToFile].
 */
public fun readSerializedFile(filename: String): ByteArray =
    virtualFileStore[filename]?.copyOf()
        ?: throw SafeTensorError.IoError("file not found: $filename")

/**
 * Remove a serialized safetensors file previously written via [serializeToFile].
 */
public fun removeSerializedFile(filename: String) {
    virtualFileStore.remove(filename)
}

/**
 * A structure owning metadata to look up tensors on a shared [data] byte buffer.
 */
public class SafeTensors internal constructor(
    public val metadata: Metadata,
    private val data: ByteArray,
) {
    public companion object {
        /**
         * Given a byte-buffer representing the whole safetensor file,
         * parses the header, and returns the size of the header + the parsed [Metadata].
         */
        public fun readMetadata(buffer: ByteArray): ReadMetadataOutput {
            val bufferLen = buffer.size
            if (bufferLen < N_LEN) {
                throw SafeTensorError.HeaderTooSmall()
            }
            var nULong = 0uL
            for (i in 0 until N_LEN) {
                nULong = nULong or ((buffer[i].toUByte().toULong()) shl (i * 8))
            }
            if (nULong > MAX_HEADER_SIZE.toULong()) {
                throw SafeTensorError.HeaderTooLarge()
            }
            val n = nULong.toInt()
            val stopLong = n.toLong() + N_LEN.toLong()
            if (stopLong > bufferLen.toLong() || stopLong > Int.MAX_VALUE.toLong()) {
                throw SafeTensorError.InvalidHeaderLength()
            }
            val stop = stopLong.toInt()
            val headerBytes = buffer.copyOfRange(N_LEN, stop)
            val string =
                try {
                    headerBytes.decodeToString(throwOnInvalidSequence = true)
                } catch (e: CharacterCodingException) {
                    throw SafeTensorError.InvalidHeader(e.message ?: "invalid UTF-8")
                }
            val metadata = parseHashMetadata(string)
            val bufferEnd = metadata.validate()
            if (bufferEnd.toLong() + N_LEN.toLong() + n.toLong() != bufferLen.toLong()) {
                throw SafeTensorError.MetadataIncompleteBuffer()
            }
            return ReadMetadataOutput(n, metadata)
        }

        /**
         * Given a byte-buffer representing the whole safetensor file,
         * parses it and returns the deserialized [SafeTensors].
         */
        public fun deserialize(buffer: ByteArray): SafeTensors {
            val (n, metadata) = readMetadata(buffer)
            val data = buffer.copyOfRange(N_LEN + n, buffer.size)
            return SafeTensors(metadata, data)
        }
    }

    /**
     * Returns the tensors contained within the [SafeTensors].
     */
    public fun tensors(): List<NamedTensorView> {
        val result = ArrayList<NamedTensorView>(metadata.indexMap.size)
        for ((name, index) in metadata.indexMap) {
            val info = metadata.tensorsList[index]
            val tensorView =
                TensorView(
                    dtypeVal = info.dtype,
                    shapeVal = info.shape.toList(),
                    dataVal = data.copyOfRange(info.dataOffsets.start, info.dataOffsets.stop),
                )
            result.add(NamedTensorView(name, tensorView))
        }
        return result
    }

    /**
     * Returns an iterator over the tensors contained within the [SafeTensors].
     */
    public fun iter(): Iterator<NamedTensorView> = tensors().iterator()

    /**
     * Gets a specific tensor by [tensorName] within the [SafeTensors].
     */
    public fun tensor(tensorName: String): TensorView {
        val index =
            metadata.indexMap[tensorName]
                ?: throw SafeTensorError.TensorNotFound(tensorName)
        val info =
            metadata.tensorsList.getOrNull(index)
                ?: throw SafeTensorError.TensorNotFound(tensorName)
        return TensorView(
            dtypeVal = info.dtype,
            shapeVal = info.shape.toList(),
            dataVal = data.copyOfRange(info.dataOffsets.start, info.dataOffsets.stop),
        )
    }

    /**
     * Return the names of the tensors within the [SafeTensors].
     */
    public fun names(): List<String> = metadata.indexMap.keys.toList()

    /**
     * Return how many tensors are currently stored within the [SafeTensors].
     */
    public fun len(): Int = metadata.tensorsList.size

    /**
     * Indicate if the [SafeTensors] contains any tensor or not.
     */
    public fun isEmpty(): Boolean = metadata.tensorsList.isEmpty()
}

private fun parseHashMetadata(jsonString: String): Metadata {
    val trimmed = jsonString.trim { it == ' ' || it == '\t' || it == '\n' || it == '\r' }
    if (!trimmed.startsWith("{") || !trimmed.endsWith("}")) {
        throw SafeTensorError.InvalidHeaderDeserialization("expected JSON object")
    }
    val element: JsonElement =
        try {
            Json.parseToJsonElement(trimmed)
        } catch (e: Exception) {
            throw SafeTensorError.InvalidHeaderDeserialization(e.message ?: "invalid JSON")
        }
    val rootObj =
        element as? JsonObject
            ?: throw SafeTensorError.InvalidHeaderDeserialization("header root must be a JSON object")

    var metadataMap: Map<String, String>? = null
    val rawTensors = ArrayList<NamedTensorInfo>()

    for ((key, value) in rootObj.entries) {
        if (key == "__metadata__") {
            if (value is JsonPrimitive && value.isString) {
                throw SafeTensorError.InvalidHeaderDeserialization("__metadata__ must be an object")
            }
            val metaObj =
                value as? JsonObject
                    ?: throw SafeTensorError.InvalidHeaderDeserialization("__metadata__ must be an object")
            val map = LinkedHashMap<String, String>(metaObj.size)
            for ((mk, mv) in metaObj.entries) {
                val prim =
                    mv as? JsonPrimitive
                        ?: throw SafeTensorError.InvalidHeaderDeserialization("metadata value must be a string")
                if (!prim.isString) {
                    throw SafeTensorError.InvalidHeaderDeserialization("metadata value must be a string")
                }
                map[mk] = prim.content
            }
            metadataMap = map
        } else {
            val tensorObj =
                value as? JsonObject
                    ?: throw SafeTensorError.InvalidHeaderDeserialization("tensor entry '$key' must be an object")
            val dtypeElem =
                tensorObj["dtype"] as? JsonPrimitive
                    ?: throw SafeTensorError.InvalidHeaderDeserialization("missing dtype for '$key'")
            if (!dtypeElem.isString) {
                throw SafeTensorError.InvalidHeaderDeserialization("dtype must be a string")
            }
            val dtype =
                Dtype.fromWireName(dtypeElem.content)
                    ?: throw SafeTensorError.InvalidHeaderDeserialization("unknown dtype '${dtypeElem.content}'")

            val shapeArr =
                tensorObj["shape"] as? JsonArray
                    ?: throw SafeTensorError.InvalidHeaderDeserialization("missing shape for '$key'")
            val shapeULongs = ArrayList<ULong>(shapeArr.size)
            for (dimElem in shapeArr) {
                val prim =
                    dimElem as? JsonPrimitive
                        ?: throw SafeTensorError.InvalidHeaderDeserialization("shape element must be integer")
                if (prim.isString) {
                    throw SafeTensorError.InvalidHeaderDeserialization("shape element must be integer")
                }
                val dimULong =
                    prim.content.toULongOrNull()
                        ?: throw SafeTensorError.InvalidHeaderDeserialization("invalid shape dimension '${prim.content}'")
                shapeULongs.add(dimULong)
            }

            val offsetsArr =
                tensorObj["data_offsets"] as? JsonArray
                    ?: throw SafeTensorError.InvalidHeaderDeserialization("missing data_offsets for '$key'")
            if (offsetsArr.size != 2) {
                throw SafeTensorError.InvalidHeaderDeserialization("data_offsets must have 2 elements")
            }
            val startPrim =
                offsetsArr[0] as? JsonPrimitive
                    ?: throw SafeTensorError.InvalidHeaderDeserialization("data_offsets[0] must be integer")
            val stopPrim =
                offsetsArr[1] as? JsonPrimitive
                    ?: throw SafeTensorError.InvalidHeaderDeserialization("data_offsets[1] must be integer")
            if (startPrim.isString || stopPrim.isString) {
                throw SafeTensorError.InvalidHeaderDeserialization("data_offsets must be integers")
            }
            val startULong =
                startPrim.content.toULongOrNull()
                    ?: throw SafeTensorError.InvalidHeaderDeserialization("invalid start offset")
            val stopULong =
                stopPrim.content.toULongOrNull()
                    ?: throw SafeTensorError.InvalidHeaderDeserialization("invalid stop offset")

            val shapeInts = ArrayList<Int>(shapeULongs.size)
            for (d in shapeULongs) {
                shapeInts.add(if (d > Int.MAX_VALUE.toULong()) Int.MAX_VALUE else d.toInt())
            }
            val startInt = if (startULong > Int.MAX_VALUE.toULong()) Int.MAX_VALUE else startULong.toInt()
            val stopInt = if (stopULong > Int.MAX_VALUE.toULong()) Int.MAX_VALUE else stopULong.toInt()

            val info =
                TensorInfo(
                    dtype = dtype,
                    shape = shapeInts,
                    dataOffsets = ByteSpan(startInt, stopInt),
                    rawShapeULongs = shapeULongs,
                )
            rawTensors.add(NamedTensorInfo(key, info))
        }
    }

    rawTensors.sortWith { a, b ->
        val aOffsets = a.info.dataOffsets
        val bOffsets = b.info.dataOffsets
        val startCmp = aOffsets.start.compareTo(bOffsets.start)
        if (startCmp != 0) startCmp else aOffsets.stop.compareTo(bOffsets.stop)
    }
    return Metadata.new(metadataMap, rawTensors)
}

private fun checkedMulULong(a: ULong, b: ULong): ULong? {
    if (a == 0uL || b == 0uL) return 0uL
    val result = a * b
    return if (result / a == b) result else null
}

/**
 * The structure representing the header of safetensor files which allows
 * indexing into the raw byte-buffer array and how to interpret it.
 */
public class Metadata internal constructor(
    private val metadataMap: Map<String, String>?,
    internal val tensorsList: List<TensorInfo>,
    internal val indexMap: Map<String, Int>,
) {
    public companion object {
        /**
         * Creates a new [Metadata] structure.
         * May fail if there is incorrect data in the [TensorInfo].
         * Notably the tensors need to be ordered by increasing `dataOffsets`.
         */
        public fun new(
            metadata: Map<String, String>?,
            tensors: List<NamedTensorInfo>,
        ): Metadata {
            val indexMap = LinkedHashMap<String, Int>(tensors.size)
            val tensorInfos = ArrayList<TensorInfo>(tensors.size)
            for ((index, named) in tensors.withIndex()) {
                indexMap[named.name] = index
                tensorInfos.add(named.info)
            }
            val result = Metadata(metadata?.toMap(), tensorInfos, indexMap)
            result.validate()
            return result
        }
    }

    internal fun validate(): Int {
        var start = 0
        for ((i, info) in tensorsList.withIndex()) {
            val s = info.dataOffsets.start
            val e = info.dataOffsets.stop
            if (s != start || e < s) {
                val tensorName =
                    indexMap.entries.firstOrNull { it.value == i }?.key ?: "no_tensor"
                throw SafeTensorError.InvalidOffset(tensorName)
            }
            start = e

            var nelements = 1uL
            val dimsULong = info.rawShapeULongs
            if (dimsULong != null) {
                for (dim in dimsULong) {
                    nelements =
                        checkedMulULong(nelements, dim)
                            ?: throw SafeTensorError.ValidationOverflow()
                }
            } else {
                for (dim in info.shape) {
                    if (dim < 0) throw SafeTensorError.ValidationOverflow()
                    nelements =
                        checkedMulULong(nelements, dim.toULong())
                            ?: throw SafeTensorError.ValidationOverflow()
                }
            }

            val nbits =
                checkedMulULong(nelements, info.dtype.bitsize().toULong())
                    ?: throw SafeTensorError.ValidationOverflow()

            if (nbits % 8uL != 0uL) {
                throw SafeTensorError.MisalignedSlice()
            }
            val sizeULong = nbits / 8uL
            if (sizeULong > Int.MAX_VALUE.toULong()) {
                throw SafeTensorError.ValidationOverflow()
            }
            val size = sizeULong.toInt()
            if (e - s != size) {
                throw SafeTensorError.TensorInvalidInfo()
            }
        }
        return start
    }

    /** Gives back the tensor metadata for [name], or `null` if not present. */
    public fun info(name: String): TensorInfo? {
        val index = indexMap[name] ?: return null
        return tensorsList.getOrNull(index)
    }

    /** Gives back the tensor metadata map. */
    public fun tensors(): Map<String, TensorInfo> {
        val map = LinkedHashMap<String, TensorInfo>(indexMap.size)
        for ((tensorName, index) in indexMap) {
            map[tensorName] = tensorsList[index]
        }
        return map
    }

    /** Gives back the tensor names ordered by offset. */
    public fun offsetKeys(): List<String> =
        indexMap.entries
            .sortedBy { it.value }
            .map { it.key }

    /** Gives the size of the content buffer in bytes. */
    public fun dataLen(): Int = tensorsList.lastOrNull()?.dataOffsets?.stop ?: 0

    /** Gives back the file-level metadata map, if present. */
    public fun metadata(): Map<String, String>? = metadataMap
}

/**
 * A view of a Tensor within the file.
 * Contains a slice of data within the full byte-buffer
 * and is thus a readable view of a single tensor.
 */
public class TensorView internal constructor(
    internal val dtypeVal: Dtype,
    internal val shapeVal: List<Int>,
    internal val dataVal: ByteArray,
) : View {
    public companion object {
        /** Create a new [TensorView]. */
        public fun new(
            dtype: Dtype,
            shape: List<Int>,
            data: ByteArray,
        ): TensorView {
            var nElements = 1uL
            for (dim in shape) {
                if (dim < 0) throw SafeTensorError.ValidationOverflow()
                nElements =
                    checkedMulULong(nElements, dim.toULong())
                        ?: throw SafeTensorError.ValidationOverflow()
            }
            val nbits =
                checkedMulULong(nElements, dtype.bitsize().toULong())
                    ?: throw SafeTensorError.ValidationOverflow()
            if (nbits % 8uL != 0uL) {
                throw SafeTensorError.MisalignedSlice()
            }
            val sizeULong = nbits / 8uL
            if (sizeULong > Int.MAX_VALUE.toULong()) {
                throw SafeTensorError.ValidationOverflow()
            }
            val size = sizeULong.toInt()
            if (data.size != size) {
                throw SafeTensorError.InvalidTensorView(dtype, shape.toList(), data.size)
            }
            return TensorView(dtype, shape.toList(), data)
        }
    }

    /** The current tensor [Dtype]. */
    override fun dtype(): Dtype = dtypeVal

    /** The current tensor shape. */
    override fun shape(): List<Int> = shapeVal

    /** The current tensor byte-buffer. */
    override fun data(): ByteArray = dataVal

    override fun dataLen(): Int = dataVal.size

    /** The various pieces of the data buffer according to the asked slice. */
    public fun slicedData(slices: List<TensorIndexer>): SliceIterator = SliceIterator.new(this, slices)

    /** Slice this tensor along 1 dimension. */
    public fun slice(index: TensorIndexer): SliceIterator = slicedData(listOf(index))

    /** Slice this tensor along 2 dimensions. */
    public fun slice(
        index0: TensorIndexer,
        index1: TensorIndexer,
    ): SliceIterator = slicedData(listOf(index0, index1))

    /** Slice this tensor along 3 dimensions. */
    public fun slice(
        index0: TensorIndexer,
        index1: TensorIndexer,
        index2: TensorIndexer,
    ): SliceIterator = slicedData(listOf(index0, index1, index2))

    /** Slice this tensor using a list of [TensorIndexer]s. */
    public fun slice(slices: List<TensorIndexer>): SliceIterator = slicedData(slices)

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is TensorView) return false
        return dtypeVal == other.dtypeVal &&
            shapeVal == other.shapeVal &&
            dataVal.contentEquals(other.dataVal)
    }

    override fun hashCode(): Int {
        var result = dtypeVal.hashCode()
        result = 31 * result + shapeVal.hashCode()
        result = 31 * result + dataVal.contentHashCode()
        return result
    }

    override fun toString(): String =
        "TensorView(dtype=$dtypeVal, shape=$shapeVal, dataLen=${dataVal.size})"
}

/**
 * A single tensor information.
 * Endianness is assumed to be little endian.
 * Ordering is assumed to be C-order.
 */
public class TensorInfo internal constructor(
    /** The type of each element of the tensor. */
    public val dtype: Dtype,
    /** The shape of the tensor. */
    public val shape: List<Int>,
    /** The offsets to find the data within the byte-buffer array. */
    public val dataOffsets: ByteSpan,
    internal val rawShapeULongs: List<ULong>? = null,
) {
    public constructor(
        dtype: Dtype,
        shape: List<Int>,
        dataOffsets: ByteSpan,
    ) : this(dtype, shape, dataOffsets, null)

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is TensorInfo) return false
        return dtype == other.dtype && shape == other.shape && dataOffsets == other.dataOffsets
    }

    override fun hashCode(): Int {
        var result = dtype.hashCode()
        result = 31 * result + shape.hashCode()
        result = 31 * result + dataOffsets.hashCode()
        return result
    }

    override fun toString(): String =
        "TensorInfo(dtype=$dtype, shape=$shape, dataOffsets=$dataOffsets)"
}

/**
 * The various available dtypes. They MUST be in increasing alignment order.
 */
public enum class Dtype(
    internal val wireName: String,
) {
    /** Boolean type. */
    BOOL("BOOL"),

    /** MXF4. */
    F4("F4"),

    /** MXF6 E2M3. */
    F6_E2M3("F6_E2M3"),

    /** MXF6 E3M2. */
    F6_E3M2("F6_E3M2"),

    /** Unsigned byte. */
    U8("U8"),

    /** Signed byte. */
    I8("I8"),

    /** FP8 E5M2. */
    F8_E5M2("F8_E5M2"),

    /** FP8 E4M3. */
    F8_E4M3("F8_E4M3"),

    /** F8 E8M0. */
    F8_E8M0("F8_E8M0"),

    /** FP8 E4M3 (FNUZ). */
    F8_E4M3FNUZ("F8_E4M3FNUZ"),

    /** FP8 E5M2 (FNUZ). */
    F8_E5M2FNUZ("F8_E5M2FNUZ"),

    /** Signed integer (16-bit). */
    I16("I16"),

    /** Unsigned integer (16-bit). */
    U16("U16"),

    /** Half-precision floating point. */
    F16("F16"),

    /** Brain floating point. */
    BF16("BF16"),

    /** Signed integer (32-bit). */
    I32("I32"),

    /** Unsigned integer (32-bit). */
    U32("U32"),

    /** Floating point (32-bit). */
    F32("F32"),

    /** Complex (32-bit parts). */
    C64("C64"),

    /** Floating point (64-bit). */
    F64("F64"),

    /** Signed integer (64-bit). */
    I64("I64"),

    /** Unsigned integer (64-bit). */
    U64("U64"),
    ;

    /** Gives out the size (in bits) of 1 element of this dtype. */
    public fun bitsize(): Int =
        when (this) {
            F4 -> 4
            F6_E3M2 -> 6
            F6_E2M3 -> 6
            BOOL -> 8
            U8 -> 8
            I8 -> 8
            F8_E5M2 -> 8
            F8_E4M3 -> 8
            F8_E8M0 -> 8
            F8_E4M3FNUZ -> 8
            F8_E5M2FNUZ -> 8
            I16 -> 16
            U16 -> 16
            I32 -> 32
            U32 -> 32
            I64 -> 64
            U64 -> 64
            F16 -> 16
            BF16 -> 16
            F32 -> 32
            F64 -> 64
            C64 -> 64
        }

    /** Gives out the size (in bytes) of 1 element of this dtype. */
    public fun size(): Int = bitsize() / 8

    override fun toString(): String = wireName

    public companion object {
        public fun fromWireName(name: String): Dtype? = entries.firstOrNull { it.wireName == name }
    }
}
