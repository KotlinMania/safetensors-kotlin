// port-lint: source tensor.rs
package io.github.kotlinmania.safetensors

import io.github.kotlinmania.serde.SerdeError
import io.github.kotlinmania.serde.SerdeException
import io.github.kotlinmania.serde.SerdeResult
import io.github.kotlinmania.serde.serdeCatching
import io.github.kotlinmania.serdecore.de.Deserialize
import io.github.kotlinmania.serdecore.de.DeserializeSeed
import io.github.kotlinmania.serdecore.de.Deserializer
import io.github.kotlinmania.serdecore.de.EnumAccess
import io.github.kotlinmania.serdecore.de.MapAccess
import io.github.kotlinmania.serdecore.de.StringDeserialize
import io.github.kotlinmania.serdecore.de.U64Deserialize
import io.github.kotlinmania.serdecore.de.Visitor
import io.github.kotlinmania.serdecore.de.mutableListDeserialize
import io.github.kotlinmania.serdecore.de.pairDeserialize
import io.github.kotlinmania.serdecore.ser.Serialize
import io.github.kotlinmania.serdecore.ser.Serializer
import io.github.kotlinmania.serdecore.ser.serialize
import io.github.kotlinmania.serdejson.Value
import io.github.kotlinmania.serdejson.fromStr
import io.github.kotlinmania.serdejson.toVec

internal const val MAX_HEADER_SIZE: Int = 100_000_000
internal const val N_LEN: Int = 8
internal const val MAX_DIMENSION: Int = 8
internal const val MAX_SIZE: Int = 8
internal const val MAX_TENSORS: Int = 8

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

    /**
     * The lower-level source of this error, if any.
     */
    public fun source(): Throwable? = cause

    public companion object {
        public fun from(error: String): SafeTensorError = IoError(error)

        public fun from(cause: Throwable): SafeTensorError =
            when (cause) {
                is SafeTensorError -> cause
                else -> IoError(cause.message ?: "unknown I/O error")
            }
    }

    override fun toString(): String = message.orEmpty()
}

/**
 * Type alias matching upstream `SafeTensorError` error type.
 */
public typealias Error = SafeTensorError

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

private fun serializeMetadataToJsonBytes(metadata: Metadata): ByteArray =
    toVec(metadata).getOrElse { e ->
        throw SafeTensorError.JsonError(e.message)
    }

private fun nextMultipleOf(value: Int, multiple: Int): Int {
    val rem = value % multiple
    return if (rem == 0) value else value + (multiple - rem)
}

private fun prepare(
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
    val (prepared, tensors) = prepare(entries, dataInfo)
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

private fun bufferedWriteToFile(
    filename: String,
    n: ULong,
    headerBytes: ByteArray,
    tensors: List<View>,
    totalSize: Int,
) {
    val buffer = ByteArray(totalSize)
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
    virtualFileStore[filename] = buffer
}

/**
 * Serialize a map of tensors to a file at [filename].
 */
public fun serializeToFile(
    data: Map<String, View>,
    dataInfo: Map<String, String>?,
    filename: String,
) {
    val entries = data.entries.map { it.key to it.value }
    val (prepared, tensors) = prepare(entries, dataInfo)
    if (prepared.n > MAX_HEADER_SIZE.toULong()) {
        throw SafeTensorError.HeaderTooLarge()
    }
    val totalSize = N_LEN + prepared.headerBytes.size + prepared.offset
    bufferedWriteToFile(filename, prepared.n, prepared.headerBytes, tensors, totalSize)
}

/**
 * Serialize a list of named tensor views to a file at [filename].
 */
public fun serializeNamedToFile(
    data: List<NamedTensorView>,
    dataInfo: Map<String, String>?,
    filename: String,
) {
    val entries = data.map { it.name to it.view }
    val (prepared, tensors) = prepare(entries, dataInfo)
    if (prepared.n > MAX_HEADER_SIZE.toULong()) {
        throw SafeTensorError.HeaderTooLarge()
    }
    val totalSize = N_LEN + prepared.headerBytes.size + prepared.offset
    bufferedWriteToFile(filename, prepared.n, prepared.headerBytes, tensors, totalSize)
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
            val hashdata = parseHashMetadata(string)
            val metadata = hashdata.tryInto()
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

private class SeedOf<T>(
    private val deserialize: Deserialize<T>,
) : DeserializeSeed<T> {
    override fun <D : Deserializer> deserialize(deserializer: D): SerdeResult<T> =
        deserialize.deserialize(deserializer)
}

private class StringSerialize(
    private val value: String,
) : Serialize {
    override fun <Ok> serialize(serializer: Serializer<Ok>): SerdeResult<Ok> = value.serialize(serializer)
}

private class StringMapSerialize(
    private val value: Map<String, String>,
) : Serialize {
    override fun <Ok> serialize(serializer: Serializer<Ok>): SerdeResult<Ok> = value.serialize(serializer)
}

private class IntListSerialize(
    private val value: List<Int>,
) : Serialize {
    override fun <Ok> serialize(serializer: Serializer<Ok>): SerdeResult<Ok> = value.serialize(serializer)
}

private class OffsetsPairSerialize(
    private val start: Int,
    private val stop: Int,
) : Serialize {
    override fun <Ok> serialize(serializer: Serializer<Ok>): SerdeResult<Ok> = Pair(start, stop).serialize(serializer)
}

/**
 * Helper struct used only for serialization and deserialization.
 */
internal class HashMetadata(
    val metadata: Map<String, String>?,
    val tensors: Map<String, TensorInfo>,
) {
    public companion object : Deserialize<HashMetadata> {
        override fun <D : Deserializer> deserialize(deserializer: D): SerdeResult<HashMetadata> =
            deserializer.deserializeMap(
                object : Visitor<HashMetadata> {
                    override fun expecting(): String = "struct HashMetadata"

                    override fun <A : MapAccess> visitMap(access: A): SerdeResult<HashMetadata> =
                        serdeCatching {
                            var meta: Map<String, String>? = null
                            val tensorsMap = LinkedHashMap<String, TensorInfo>()
                            val metaDeserialize =
                                io.github.kotlinmania.serdecore.de.mapDeserialize(
                                    StringDeserialize,
                                    StringDeserialize,
                                )
                            while (true) {
                                val key = access.nextKeySeed(SeedOf(StringDeserialize)).getOrThrow() ?: break
                                if (key == "__metadata__") {
                                    meta = access.nextValueSeed(SeedOf(metaDeserialize)).getOrThrow()
                                } else {
                                    val info = access.nextValueSeed(SeedOf(TensorInfo.Companion)).getOrThrow()
                                    tensorsMap[key] = info
                                }
                            }
                            HashMetadata(meta, tensorsMap)
                        }
                },
            )

        public fun tryFrom(hashdata: HashMetadata): Metadata {
            val sortedTensors =
                hashdata.tensors.entries
                    .map { NamedTensorInfo(it.key, it.value) }
                    .toMutableList()
            sortedTensors.sortWith { a, b ->
                val aOffsets = a.info.dataOffsets
                val bOffsets = b.info.dataOffsets
                val startCmp = aOffsets.start.compareTo(bOffsets.start)
                if (startCmp != 0) startCmp else aOffsets.stop.compareTo(bOffsets.stop)
            }
            return Metadata.new(hashdata.metadata, sortedTensors)
        }
    }

    public fun tryInto(): Metadata = tryFrom(this)
}

private fun parseHashMetadata(jsonString: String): HashMetadata {
    val trimmed = jsonString.trim { it == ' ' || it == '\t' || it == '\n' || it == '\r' }
    if (!trimmed.startsWith("{") || !trimmed.endsWith("}")) {
        throw SafeTensorError.InvalidHeaderDeserialization("expected JSON object")
    }
    val element: Value =
        fromStr(trimmed).getOrElse { e ->
            throw SafeTensorError.InvalidHeaderDeserialization(e.message)
        }
    val rootObj =
        element.asObject()
            ?: throw SafeTensorError.InvalidHeaderDeserialization("header root must be a JSON object")

    var metadataMap: Map<String, String>? = null
    val tensorsMap = LinkedHashMap<String, TensorInfo>()

    for ((key, value) in rootObj) {
        if (key == "__metadata__") {
            val metaObj =
                value.asObject()
                    ?: throw SafeTensorError.InvalidHeaderDeserialization("__metadata__ must be an object")
            val map = LinkedHashMap<String, String>(metaObj.len())
            for ((mk, mv) in metaObj) {
                val strVal =
                    mv.asStr()
                        ?: throw SafeTensorError.InvalidHeaderDeserialization("metadata value must be a string")
                map[mk] = strVal
            }
            metadataMap = map
        } else {
            val tensorObj =
                value.asObject()
                    ?: throw SafeTensorError.InvalidHeaderDeserialization("tensor entry '$key' must be an object")
            val dtypeStr =
                tensorObj["dtype"]?.asStr()
                    ?: throw SafeTensorError.InvalidHeaderDeserialization("missing dtype for '$key'")
            val dtype =
                Dtype.fromWireName(dtypeStr)
                    ?: throw SafeTensorError.InvalidHeaderDeserialization("unknown dtype '$dtypeStr'")

            val shapeArr =
                tensorObj["shape"]?.asArray()
                    ?: throw SafeTensorError.InvalidHeaderDeserialization("missing shape for '$key'")
            val shapeULongs = ArrayList<ULong>(shapeArr.size)
            for (dimElem in shapeArr) {
                val dimULong =
                    dimElem.asU64()
                        ?: throw SafeTensorError.InvalidHeaderDeserialization("shape element must be integer")
                shapeULongs.add(dimULong)
            }

            val offsetsArr =
                tensorObj["data_offsets"]?.asArray()
                    ?: throw SafeTensorError.InvalidHeaderDeserialization("missing data_offsets for '$key'")
            if (offsetsArr.size != 2) {
                throw SafeTensorError.InvalidHeaderDeserialization("data_offsets must have 2 elements")
            }
            val startULong =
                offsetsArr[0].asU64()
                    ?: throw SafeTensorError.InvalidHeaderDeserialization("invalid start offset")
            val stopULong =
                offsetsArr[1].asU64()
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
            tensorsMap[key] = info
        }
    }

    return HashMetadata(metadataMap, tensorsMap)
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
) : Serialize {
    public companion object : Deserialize<Metadata> {
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

        override fun <D : Deserializer> deserialize(deserializer: D): SerdeResult<Metadata> =
            serdeCatching {
                val hashdata = HashMetadata.deserialize(deserializer).getOrThrow()
                try {
                    hashdata.tryInto()
                } catch (e: SafeTensorError) {
                    throw SerdeException(SerdeError.custom(e.message ?: "invalid metadata"))
                }
            }
    }

    override fun <Ok> serialize(serializer: Serializer<Ok>): SerdeResult<Ok> =
        serdeCatching {
            val names = Array(indexMap.size) { "" }
            for ((name, index) in indexMap) {
                names[index] = name
            }
            val length = if (metadataMap != null) 1 else 0
            val map = serializer.serializeMap(tensorsList.size + length).getOrThrow()
            if (metadataMap != null) {
                map.serializeEntry(StringSerialize("__metadata__"), StringMapSerialize(metadataMap)).getOrThrow()
            }
            for (i in names.indices) {
                map.serializeEntry(StringSerialize(names[i]), tensorsList[i]).getOrThrow()
            }
            map.end().getOrThrow()
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
) : View,
    IndexOp<TensorIndexer> {
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
    override fun slice(index: TensorIndexer): SliceIterator = slicedData(listOf(index))

    /** Slice this tensor along 2 dimensions. */
    public fun slice(
        index0: TensorIndexer,
        index1: TensorIndexer,
    ): SliceIterator = slicedData(listOf(index0, index1))

    /** Slice this tensor along 2 dimensions via [Pair]. */
    public fun slice(index: Pair<TensorIndexer, TensorIndexer>): SliceIterator =
        slicedData(listOf(index.first, index.second))

    /** Slice this tensor along 3 dimensions. */
    public fun slice(
        index0: TensorIndexer,
        index1: TensorIndexer,
        index2: TensorIndexer,
    ): SliceIterator = slicedData(listOf(index0, index1, index2))

    /** Slice this tensor along 3 dimensions via [Triple]. */
    public fun slice(index: Triple<TensorIndexer, TensorIndexer, TensorIndexer>): SliceIterator =
        slicedData(listOf(index.first, index.second, index.third))

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
) : Serialize {
    public constructor(
        dtype: Dtype,
        shape: List<Int>,
        dataOffsets: ByteSpan,
    ) : this(dtype, shape, dataOffsets, null)

    public companion object : Deserialize<TensorInfo> {
        private val FIELDS: List<String> = listOf("dtype", "shape", "data_offsets")

        override fun <D : Deserializer> deserialize(deserializer: D): SerdeResult<TensorInfo> =
            deserializer.deserializeStruct(
                "TensorInfo",
                FIELDS,
                object : Visitor<TensorInfo> {
                    override fun expecting(): String = "struct TensorInfo"

                    override fun <A : MapAccess> visitMap(access: A): SerdeResult<TensorInfo> =
                        serdeCatching {
                            var dtypeVal: Dtype? = null
                            var shapeULongs: List<ULong>? = null
                            var offsetsPair: Pair<ULong, ULong>? = null
                            val shapeDe = mutableListDeserialize(U64Deserialize)
                            val offsetsDe = pairDeserialize(U64Deserialize, U64Deserialize)
                            while (true) {
                                val key = access.nextKeySeed(SeedOf(StringDeserialize)).getOrThrow() ?: break
                                when (key) {
                                    "dtype" -> {
                                        if (dtypeVal != null) throw SerdeException(SerdeError.duplicateField("dtype"))
                                        dtypeVal = access.nextValueSeed(SeedOf(Dtype.Companion)).getOrThrow()
                                    }

                                    "shape" -> {
                                        if (shapeULongs != null) throw SerdeException(SerdeError.duplicateField("shape"))
                                        shapeULongs = access.nextValueSeed(SeedOf(shapeDe)).getOrThrow()
                                    }

                                    "data_offsets" -> {
                                        if (offsetsPair != null) throw SerdeException(SerdeError.duplicateField("data_offsets"))
                                        offsetsPair = access.nextValueSeed(SeedOf(offsetsDe)).getOrThrow()
                                    }

                                    else -> {
                                        throw SerdeException(SerdeError.unknownField(key, FIELDS))
                                    }
                                }
                            }
                            val dt = dtypeVal ?: throw SerdeException(SerdeError.missingField("dtype"))
                            val shULongs = shapeULongs ?: throw SerdeException(SerdeError.missingField("shape"))
                            val off = offsetsPair ?: throw SerdeException(SerdeError.missingField("data_offsets"))
                            val shapeInts = shULongs.map { d -> if (d > Int.MAX_VALUE.toULong()) Int.MAX_VALUE else d.toInt() }
                            val startInt = if (off.first > Int.MAX_VALUE.toULong()) Int.MAX_VALUE else off.first.toInt()
                            val stopInt = if (off.second > Int.MAX_VALUE.toULong()) Int.MAX_VALUE else off.second.toInt()
                            TensorInfo(
                                dtype = dt,
                                shape = shapeInts,
                                dataOffsets = ByteSpan(startInt, stopInt),
                                rawShapeULongs = shULongs,
                            )
                        }
                },
            )
    }

    override fun <Ok> serialize(serializer: Serializer<Ok>): SerdeResult<Ok> =
        serdeCatching {
            val state = serializer.serializeStruct("TensorInfo", 3).getOrThrow()
            state.serializeField("dtype", dtype).getOrThrow()
            state.serializeField("shape", IntListSerialize(shape)).getOrThrow()
            state.serializeField("data_offsets", OffsetsPairSerialize(dataOffsets.start, dataOffsets.stop)).getOrThrow()
            state.end().getOrThrow()
        }

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
) : Serialize {
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

    override fun <Ok> serialize(serializer: Serializer<Ok>): SerdeResult<Ok> =
        serializer.serializeUnitVariant("Dtype", ordinal.toUInt(), wireName)

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

    public companion object : Deserialize<Dtype> {
        private val VARIANTS: List<String> = entries.map { it.wireName }

        public fun fromWireName(name: String): Dtype? = entries.firstOrNull { it.wireName == name }

        override fun <D : Deserializer> deserialize(deserializer: D): SerdeResult<Dtype> =
            deserializer.deserializeEnum(
                "Dtype",
                VARIANTS,
                object : Visitor<Dtype> {
                    override fun expecting(): String = "enum Dtype"

                    override fun visitStr(v: String): SerdeResult<Dtype> =
                        fromWireName(v)?.let { SerdeResult.success(it) }
                            ?: SerdeResult.failure(SerdeError.unknownVariant(v, VARIANTS))

                    override fun visitString(v: String): SerdeResult<Dtype> = visitStr(v)

                    override fun <A : EnumAccess> visitEnum(access: A): SerdeResult<Dtype> =
                        serdeCatching {
                            val (variantName, variant) = access.variantSeed(SeedOf(StringDeserialize)).getOrThrow()
                            variant.unitVariant().getOrThrow()
                            fromWireName(variantName)
                                ?: throw SerdeException(SerdeError.unknownVariant(variantName, VARIANTS))
                        }
                },
            )
    }
}
