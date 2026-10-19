// port-lint: source slice.rs
package io.github.kotlinmania.safetensors

/**
 * Error representing invalid slicing attempt.
 */
public sealed class InvalidSlice(
    message: String,
) : Exception(message) {
    /** When the client asked for more slices than the tensor has dimensions. */
    public class TooManySlices : InvalidSlice("more slicing indexes than dimensions in tensor") {
        override fun equals(other: Any?): Boolean = other is TooManySlices

        override fun hashCode(): Int = 1
    }

    /** When the client asked for a slice that exceeds the allowed bounds. */
    public data class SliceOutOfRange(
        /** The rank of the dimension that has the out of bounds. */
        public val dimIndex: Int,
        /** The problematic value. */
        public val asked: Int,
        /** The dimension size we shouldn't go over. */
        public val dimSize: Int,
    ) : InvalidSlice("index $asked out of bounds for tensor dimension #$dimIndex of size $dimSize")

    /**
     * For smaller than 1 byte dtypes, some slices will happen outside of the byte boundary;
     * some special care has to be taken and standard functions will fail.
     */
    public class MisalignedSlice :
        InvalidSlice(
            "The slice is slicing for subbytes dtypes, and the slice does not end up at a byte boundary, this is invalid.",
        ) {
        override fun equals(other: Any?): Boolean = other is MisalignedSlice

        override fun hashCode(): Int = 2
    }

    override fun toString(): String = message.orEmpty()
}

/**
 * Endpoint bound of a tensor slice range.
 */
public sealed class TensorBound {
    /** Unbounded endpoint. */
    public data object Unbounded : TensorBound()

    /** Inclusive endpoint at [index]. */
    public data class Included(
        public val index: Int,
    ) : TensorBound()

    /** Exclusive endpoint at [index]. */
    public data class Excluded(
        public val index: Int,
    ) : TensorBound()
}

private fun displayBound(bound: TensorBound): String =
    when (bound) {
        is TensorBound.Unbounded -> ""
        is TensorBound.Excluded -> bound.index.toString()
        is TensorBound.Included -> bound.index.toString()
    }

/**
 * Structure used to index a slice of the tensor.
 */
public sealed class TensorIndexer {
    /** Selecting an entire dimension at [index]. */
    public data class Select(
        public val index: Int,
    ) : TensorIndexer()

    /**
     * A slice `start:stop:step`. [step] is always >= 1; a contiguous slice has `step == 1`.
     */
    public data class Narrow(
        public val left: TensorBound,
        public val right: TensorBound,
        public val step: Int = 1,
    ) : TensorIndexer() {
        init {
            require(step >= 1) { "step must be >= 1" }
        }
    }

    public companion object {
        public fun from(index: Int): TensorIndexer = Select(index)

        public fun from(range: IntRange): TensorIndexer =
            if (range.last == Int.MAX_VALUE) {
                Narrow(TensorBound.Included(range.first), TensorBound.Unbounded, 1)
            } else {
                Narrow(TensorBound.Included(range.first), TensorBound.Excluded(range.last + 1), 1)
            }

        public fun full(): TensorIndexer = Narrow(TensorBound.Unbounded, TensorBound.Unbounded, 1)

        public fun fromStart(start: Int): TensorIndexer =
            Narrow(TensorBound.Included(start), TensorBound.Unbounded, 1)

        public fun toExclusive(endExclusive: Int): TensorIndexer =
            Narrow(TensorBound.Unbounded, TensorBound.Excluded(endExclusive), 1)

        public fun toInclusive(endInclusive: Int): TensorIndexer =
            Narrow(TensorBound.Unbounded, TensorBound.Included(endInclusive), 1)
    }

    override fun toString(): String =
        when (this) {
            is Select -> index.toString()
            is Narrow -> if (step == 1) "${displayBound(left)}:${displayBound(right)}" else "${displayBound(left)}:${displayBound(right)}:$step"
        }
}

/**
 * Trait used to implement multiple signatures for ease of use of the slicing
 * of a tensor.
 */
public interface IndexOp<T> {
    /**
     * Returns a slicing iterator which are the chunks of data necessary to
     * reconstruct the desired tensor.
     */
    public fun slice(index: T): SliceIterator
}

/**
 * Byte ranges into a tensor's data section in iteration order;
 * concatenating them yields the dense destination layout.
 */
public typealias SliceByteRanges = List<ByteSpan>

/**
 * Post-slice tensor shape (element counts per dim).
 */
public typealias SlicedShape = List<Int>

/**
 * Element item type of [SliceIterator].
 */
public typealias Item = ByteArray

/**
 * Half-open byte range `[start, stop)` into a tensor's data section.
 */
public data class ByteSpan(
    public val start: Int,
    public val stop: Int,
)

/**
 * Computed byte ranges and post-slice shape for a slicing operation.
 */
public data class SliceByteRangesOutput(
    public val ranges: SliceByteRanges,
    public val shape: SlicedShape,
)

/**
 * Iterator used to return the chunks of the overall tensor buffer
 * when a client asks for a slice of the original tensor.
 */
public class SliceIterator internal constructor(
    private val view: TensorView,
    private val indices: MutableList<ByteSpan>,
    private val newshapeList: List<Int>,
) : Iterator<ByteArray> {
    public companion object {
        public fun new(
            view: TensorView,
            slices: List<TensorIndexer>,
        ): SliceIterator {
            val (ranges, newshape) = sliceByteRanges(view.dtype(), view.shape(), slices)
            val reversed = ranges.asReversed().toMutableList()
            return SliceIterator(view, reversed, newshape)
        }
    }

    /** Gives back the amount of bytes still remaining in the iterator. */
    public fun remainingByteLen(): Int = indices.sumOf { span -> span.stop - span.start }

    /** Gives back the post-slice shape of the tensor. */
    public fun newshape(): List<Int> = newshapeList.toList()

    override fun hasNext(): Boolean = indices.isNotEmpty()

    override fun next(): ByteArray {
        if (indices.isEmpty()) {
            throw NoSuchElementException("SliceIterator has no more elements")
        }
        val span = indices.removeAt(indices.lastIndex)
        return view.data().copyOfRange(span.start, span.stop)
    }

    /** Returns the next chunk or `null` if iteration has completed. */
    public fun nextOrNull(): ByteArray? = if (hasNext()) next() else null

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is SliceIterator) return false
        return view == other.view && indices == other.indices && newshapeList == other.newshapeList
    }

    override fun hashCode(): Int {
        var result = view.hashCode()
        result = 31 * result + indices.hashCode()
        result = 31 * result + newshapeList.hashCode()
        return result
    }
}

/**
 * Resolve a `(start, stop)` half-open element range from slice bounds,
 * defaulting unbounded ends to `0` and [dim].
 */
private fun narrowBounds(
    left: TensorBound,
    right: TensorBound,
    dim: Int,
): ByteSpan {
    val start =
        when (left) {
            is TensorBound.Unbounded -> 0
            is TensorBound.Included -> left.index
            is TensorBound.Excluded -> left.index + 1
        }
    val stop =
        when (right) {
            is TensorBound.Unbounded -> dim
            is TensorBound.Included -> right.index + 1
            is TensorBound.Excluded -> right.index
        }
    return ByteSpan(start, stop)
}

private fun divCeil(a: Int, b: Int): Int = (a + b - 1) / b

/**
 * Compute the byte ranges and post-slice shape for a slicing operation
 * without requiring the underlying data buffer.
 */
public fun sliceByteRanges(
    dtype: Dtype,
    shape: List<Int>,
    slices: List<TensorIndexer>,
): SliceByteRangesOutput {
    val nSlice = slices.size
    val nShape = shape.size
    if (nSlice > nShape) {
        throw InvalidSlice.TooManySlices()
    }
    val newshape = ArrayList<Int>(nShape)

    var span = dtype.bitsize()
    val indices = ArrayList<ByteSpan>()

    for (i in shape.indices.reversed()) {
        val dim = shape[i]
        if (i >= slices.size) {
            newshape.add(dim)
        } else {
            val slice = slices[i]
            val start: Int
            val stop: Int
            val step: Int
            when (slice) {
                is TensorIndexer.Select -> {
                    start = slice.index
                    stop = slice.index + 1
                    step = 1
                }

                is TensorIndexer.Narrow -> {
                    val bounds = narrowBounds(slice.left, slice.right, dim)
                    start = bounds.start
                    stop = bounds.stop
                    step = slice.step
                }
            }
            if (start < 0 || start >= dim || stop < 0 || stop > dim) {
                val asked =
                    if (start >= dim || start < 0) {
                        start
                    } else {
                        maxOf(0, stop - 1)
                    }
                throw InvalidSlice.SliceOutOfRange(
                    dimIndex = i,
                    asked = asked,
                    dimSize = dim,
                )
            }
            if (slice !is TensorIndexer.Select) {
                val count = if (stop > start) divCeil(stop - start, step) else 0
                newshape.add(count)
            }
            if (indices.isEmpty()) {
                if (step == 1 && start == 0 && stop == dim) {
                    // Full range, nothing sliced yet; just grow the span.
                } else if (step == 1) {
                    if ((start * span) % 8 != 0) {
                        throw InvalidSlice.MisalignedSlice()
                    }
                    val offset = (start * span) / 8
                    if ((stop * span) % 8 != 0) {
                        throw InvalidSlice.MisalignedSlice()
                    }
                    val smallSpan = (stop * span) / 8 - offset
                    indices.add(ByteSpan(offset, offset + smallSpan))
                } else {
                    var n = start
                    while (n < stop) {
                        if ((n * span) % 8 != 0 || ((n + 1) * span) % 8 != 0) {
                            throw InvalidSlice.MisalignedSlice()
                        }
                        indices.add(ByteSpan((n * span) / 8, ((n + 1) * span) / 8))
                        n += step
                    }
                }
            } else {
                val stepsCount = if (stop > start) divCeil(stop - start, step) else 0
                val capacity = stepsCount * indices.size
                val newindices = ArrayList<ByteSpan>(capacity)
                var n = start
                while (n < stop) {
                    if ((n * span) % 8 != 0) {
                        throw InvalidSlice.MisalignedSlice()
                    }
                    val offset = (n * span) / 8
                    for (old in indices) {
                        newindices.add(ByteSpan(old.start + offset, old.stop + offset))
                    }
                    n += step
                }
                indices.clear()
                indices.addAll(newindices)
            }
        }
        span *= dim
    }
    if (indices.isEmpty()) {
        val totalBits = span
        if (totalBits % 8 != 0) {
            throw InvalidSlice.MisalignedSlice()
        }
        indices.add(ByteSpan(0, totalBits / 8))
    }
    newshape.reverse()
    return SliceByteRangesOutput(indices, newshape)
}
