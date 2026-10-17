// port-lint: tests slice.rs
package io.github.kotlinmania.safetensors

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

private fun floatToLeBytes(value: Float): ByteArray {
    val bits = value.toRawBits()
    return byteArrayOf(
        (bits and 0xFF).toByte(),
        ((bits ushr 8) and 0xFF).toByte(),
        ((bits ushr 16) and 0xFF).toByte(),
        ((bits ushr 24) and 0xFF).toByte(),
    )
}

private fun floatsToLeBytes(vararg values: Float): ByteArray {
    val out = ByteArray(values.size * 4)
    for (i in values.indices) {
        val b = floatToLeBytes(values[i])
        b.copyInto(out, i * 4)
    }
    return out
}

class SliceTest {
    @Test
    fun testHelpers() {
        val data = floatsToLeBytes(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f)
        val attn0 = TensorView.new(Dtype.F32, listOf(1, 2, 3), data)

        val iterator1 =
            SliceIterator.new(
                attn0,
                listOf(TensorIndexer.Narrow(TensorBound.Unbounded, TensorBound.Unbounded, 1)),
            )
        assertEquals(24, iterator1.remainingByteLen())
        assertEquals(listOf(1, 2, 3), iterator1.newshape())

        val iterator2 =
            SliceIterator.new(
                attn0,
                listOf(
                    TensorIndexer.Narrow(TensorBound.Unbounded, TensorBound.Unbounded, 1),
                    TensorIndexer.Narrow(TensorBound.Included(0), TensorBound.Excluded(1), 1),
                ),
            )
        assertEquals(12, iterator2.remainingByteLen())
        assertEquals(listOf(1, 1, 3), iterator2.newshape())
    }

    @Test
    fun testFp4Simple() {
        val data = byteArrayOf(0, 1)
        val attn0 = TensorView.new(Dtype.F4, listOf(1, 2, 2), data)

        val iterator1 =
            SliceIterator.new(
                attn0,
                listOf(TensorIndexer.Narrow(TensorBound.Unbounded, TensorBound.Unbounded, 1)),
            )
        assertEquals(2, iterator1.remainingByteLen())
        assertEquals(listOf(1, 2, 2), iterator1.newshape())

        val iterator2 =
            SliceIterator.new(
                attn0,
                listOf(
                    TensorIndexer.Narrow(TensorBound.Unbounded, TensorBound.Unbounded, 1),
                    TensorIndexer.Narrow(TensorBound.Included(0), TensorBound.Excluded(1), 1),
                ),
            )
        assertEquals(1, iterator2.remainingByteLen())
        assertEquals(listOf(1, 1, 2), iterator2.newshape())
    }

    @Test
    fun testFp4Misaligned() {
        val data = byteArrayOf(0)
        val attn0 = TensorView.new(Dtype.F4, listOf(1, 2), data)

        val iterator1 =
            SliceIterator.new(
                attn0,
                listOf(TensorIndexer.Narrow(TensorBound.Unbounded, TensorBound.Unbounded, 1)),
            )
        assertEquals(1, iterator1.remainingByteLen())
        assertEquals(listOf(1, 2), iterator1.newshape())

        assertFailsWith<InvalidSlice.MisalignedSlice> {
            SliceIterator.new(
                attn0,
                listOf(
                    TensorIndexer.Narrow(TensorBound.Unbounded, TensorBound.Unbounded, 1),
                    TensorIndexer.Narrow(TensorBound.Included(0), TensorBound.Excluded(1), 1),
                ),
            )
        }
    }

    @Test
    fun testDummy() {
        val data = floatsToLeBytes(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f)
        val attn0 = TensorView.new(Dtype.F32, listOf(1, 2, 3), data)

        val iterator1 =
            SliceIterator.new(
                attn0,
                listOf(TensorIndexer.Narrow(TensorBound.Unbounded, TensorBound.Unbounded, 1)),
            )
        assertContentEquals(data.copyOfRange(0, 24), iterator1.nextOrNull())
        assertNull(iterator1.nextOrNull())

        val iterator2 =
            SliceIterator.new(
                attn0,
                listOf(
                    TensorIndexer.Narrow(TensorBound.Unbounded, TensorBound.Unbounded, 1),
                    TensorIndexer.Narrow(TensorBound.Unbounded, TensorBound.Unbounded, 1),
                ),
            )
        assertContentEquals(data.copyOfRange(0, 24), iterator2.nextOrNull())
        assertNull(iterator2.nextOrNull())

        val iterator3 =
            SliceIterator.new(
                attn0,
                listOf(
                    TensorIndexer.Narrow(TensorBound.Unbounded, TensorBound.Unbounded, 1),
                    TensorIndexer.Narrow(TensorBound.Unbounded, TensorBound.Unbounded, 1),
                ),
            )
        assertContentEquals(data.copyOfRange(0, 24), iterator3.nextOrNull())
        assertNull(iterator3.nextOrNull())

        val iterator4 =
            SliceIterator.new(
                attn0,
                listOf(
                    TensorIndexer.Narrow(TensorBound.Unbounded, TensorBound.Unbounded, 1),
                    TensorIndexer.Narrow(TensorBound.Unbounded, TensorBound.Unbounded, 1),
                    TensorIndexer.Narrow(TensorBound.Unbounded, TensorBound.Unbounded, 1),
                ),
            )
        assertContentEquals(data.copyOfRange(0, 24), iterator4.nextOrNull())
        assertNull(iterator4.nextOrNull())

        assertFailsWith<InvalidSlice.TooManySlices> {
            SliceIterator.new(
                attn0,
                listOf(
                    TensorIndexer.Narrow(TensorBound.Unbounded, TensorBound.Unbounded, 1),
                    TensorIndexer.Narrow(TensorBound.Unbounded, TensorBound.Unbounded, 1),
                    TensorIndexer.Narrow(TensorBound.Unbounded, TensorBound.Unbounded, 1),
                    TensorIndexer.Narrow(TensorBound.Unbounded, TensorBound.Unbounded, 1),
                ),
            )
        }
    }

    @Test
    fun testSliceVariety() {
        val data = floatsToLeBytes(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f)
        val attn0 = TensorView.new(Dtype.F32, listOf(1, 2, 3), data)

        val iterator1 =
            SliceIterator.new(
                attn0,
                listOf(TensorIndexer.Narrow(TensorBound.Included(0), TensorBound.Excluded(1), 1)),
            )
        assertContentEquals(data.copyOfRange(0, 24), iterator1.nextOrNull())
        assertNull(iterator1.nextOrNull())

        val iterator2 =
            SliceIterator.new(
                attn0,
                listOf(
                    TensorIndexer.Narrow(TensorBound.Unbounded, TensorBound.Unbounded, 1),
                    TensorIndexer.Narrow(TensorBound.Included(0), TensorBound.Excluded(1), 1),
                ),
            )
        assertContentEquals(data.copyOfRange(0, 12), iterator2.nextOrNull())
        assertNull(iterator2.nextOrNull())

        val iterator3 =
            SliceIterator.new(
                attn0,
                listOf(
                    TensorIndexer.Narrow(TensorBound.Unbounded, TensorBound.Unbounded, 1),
                    TensorIndexer.Narrow(TensorBound.Unbounded, TensorBound.Unbounded, 1),
                    TensorIndexer.Narrow(TensorBound.Included(0), TensorBound.Excluded(1), 1),
                ),
            )
        assertContentEquals(data.copyOfRange(0, 4), iterator3.nextOrNull())
        assertContentEquals(data.copyOfRange(12, 16), iterator3.nextOrNull())
        assertNull(iterator3.nextOrNull())

        val iterator4 =
            SliceIterator.new(
                attn0,
                listOf(
                    TensorIndexer.Narrow(TensorBound.Unbounded, TensorBound.Unbounded, 1),
                    TensorIndexer.Narrow(TensorBound.Included(1), TensorBound.Excluded(2), 1),
                    TensorIndexer.Narrow(TensorBound.Included(0), TensorBound.Excluded(1), 1),
                ),
            )
        assertContentEquals(data.copyOfRange(12, 16), iterator4.nextOrNull())
        assertNull(iterator4.nextOrNull())
    }

    @Test
    fun testSliceVariety2() {
        val data = floatsToLeBytes(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f)
        val attn0 = TensorView.new(Dtype.F32, listOf(2, 3), data)

        val iterator =
            SliceIterator.new(
                attn0,
                listOf(
                    TensorIndexer.Narrow(TensorBound.Unbounded, TensorBound.Unbounded, 1),
                    TensorIndexer.Narrow(TensorBound.Included(1), TensorBound.Excluded(3), 1),
                ),
            )
        assertTrue(iterator.hasNext())
        assertContentEquals(data.copyOfRange(4, 12), iterator.next())
        assertTrue(iterator.hasNext())
        assertContentEquals(data.copyOfRange(16, 24), iterator.next())
        assertFalse(iterator.hasNext())
    }

    @Test
    fun testSliceSelect() {
        val data = floatsToLeBytes(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f)
        val attn0 = TensorView.new(Dtype.F32, listOf(2, 3), data)

        val iterator1 =
            SliceIterator.new(
                attn0,
                listOf(
                    TensorIndexer.Select(1),
                    TensorIndexer.Narrow(TensorBound.Included(1), TensorBound.Excluded(3), 1),
                ),
            )
        assertContentEquals(data.copyOfRange(16, 24), iterator1.nextOrNull())
        assertNull(iterator1.nextOrNull())

        val iterator2 =
            SliceIterator.new(
                attn0,
                listOf(
                    TensorIndexer.Select(0),
                    TensorIndexer.Narrow(TensorBound.Included(1), TensorBound.Excluded(3), 1),
                ),
            )
        assertContentEquals(data.copyOfRange(4, 12), iterator2.nextOrNull())
        assertNull(iterator2.nextOrNull())

        val iterator3 =
            SliceIterator.new(
                attn0,
                listOf(
                    TensorIndexer.Narrow(TensorBound.Included(1), TensorBound.Excluded(2), 1),
                    TensorIndexer.Select(0),
                ),
            )
        assertContentEquals(data.copyOfRange(12, 16), iterator3.nextOrNull())
        assertNull(iterator3.nextOrNull())
    }

    @Test
    fun testInvalidRange() {
        val data = floatsToLeBytes(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f)
        val attn0 = TensorView.new(Dtype.F32, listOf(2, 3), data)

        val err1 =
            assertFailsWith<InvalidSlice.SliceOutOfRange> {
                SliceIterator.new(
                    attn0,
                    listOf(
                        TensorIndexer.Select(1),
                        TensorIndexer.Narrow(TensorBound.Included(1), TensorBound.Excluded(4), 1),
                    ),
                )
            }
        assertEquals(3, err1.asked)
        assertEquals(1, err1.dimIndex)
        assertEquals(3, err1.dimSize)

        val err2 =
            assertFailsWith<InvalidSlice.SliceOutOfRange> {
                SliceIterator.new(
                    attn0,
                    listOf(
                        TensorIndexer.Select(1),
                        TensorIndexer.Narrow(TensorBound.Included(3), TensorBound.Excluded(2), 1),
                    ),
                )
            }
        assertEquals(3, err2.asked)
        assertEquals(1, err2.dimIndex)
        assertEquals(3, err2.dimSize)

        assertFailsWith<InvalidSlice.TooManySlices> {
            SliceIterator.new(
                attn0,
                listOf(
                    TensorIndexer.Select(1),
                    TensorIndexer.Select(1),
                    TensorIndexer.Select(1),
                ),
            )
        }
    }
}
