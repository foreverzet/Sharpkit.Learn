// -----------------------------------------------------------------------
// <copyright file="DoubleArrayPointer.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

namespace Liblinear
{
    using System;

    /// <summary>
    /// TODO: Update summary.
    /// </summary>
    public sealed class ArrayPointer<T>
    {
        private readonly T[] _array;
        private int _offset;

        public void setOffset(int offset)
        {
            if (offset < 0 || offset >= _array.Length) throw new ArgumentException("offset must be between 0 and the length of the array");
            _offset = offset;
        }

        public ArrayPointer(T[] array, int offset)
        {
            _array = array;
            setOffset(offset);
        }


        public T this[int index]
        {
            get { return _array[_offset + index]; }
            set { _array[_offset + index] = value; }
        }
    }
}
