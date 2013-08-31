// -----------------------------------------------------------------------
// <copyright file="MatrixExtensions.cs" company="">
// 
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn
{
    using System;
    using System.Linq;
    using MathNet.Numerics.LinearAlgebra.Double;
    using MathNet.Numerics.LinearAlgebra.Generic;

    /// <summary>
    /// <see cref="Matrix"/> and <see cref="Vector"/> extention methods.
    /// </summary>
    public static class MatrixExtensions
    {
        /// <summary>
        /// Returns dimensions of a matrix as Tuple.
        /// </summary>
        /// <param name="matrix">Matrix to get dimensions.</param>
        /// <returns>Tuple (rows, columns).</returns>
        public static Tuple<int, int> Shape(this Matrix<double> matrix)
        {
            return Tuple.Create(matrix.RowCount, matrix.ColumnCount);
        }

        /// <summary>
        /// Returns matrix where every element is square of corresponding
        /// element in the original matrix.
        /// </summary>
        /// <param name="matrix">Matrix which elements will be squared.</param>
        /// <returns>Matrix with squared elements.</returns>
        public static Matrix<double> Sqr(this Matrix<double> matrix)
        {
            return matrix.PointwiseMultiply(matrix);
        }

        /// <summary>
        /// Returns <see cref="Vector"/> where every element is square root of corresponding
        /// element in the original vector.
        /// </summary>
        /// <param name="vector">Original vector.</param>
        /// <returns>Vector with square root elements.</returns>
        public static Vector<double> Sqrt(this Vector<double> vector)
        {
            var newvec = vector.Clone();
            newvec.MapInplace(Math.Sqrt);
            return newvec;
        }

        /// <summary>
        /// Returns matrix where every element is Log of corresponding
        /// element in the original matrix.
        /// </summary>
        /// <param name="matrix">Matrix which elements will be transformed.</param>
        /// <returns>Resulting matrix.</returns>
        public static Matrix<double> Log(this Matrix<double> matrix)
        {
            var newMatrix = matrix.Clone();
            newMatrix.MapInplace(Math.Log);
            return newMatrix;
        }

        /// <summary>
        /// Returns vector where every element is square of corresponding
        /// element in the original vector.
        /// </summary>
        /// <param name="vector">Vector which elements will be squared.</param>
        /// <returns>Vector with squared elements.</returns>
        public static Vector<double> Sqr(this Vector<double> vector)
        {
            return vector.PointwiseMultiply(vector);
        }

        /// <summary>
        /// Returns sum of all values of a matrix.
        /// </summary>
        /// <param name="matrix">Matrix to calculate sum for.</param>
        /// <returns>Sum of all elements of the matrix.</returns>
        public static double Sum(this Matrix<double> matrix)
        {
            return matrix.RowEnumerator().SelectMany(r => r.Item2).Sum();
        }

        /// <summary>
        /// Subtracts <paramref name="vector"/> from all elements of matrix <see cref="matrix"/>
        /// and places result into <paramref name="destMatrix"/>.
        /// </summary>
        /// <param name="matrix">Source matrix.</param>
        /// <param name="vector">Vector to subtract.</param>
        /// <param name="destMatrix">Resulting matrix.</param>
        public static void SubtractRowVector(this Matrix<double> matrix, Vector<double> vector, Matrix<double> destMatrix)
        {
            foreach (var row in matrix.RowEnumerator())
            {
                destMatrix.SetRow(row.Item1, row.Item2.Subtract(vector));
            }
        }

        /// <summary>
        /// Subtracts <paramref name="vector"/> from all rows elements of matrix <see cref="matrix"/>.
        /// </summary>
        /// <param name="matrix">Source matrix.</param>
        /// <param name="vector">Vector to subtract.</param>
        /// <returns>
        /// Resulting matrix.
        /// </returns>
        public static Matrix<double> SubtractRowVector(this Matrix<double> matrix, Vector<double> vector)
        {
            var destMatrix = matrix.Clone();
            matrix.SubtractRowVector(vector, destMatrix);
            return destMatrix;
        }

        /// <summary>
        /// Divides rows of matrix <paramref name="matrix"/> by <paramref name="vector"/> pointwise
        /// and places result into <paramref name="destMatrix"/>.
        /// </summary>
        /// <param name="matrix">Source matrix.</param>
        /// <param name="vector">Divider.</param>
        /// <param name="destMatrix">Resulting matrix.</param>
        public static void DivRowVector(this Matrix<double> matrix, Vector<double> vector, Matrix<double> destMatrix)
        {
            foreach (var row in matrix.RowEnumerator())
            {
                destMatrix.SetRow(row.Item1, row.Item2.PointwiseDivide(vector));
            }
        }

        /// <summary>
        /// Divides all columns of matrix <paramref name="matrix"/> by <paramref name="vector"/> pointwise
        /// and places result into <paramref name="destMatrix"/>.
        /// </summary>
        /// <param name="matrix">Source matrix.</param>
        /// <param name="vector">Divider.</param>
        /// <param name="destMatrix">Resulting matrix.</param>
        public static void DivColumnVector(this Matrix<double> matrix, Vector<double> vector, Matrix<double> destMatrix)
        {
            foreach (var column in matrix.ColumnEnumerator())
            {
                destMatrix.SetColumn(column.Item1, column.Item2.PointwiseDivide(vector));
            }
        }

        /// <summary>
        /// Adds <paramref name="vector"/> to all rows of matrix <see cref="matrix"/>.
        /// </summary>
        /// <param name="matrix">Source matrix.</param>
        /// <param name="vector">Vector to add.</param>
        /// <param name="destMatrix">Resulting matrix.</param>
        public static void AddRowVector(this Matrix<double> matrix, Vector<double> vector, Matrix<double> destMatrix)
        {
            foreach (var row in matrix.RowEnumerator())
            {
                destMatrix.SetRow(row.Item1, row.Item2.Add(vector));
            }
        }

        /// <summary>
        /// Adds <paramref name="vector"/> to all rows of matrix <see cref="matrix"/>.
        /// </summary>
        /// <param name="matrix">Source matrix.</param>
        /// <param name="vector">Vector to add.</param>
        /// <returns>Resulting matrix.</returns>
        public static Matrix<double> AddRowVector(this Matrix<double> matrix, Vector<double> vector)
        {
            var r = matrix.Clone();
            foreach (var row in matrix.RowEnumerator())
            {
                r.SetRow(row.Item1, row.Item2.Add(vector));
            }

            return r;
        }

        /// <summary>
        /// Adds <paramref name="vector"/> to all columns of matrix <see cref="matrix"/>.
        /// </summary>
        /// <param name="matrix">Source matrix.</param>
        /// <param name="vector">Vector to add.</param>
        /// <param name="destMatrix">Resulting matrix.</param>
        public static void AddColumnVector(this Matrix<double> matrix, Vector<double> vector, Matrix<double> destMatrix)
        {
            foreach (var row in matrix.ColumnEnumerator())
            {
                destMatrix.SetColumn(row.Item1, row.Item2.Add(vector));
            }
        }
        /// <summary>
        /// Multiplies rows of matrix <paramref name="matrix"/> by <paramref name="vector"/> pointwise.
        /// </summary>
        /// <param name="matrix">Source matrix.</param>
        /// <param name="vector">Multiplier.</param>
        /// <returns>Resulting matrix.</returns>
        public static Matrix<double> MulRowVector(this Matrix<double> matrix, Vector<double> vector)
        {
            var r = matrix.Clone();
            foreach (var row in matrix.RowEnumerator())
            {
                r.SetRow(row.Item1, row.Item2.PointwiseMultiply(vector));
            }

            return r;
        }

        /// <summary>
        /// Multiplies columns of matrix <paramref name="matrix"/> by <paramref name="vector"/> pointwise.
        /// </summary>
        /// <param name="matrix">Source matrix.</param>
        /// <param name="vector">Multiplier.</param>
        /// <returns>Resulting matrix.</returns>
        public static Matrix<double> MulColumnVector(this Matrix<double> matrix, Vector<double> vector)
        {
            var r = matrix.Clone();
            foreach (var column in matrix.ColumnEnumerator())
            {
                r.SetColumn(column.Item1, column.Item2.PointwiseMultiply(vector));
            }

            return r;
        }

        /// <summary>
        /// Computes mean of every row.
        /// </summary>
        /// <param name="matrix">Matrix to compute means of rows.</param>
        /// <returns>Vector where every element is a mean of corresponding row in the source matrix.</returns>
        public static Vector MeanOfEveryRow(this Matrix<double> matrix)
        {
            Vector result = DenseVector.Create(matrix.RowCount, i => 0.0);
            foreach (var column in matrix.ColumnEnumerator())
            {
                result.Add(column.Item2, result);
            }

            result.Divide(matrix.ColumnCount, result);

            return result;
        }

        /// <summary>
        /// Computes mean of every column.
        /// </summary>
        /// <param name="matrix">Matrix to compute means of columns.</param>
        /// <returns>Vector where every element is a mean of corresponding column in the source matrix.</returns>
        public static Vector MeanOfEveryColumn(this Matrix<double> matrix)
        {
            Vector result = DenseVector.Create(matrix.ColumnCount, i => 0.0);
            foreach (var row in matrix.RowEnumerator())
            {
                result.Add(row.Item2, result);
            }

            result.Divide(matrix.RowCount, result);

            return result;
        }

        public static Vector SumOfEveryColumn(this Matrix<double> matrix)
        {
            Vector result = DenseVector.Create(matrix.ColumnCount, i => 0.0);
            foreach (var row in matrix.RowEnumerator())
            {
                result.Add(row.Item2, result);
            }

            return result;
        }

        public static Vector SumOfEveryRow(this Matrix<double> matrix)
        {
            Vector result = DenseVector.Create(matrix.RowCount, i => 0.0);
            foreach (var row in matrix.ColumnEnumerator())
            {
                result.Add(row.Item2, result);
            }

            return result;
        }

        /// <summary>
        /// Converts array of double to <see cref="DenseMatrix"/>.
        /// </summary>
        /// <param name="data">Double array.</param>
        /// <returns>Instance of <see cref="DenseMatrix"/>.</returns>
        public static DenseMatrix ToDenseMatrix(this double[,] data)
        {
            return DenseMatrix.OfArray(data);
        }

        /// <summary>
        /// Converts array of double to <see cref="DenseVector"/>.
        /// </summary>
        /// <param name="data">Double array.</param>
        /// <returns>Instance of <see cref="DenseVector"/>.</returns>
        public static DenseVector ToDenseVector(this double[] data)
        {
            return new DenseVector(data);
        }

        /// <summary>
        /// Converts <see cref="Vector"/> to column matrix.
        /// </summary>
        /// <param name="vector">Vector to convert.</param>
        /// <returns>Column matrix.</returns>
        public static Matrix<double> ToColumnMatrix(this Vector<double> vector)
        {
            var m = vector.CreateMatrix(vector.Count, 1);
            m.SetColumn(0, vector);
            return m;
        }

        /// <summary>
        /// Computes frobenious norm for the vector.
        /// </summary>
        /// <param name="vector">Vector to compute norm for.</param>
        /// <returns>Norm for the vector.</returns>
        public static double FrobeniusNorm(this Vector<double> vector)
        {
            return ToColumnMatrix(vector).FrobeniusNorm();
        }

        public static bool AlmostEquals(this Matrix<double> matrix1, Matrix<double> matrix2)
        {
            return (matrix1 - matrix2).FrobeniusNorm() < 1e-10;
        }

        public static bool AlmostEquals(this Vector<double> vector1, Vector<double> vector2)
        {
            return (vector1 - vector2).FrobeniusNorm() < 1e-10;
        }

        public static Matrix VStack(this Matrix<double> upper, Matrix<double> lower)
        {
            var result = new DenseMatrix(upper.RowCount + lower.RowCount, Math.Max(upper.ColumnCount, lower.ColumnCount));
            result.SetSubMatrix(0, upper.RowCount, 0, upper.ColumnCount, upper);
            result.SetSubMatrix(upper.RowCount, lower.RowCount, 0, lower.ColumnCount, lower);
            return result;
        }

        public static Matrix HStack(this Matrix<double> left, Matrix<double> right)
        {
            var result = new DenseMatrix(Math.Max(left.RowCount, right.RowCount), left.ColumnCount + right.ColumnCount);
            result.SetSubMatrix(0, left.RowCount, 0, left.ColumnCount, left);
            result.SetSubMatrix(0, right.RowCount, left.ColumnCount, right.ColumnCount, right);
            return result;
        }

        public static Matrix Outer(this Vector<double> left, Vector<double> right)
        {
            return DenseMatrix.Create(left.Count, right.Count, (i, j) => left[i]*right[j]);
        }

        public static int[] ArgmaxColumns(this Matrix<double> m)
        {
            int[] r = new int[m.RowCount];
            foreach (var row in m.RowEnumerator())
            {
                r[row.Item1] = row.Item2.MaximumIndex();
            }

            return r;
        }
    }
}
