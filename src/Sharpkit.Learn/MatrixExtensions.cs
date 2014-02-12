// -----------------------------------------------------------------------
// <copyright file="MatrixExtensions.cs" company="Sharpkit.Learn">
//  Copyright Sergey Zyuzin 2014.
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using MathNet.Numerics;
    using MathNet.Numerics.LinearAlgebra.Double;
    using MathNet.Numerics.LinearAlgebra.Generic;
    using MathNet.Numerics.LinearAlgebra.Generic.Factorization;

    /// <summary>
    /// <see cref="Matrix"/> and <see cref="Vector"/> extention methods.
    /// </summary>
    internal static class MatrixExtensions
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
            return vector.Map(Math.Sqrt);
        }

        /// <summary>
        /// Returns matrix where every element is Log of corresponding
        /// element in the original matrix.
        /// </summary>
        /// <param name="matrix">Matrix which elements will be transformed.</param>
        /// <returns>Resulting matrix.</returns>
        public static Matrix<double> Log(this Matrix<double> matrix)
        {
            return matrix.Map(Math.Log);
        }

        /// <summary>
        /// Returns vector where every element is Log of corresponding
        /// element in the original vector.
        /// </summary>
        /// <param name="vector">Vector which elements will be transformed.</param>
        /// <returns>Resulting vector.</returns>
        public static Vector<double> Log(this Vector<double> vector)
        {
            return vector.Map(Math.Log);
        }

        /// <summary>
        /// Returns matrix where every element is exponent of corresponding
        /// element in the original matrix.
        /// </summary>
        /// <param name="matrix">Matrix which elements will be transformed.</param>
        /// <returns>Resulting matrix.</returns>
        public static Matrix<double> Exp(this Matrix<double> matrix)
        {
            return matrix.Map(Math.Exp);
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
        /// Returns vector where every element is abs of corresponding
        /// element in the original vector.
        /// </summary>
        /// <param name="vector">Vector which elements will be squared.</param>
        /// <returns>Vector with squared elements.</returns>
        public static Vector<double> Abs(this Vector<double> vector)
        {
            return vector.Map(Math.Abs);
        }

        /// <summary>
        /// Returns vector where every element is transformed by f of corresponding
        /// element in the original vector.
        /// </summary>
        /// <param name="vector">Vector which elements will be squared.</param>
        /// <returns>Vector with squared elements.</returns>
        public static Vector<double> Map(this Vector<double> vector, Func<double, double> f)
        {
            var newVector = vector.Clone();
            newVector.MapInplace(f);
            return newVector;
        }

        /// <summary>
        /// Returns vector where every element is transformed by f of corresponding
        /// element in the original vector.
        /// </summary>
        /// <param name="vector">Vector which elements will be squared.</param>
        /// <returns>Vector with squared elements.</returns>
        public static Matrix<double> Map(this Matrix<double> matrix, Func<double, double> f)
        {
            var newMatrix = matrix.Clone();
            newMatrix.MapInplace(f);
            return newMatrix;
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
        /// Subtracts <paramref name="vector"/> from all elements of <paramref name="matrix"/>
        /// and places result into <paramref name="destMatrix"/>.
        /// </summary>
        /// <param name="matrix">Source matrix.</param>
        /// <param name="vector">Vector to subtract.</param>
        /// <param name="destMatrix">Resulting matrix.</param>
        public static void SubtractRowVector(
            this Matrix<double> matrix,
            Vector<double> vector,
            Matrix<double> destMatrix)
        {
            foreach (var row in matrix.RowEnumerator())
            {
                destMatrix.SetRow(row.Item1, row.Item2.Subtract(vector));
            }
        }

        /// <summary>
        /// Subtracts <paramref name="vector"/> from all rows elements of <paramref name="matrix"/>.
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
        /// Divides rows of matrix <paramref name="matrix"/> by <paramref name="vector"/> pointwise
        /// </summary>
        /// <param name="matrix">Source matrix.</param>
        /// <param name="vector">Divider.</param>
        public static Matrix<double> DivRowVector(this Matrix<double> matrix, Vector<double> vector)
        {
            Matrix<double> destMatrix = matrix.Clone();
            DivRowVector(matrix, vector, destMatrix);
            return destMatrix;
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
        /// Divides all columns of matrix <paramref name="matrix"/> by <paramref name="vector"/> pointwise.
        /// </summary>
        /// <param name="matrix">Source matrix.</param>
        /// <param name="vector">Divider.</param>
        public static Matrix<double> DivColumnVector(this Matrix<double> matrix, Vector<double> vector)
        {
            Matrix<double> destMatrix = matrix.Clone();
            foreach (var column in matrix.ColumnEnumerator())
            {
                destMatrix.SetColumn(column.Item1, column.Item2.PointwiseDivide(vector));
            }

            return destMatrix;
        }

        /// <summary>
        /// Adds <paramref name="vector"/> to all rows of <paramref name="matrix"/>.
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
        /// Adds <paramref name="vector"/> to all rows of <paramref name="matrix"/>.
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
        /// Adds <paramref name="vector"/> to all columns of <paramref name="matrix"/>.
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
        /// Adds <paramref name="vector"/> to all columns of <paramref name="matrix"/>.
        /// </summary>
        /// <param name="matrix">Source matrix.</param>
        /// <param name="vector">Vector to add.</param>
        /// <returns>Resulting matrix.</returns>
        public static Matrix<double> AddColumnVector(this Matrix<double> matrix, Vector<double> vector)
        {
            var result = matrix.Clone();
            AddColumnVector(matrix, vector, result);
            return result;
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

        public static double Mean(this Matrix<double> matrix)
        {
            double result = 0;
            foreach (var column in matrix.ColumnEnumerator())
            {
                result += column.Item2.Sum();
            }

            result /= matrix.ColumnCount * matrix.RowCount;

            return result;
        }

        public static T[] Subarray<T>(this T[] array, int start, int length)
        {
            T[] result = new T[length];
            Array.Copy(array, start, result, 0, length);

            return result;
        }

        public static T[,] Subarray<T>(this T[,] array, int xstart, int xlength, int ystart, int ylength)
        {
            T[,] result = new T[xlength, ylength];
            for (int x = 0; x < xlength; x++)
                for (int y = 0; y < ylength; y++)
                    result[x, y] = array[xstart + x, ystart + y];

            return result;
        }

        public static Tuple<int, int> Shape<T>(this T[,] array)
        {
            return Tuple.Create(array.GetLength(0), array.GetLength(1));
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
            return data == null ? null : new DenseVector(data);
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
        /// Converts <see cref="IEnumerable"/> to column matrix.
        /// </summary>
        /// <param name="vector">Enumerable to convert.</param>
        /// <returns>Column matrix.</returns>
        public static Matrix<double> ToColumnMatrix(this IEnumerable<double> vector)
        {
            return DenseVector.OfEnumerable(vector).ToColumnMatrix();
        }

        /// <summary>
        /// Converts <see cref="IEnumerable"/> to column matrix.
        /// </summary>
        /// <param name="vector">Enumerable to convert.</param>
        /// <returns>Column matrix.</returns>
        public static Matrix<double> ToColumnMatrix(this IEnumerable<int> vector)
        {
            return DenseVector.OfEnumerable(vector.Select(v => (double)v)).ToColumnMatrix();
        }

        /// <summary>
        /// Converts <see cref="IEnumerable{int}"/> to Vector.
        /// </summary>
        /// <param name="vector">Enumerable to convert.</param>
        /// <returns>Vector.</returns>
        public static Vector<double> ToVector(this IEnumerable<int> vector)
        {
            return DenseVector.OfEnumerable(vector.Select(v => (double)v));
        }

        /// <summary>
        /// Converts <see cref="IEnumerable{int}"/> to Vector.
        /// </summary>
        /// <param name="vector">Enumerable to convert.</param>
        /// <returns>Vector.</returns>
        public static Vector<double> ToVector(this IEnumerable<double> vector)
        {
            return DenseVector.OfEnumerable(vector);
        }

        // todo this shall return Matrix<int>
        public static Matrix<double> ArgsortColumns(this Matrix<double> matrix)
        {
            Matrix<double> result = DenseMatrix.Create(matrix.RowCount, matrix.ColumnCount, (x, y) => 0);
            foreach (var column in matrix.ColumnEnumerator())
            {
                var newColumn = column.Item2
                    .Select((v, i) => Tuple.Create(v, i))
                    .OrderBy(t => t.Item1)
                    .Select(t => t.Item2).Select(v => (double)v).ToArray();

                result.SetColumn(column.Item1, newColumn);
            }

            return result;
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

        public static bool AlmostEquals(this Matrix<double> matrix1, Matrix<double> matrix2, double epsilon = 1e-10)
        {
            return (matrix1 - matrix2).FrobeniusNorm() < epsilon;
        }

        public static bool AlmostEquals(this Vector<double> vector1, Vector<double> vector2, double epsilon = 1e-10)
        {
            return (vector1 - vector2).FrobeniusNorm() < 1e-10;
        }

        public static bool AlmostEquals(this double[] vector1, double[] vector2, double epsilon = 1e-10)
        {
            return (new DenseVector(vector1) - new DenseVector(vector2)).FrobeniusNorm() < epsilon;
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
            return DenseMatrix.Create(left.Count, right.Count, (i, j) => left[i] * right[j]);
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

        public static Matrix<double> SvdSolve(this Matrix<double> x, Matrix<double> y)
        {
            var svd = x.Svd(true);
            // todo: math.net svd cannot solve underdetermined systems.
            // report bug. For now workaraound with pseudoinverse.
            if (svd.Rank >= x.ColumnCount)
            {
                return svd.Solve(y);
            }
            else
            {
                return PseudoInverse(svd) * y;
            }
        }

        public static Matrix<double> RowsAt(this Matrix<double> x, IEnumerable<int> indices)
        {
            int numberOfRows = indices.Count();
            var result = x.CreateMatrix(numberOfRows, x.ColumnCount);
            int i = 0;
            foreach (var ind in indices)
            {
                result.SetRow(i, x.Row(ind));
                i++;
            }
            
            return result;
        }

        public static Matrix<double> ColumnsAt(this Matrix<double> x, IList<int> indices)
        {
            var result = x.CreateMatrix(x.RowCount, indices.Count);
            for (int i = 0; i < indices.Count; i++)
            {
                result.SetColumn(i, x.Column(indices[i]));
            }

            return result;
        }

        public static T[] ElementsAt<T>(this T[] x, IEnumerable<int> indices)
        {
            var count = indices.Count();
            var result = new T[count];
            int i = 0;
            foreach (var ind in indices)
            {
                result[i] = x[ind];
                i++;
            }

            return result;
        }

        public static int[] Indices<T>(this T[] x, Func<T, bool> func)
        {
            return x.Select((v, i) => Tuple.Create(v, i)).Where(t => func(t.Item1)).Select(t => t.Item2).ToArray();
        }

        public static IEnumerable<double> Diff(this IEnumerable<double> en)
        {
            var list = en.ToList();
            return list.Take(list.Count - 1).Zip(list.Skip(1), Tuple.Create).Select(t => t.Item2 - t.Item1);
        }

        private static Matrix<double> PseudoInverse(Svd<double> svd)
        {
            Matrix<double> W = svd.W();
            Vector<double> s = svd.S();

            // The first element of W has the maximum value. 
            double tolerance = Precision.EpsilonOf(2) * Math.Max(svd.U().RowCount, svd.VT().ColumnCount) * W[0, 0];

            for (int i = 0; i < s.Count; i++)
            {
                if (s[i] < tolerance)
                    s[i] = 0;
                else
                    s[i] = 1 / s[i];
            }

            W.SetDiagonal(s);

            // (U * W * VT)T is equivalent with V * WT * UT 
            return (svd.U() * W * svd.VT()).Transpose();
        }

        public static string ToPythonArray(this Matrix<double> m)
        {
            return "[" + string.Join(",", m.RowEnumerator().Select(v => v.Item2.ToPythonArray())) + "]";
        }

        public static string ToPythonArray(this Vector<double> m)
        {
            return "[" + string.Join(",", m) + "]";
        }
    }
}
