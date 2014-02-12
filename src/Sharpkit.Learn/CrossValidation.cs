// -----------------------------------------------------------------------
// <copyright file="CrossValidation.cs" company="Sharpkit.Learn">
// Author: Alexandre Gramfort alexandre.gramfort@inria.fr,
//         Gael Varoquaux gael.varoquaux@normalesup.org,
//         Olivier Grisel olivier.grisel@ensta.org
// License: BSD 3 clause
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using MathNet.Numerics.LinearAlgebra.Generic;
    using Sharpkit.Learn.Utils;

    /// <summary>
    /// Utilities for cross-validation and performance evaluation.
    /// </summary>
    /// <remarks>
    /// Ported from 
    /// https://github.com/scikit-learn/scikit-learn/tree/687452d9d1b1510a40ba2d2b9fe47c02c0f23ed4/sklearn/cross_validation.py
    /// </remarks>
    public static class CrossValidation
    {
        /// <summary>
        /// <para>
        /// Random permutation cross-validation iterator.
        /// </para>
        /// <para>
        /// Yields indices to split data into training and test sets. 
        /// Note: contrary to other cross-validation strategies, random splits
        /// do not guarantee that all folds will be different, although this is
        /// still very likely for sizeable datasets.
        /// </para>
        /// </summary>
        /// <param name="n">Total number of elements in the dataset.</param>
        /// <param name="nIter">Number of re-shuffling &amp; splitting iterations.</param>
        /// <param name="testSize">should be between 0.0 and 1.0 and represent the
        /// proportion of the dataset to include in the test split. If <c>null</c>,
        /// the value is automatically set to the complement of the train size.</param>
        /// <param name="trainSize">Should be between 0.0 and 1.0 and represent the
        /// proportion of the dataset to include in the train split. If <c>null</c>,
        /// the value is automatically set to the complement of the test size.</param>
        /// <param name="randomState">Pseudo-random number generator state used for random sampling.</param>
        /// <remarks>
        /// See also
        /// Bootstrap: cross-validation using re-sampling with replacement.
        /// </remarks>
        /// <returns>Enumerable of <see cref="ShuffleSplitItem"/>.</returns>
        public static IEnumerable<ShuffleSplitItem> ShuffleSplit(
            int n,
            int nIter = 10,
            double? testSize = null,
            double? trainSize = null,
            Random randomState = null)
        {
            /*Examples
    --------
    >>> from sklearn import cross_validation
    >>> rs = cross_validation.ShuffleSplit(4, n_iter=3,
    ...     test_size=.25, random_state=0)
    >>> len(rs)
    3
    >>> print(rs)
    ... # doctest: +ELLIPSIS
    ShuffleSplit(4, n_iter=3, test_size=0.25, ...)
    >>> for train_index, test_index in rs:
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...
    TRAIN: [3 1 0] TEST: [2]
    TRAIN: [2 1 3] TEST: [0]
    TRAIN: [0 2 1] TEST: [3]


    >>> rs = cross_validation.ShuffleSplit(4, n_iter=3,
    ...     train_size=0.5, test_size=.25, random_state=0)
    >>> for train_index, test_index in rs:
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...
    TRAIN: [3 1] TEST: [2]
    TRAIN: [2 1] TEST: [0]
    TRAIN: [0 2] TEST: [3]
    */
            var result = ValidateShuffleSplit(n, testSize: testSize, trainSize: trainSize);
            int nTrain = result.Item1;
            int nTest = result.Item2;
            var rng = randomState ?? new Random();
            for (int i = 0; i < nIter; i++)
            {
                // random partition
                var permutation = rng.Permutation(n);
                int[] indTest = permutation.Subarray(0, nTest);
                int[] indTrain = permutation.Subarray(nTest, nTrain);
                yield return new ShuffleSplitItem { TrainIndices = indTrain, TestIndices = indTest };
            }
        }

        /// <summary>
        /// Split arrays or matrices into random train and test subsets
        /// </summary>
        /// <param name="arrays">Array of matrices with same row number.</param>
        /// <param name="testSize">Should be between 0.0 and 1.0 and represent the
        /// proportion of the dataset to include in the test split. If <c>null</c>,
        /// the value is automatically set to the complement of the train size.
        /// If train size is also <c> null</c>, test size is set to 0.25.</param>
        /// <param name="trainSize">Should be between 0.0 and 1.0 and represent the
        /// proportion of the dataset to include in the train split. If <c>null</c>,
        /// the value is automatically set to the complement of the test size.</param>
        /// <param name="randomState">Pseudo-random number generator state used for random sampling.</param>
        /// <returns>List containing train-test split of input array.</returns>
        public static IList<Tuple<Matrix<double>, Matrix<double>>> TrainTestSplit(
            Matrix<double>[] arrays,
            double? testSize = null,
            double? trainSize = null,
            Random randomState = null)
        {
            /*
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.cross_validation import train_test_split
    >>> a, b = np.arange(10).reshape((5, 2)), range(5)
    >>> a
    array([[0, 1],
           [2, 3],
           [4, 5],
           [6, 7],
           [8, 9]])
    >>> list(b)
    [0, 1, 2, 3, 4]


    >>> a_train, a_test, b_train, b_test = train_test_split(
    ...     a, b, test_size=0.33, random_state=42)
    ...
    >>> a_train
    array([[4, 5],
           [0, 1],
           [6, 7]])
    >>> b_train
    array([2, 0, 3])
    >>> a_test
    array([[2, 3],
           [8, 9]])
    >>> b_test
    array([1, 4])


  */
            int nArrays = arrays.Length;
            if (nArrays == 0)
            {
                throw new ArgumentException("At least one array required as input");
            }

            if (arrays.Any(a => a.RowCount != arrays[0].RowCount))
            {
                throw new ArgumentException("All arrays must have same row count");
            }

            if (testSize == null && trainSize == null)
            {
                testSize = 0.25;
            }

            int nSamples = arrays[0].RowCount;
            var cv = ShuffleSplit(
                nSamples,
                testSize: testSize,
                trainSize: trainSize,
                randomState: randomState);

            var r = cv.First();
            int[] train = r.TrainIndices;
            int[] test = r.TestIndices;

            return arrays.Select(a => Tuple.Create(a.RowsAt(train), a.RowsAt(test))).ToList();
        }

        private static Tuple<int, int> ValidateShuffleSplit(
           int n,
           double? testSize = null,
           double? trainSize = null)
        {
            if (testSize == null && trainSize == null)
            {
                throw new ArgumentException("test_size and train_size can not both be null");
            }

            int nTest = 0;
            if (testSize != null)
            {
                if (testSize >= 1.0)
                {
                    throw new ArgumentException(
                        "test_size={0} should be smaller than 1.0 or be an integer".Frmt(testSize));
                }

                nTest = (int)Math.Ceiling(testSize.Value * n);
            }

            int nTrain = 0;
            if (trainSize != null)
            {
                if (trainSize >= 1.0)
                {
                    throw new ArgumentException(
                        "train_size={0} should be smaller than 1.0 or be an integer".Frmt(trainSize));
                }
                else if (trainSize + testSize > 1.0)
                {
                    throw new ArgumentException(
                        "The sum of test_size and train_size = {0}, should be smaller than 1.0. Reduce test_size and/or train_size."
                            .Frmt(trainSize + testSize));
                }

                nTrain = (int)Math.Floor(trainSize.Value * n);
            }

            if (trainSize == null)
            {
                nTrain = n - nTest;
            }

            if (testSize == null)
            {
                nTest = n - nTrain;
            }

            if (nTrain + nTest > n)
            {
                throw new ArgumentException(("The sum of train_size and test_size = {0}, " +
                                             "should be smaller than the number of " +
                                             "samples {1}. Reduce test_size and/or " +
                                             "train_size.").Frmt(nTrain + nTest, n));
            }

            return Tuple.Create(nTrain, nTest);
        }

        /// <summary>
        /// Result of <see cref="ShuffleSplit"/>.
        /// </summary>
        public class ShuffleSplitItem
        {
            /// <summary>
            /// Gets or sets indices of samples to be used for training.
            /// </summary>
            public int[] TrainIndices { get; set; }

            /// <summary>
            /// Gets or sets indices of samples to be used for testing.
            /// </summary>
            public int[] TestIndices { get; set; }
        }
    }
}
