// -----------------------------------------------------------------------
// <copyright file="CrossValidation.cs" company="Sharpkit.Learn">
// Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>,
//         Gael Varoquaux <gael.varoquaux@normalesup.org>,
//         Olivier Grisel <olivier.grisel@ensta.org>
// License: BSD 3 clause
// </copyright>
// -----------------------------------------------------------------------

using System.Collections.Generic;
using System.Linq;
using Sharpkit.Learn.Utils;

namespace Sharpkit.Learn
{
    using System;
    using MathNet.Numerics.LinearAlgebra.Generic;

    /// <summary>
    /// TODO: Update summary.
    /// </summary>
    /// <remarks>
    /// https://github.com/scikit-learn/scikit-learn/tree/687452d9d1b1510a40ba2d2b9fe47c02c0f23ed4/sklearn/cross_validation.py
    /// </remarks>
    public static class CrossValidation
    {
        private static Tuple<int, int> _validate_shuffle_split(int n, double? test_size = null,
                                                               double? train_size = null)
        {
            if (test_size == null && train_size == null)
            {
                throw new ArgumentException("test_size and train_size can not both be null");
            }

            int n_test = 0;
            if (test_size != null)
            {
                if (test_size >= 1.0)
                {
                    throw new ArgumentException(
                        "test_size={0} should be smaller than 1.0 or be an integer".Frmt(test_size));
                }

                n_test = (int)Math.Ceiling(test_size.Value*n);
            }

            int n_train = 0;
            if (train_size != null)
            {
                if (train_size >= 1.0)
                {
                    throw new ArgumentException(
                        "train_size={0} should be smaller than 1.0 or be an integer".Frmt(train_size));
                }
                else if (train_size + test_size > 1.0)
                {
                    throw new ArgumentException(
                        "The sum of test_size and train_size = {0}, should be smaller than 1.0. Reduce test_size and/or train_size."
                            .Frmt(train_size + test_size));
                }

                n_train = (int)Math.Floor(train_size.Value*n);
            }


            if (train_size == null)
            {
                n_train = n - n_test;
            }

            if (test_size == null)
            {
                n_test = n - n_train;
            }


            if (n_train + n_test > n)
            {
                throw new ArgumentException(("The sum of train_size and test_size = {0}, " +
                                             "should be smaller than the number of " +
                                             "samples {1}. Reduce test_size and/or " +
                                             "train_size.").Frmt(n_train + n_test, n));
            }

            return Tuple.Create(n_train, n_test);
        }

        public class ShuffleSplitItem
        {
            public int[] TrainIndices;
            public int[] TestIndices;
        }

        /// <summary>
        /// Random permutation cross-validation iterator.
        ///
        /// Yields indices to split data into training and test sets. 
        /// Note: contrary to other cross-validation strategies, random splits
        /// do not guarantee that all folds will be different, although this is
        /// still very likely for sizeable datasets.
        /// </summary>
        public static IEnumerable<ShuffleSplitItem> ShuffleSplit(int n, int n_iter = 10, double? testSize = null,
                                                                 double? trainSize = null, Random random_state = null)
        {
            /*

    Parameters
    ----------
    n : int
        Total number of elements in the dataset.


    n_iter : int (default 10)
        Number of re-shuffling & splitting iterations.


    test_size : float (default 0.1), int, or None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the test split. If
        int, represents the absolute number of test samples. If None,
        the value is automatically set to the complement of the train size.


    train_size : float, int, or None (default is None)
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.


    random_state : int or RandomState
        Pseudo-random number generator state used for random sampling.


    Examples
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


    See also
    --------
    Bootstrap: cross-validation using re-sampling with replacement.
    */
            var result = _validate_shuffle_split(n, test_size: testSize, train_size: trainSize);
            int n_train = result.Item1;
            int n_test = result.Item2;
            var rng = random_state ?? new Random();
            for (int i = 0; i < n_iter; i++)
            {
                // random partition
                var permutation = rng.Permutation(n);
                int[] ind_test = permutation.Subarray(0, n_test);
                int[] ind_train = permutation.Subarray(n_test, n_train);
                yield return new ShuffleSplitItem() {TrainIndices = ind_train, TestIndices = ind_test};
            }
        }


        /// <summary>
        /// Split arrays or matrices into random train and test subsets
        /// </summary>
        /// <param name="?"></param>
        public static IList<Tuple<Matrix<double>, Matrix<double>>> train_test_split(Matrix<double>[] arrays,
                                                                                    double? testSize = null,
                                                                                    double? trainSize = null,
                                                                                    Random random_state = null)
        {
            /*
    Quick utility that wraps calls to ``check_arrays`` and
    ``next(iter(ShuffleSplit(n_samples)))`` and application to input
    data into a single call for splitting (and optionally subsampling)
    data in a oneliner.


    Parameters
    ----------
    *arrays : sequence of arrays or scipy.sparse matrices with same shape[0]
        Python lists or tuples occurring in arrays are converted to 1D numpy
        arrays.


    test_size : float, int, or None (default is None)
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the test split. If
        int, represents the absolute number of test samples. If None,
        the value is automatically set to the complement of the train size.
        If train size is also None, test size is set to 0.25.


    train_size : float, int, or None (default is None)
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.


    random_state : int or RandomState
        Pseudo-random number generator state used for random sampling.


    dtype : a numpy dtype instance, None by default
        Enforce a specific dtype.


    Returns
    -------
    splitting : list of arrays, length=2 * len(arrays)
        List containing train-test split of input array.


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
            int n_arrays = arrays.Length;
            if (n_arrays == 0)
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

            int n_samples = arrays[0].RowCount;
            var cv = ShuffleSplit(n_samples, testSize: testSize,
                                  trainSize: trainSize,
                                  random_state: random_state);


            var r = cv.First();
            int[] train = r.TrainIndices;
            int[] test = r.TestIndices;

            return arrays.Select(a => Tuple.Create(a.RowsAt(train), a.RowsAt(test))).ToList();
        }
    }
}
