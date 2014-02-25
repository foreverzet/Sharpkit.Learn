// -----------------------------------------------------------------------
// <copyright file="Multiclass.cs" company="Sharpkit.Learn">
//  Copyright (c) 2013 Sergey Zyuzin
//  License: BSD 3 clause
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn.Utils
{
    using System;
    using System.Linq;

    /// <summary>
    /// TODO: Update summary.
    /// </summary>
    internal static class Multiclass
    {

        /*
         *     """Extract an ordered array of unique labels

    Parameters
    ----------
    lists_of_labels : list of labels,
        The supported "list of labels" are:
            - a list / tuple / numpy array of int
            - a list of lists / tuples of int;
            - a binary indicator matrix (2D numpy array)

    Returns
    -------
    out : numpy array of shape [n_unique_labels]
        An ordered array of unique labels.

    Examples
    --------
    >>> from sklearn.utils.multiclass import unique_labels
    >>> unique_labels([3, 5, 5, 5, 7, 7])
    array([3, 5, 7])
    >>> unique_labels([1, 2, 3, 4], [2, 2, 3, 4])
    array([1, 2, 3, 4])
    >>> unique_labels([1, 2, 10], [5, 11])
    array([ 1,  2,  5, 10, 11])
    >>> unique_labels(np.array([[0.0, 1.0], [1.0, 1.0]]), np.zeros((2, 2)))
    array([0, 1])
    >>> unique_labels([(1, 2), (3,)], [(1, 2), tuple()])
    array([1, 2, 3])

    """

         */
        public static int[] unique_labels(params int[][] lists_of_labels)
        {

            if (lists_of_labels == null)
                throw new ArgumentException("No list of labels has been passed.");

            return lists_of_labels.Aggregate(
                new int[0],
                (l, r)=> l.Concat(r).ToArray())
                .Distinct().OrderBy(v=>v).ToArray();
        }

        /*
         * """Determine the type of data indicated by target `y`

    Parameters
    ----------
    y : array-like

    Returns
    -------
    target_type : string
        One of:
        * 'continuous': `y` is an array-like of floats that are not all
          integers, and is 1d or a column vector.
        * 'continuous-multioutput': `y` is a 2d array of floats that are
          not all integers, and both dimensions are of size > 1.
        * 'binary': `y` contains <= 2 discrete values and is 1d or a column
          vector.
        * 'multiclass': `y` contains more than two discrete values, is not a
          sequence of sequences, and is 1d or a column vector.
        * 'mutliclass-multioutput': `y` is a 2d array that contains more
          than two discrete values, is not a sequence of sequences, and both
          dimensions are of size > 1.
        * 'multilabel-sequences': `y` is a sequence of sequences, a 1d
          array-like of objects that are sequences of labels.
        * 'multilabel-indicator': `y` is a label indicator matrix, an array
          of two dimensions with at least two columns, and at most 2 unique
          values.
        * 'unknown': `y` is array-like but none of the above, such as a 3d
          array, or an array of non-sequence objects.

    Examples
    --------
    >>> import numpy as np
    >>> type_of_target([0.1, 0.6])
    'continuous'
    >>> type_of_target([1, -1, -1, 1])
    'binary'
    >>> type_of_target(['a', 'b', 'a'])
    'binary'
    >>> type_of_target([1, 0, 2])
    'multiclass'
    >>> type_of_target(['a', 'b', 'c'])
    'multiclass'
    >>> type_of_target(np.array([[1, 2], [3, 1]]))
    'multiclass-multioutput'
    >>> type_of_target(np.array([[1.5, 2.0], [3.0, 1.6]]))
    'continuous-multioutput'
    >>> type_of_target([['a', 'b'], ['c'], []])
    'multilabel-sequences'
    >>> type_of_target([[]])
    'multilabel-sequences'
    >>> type_of_target(np.array([[0, 1], [1, 1]]))
    'multilabel-indicator'
    """
         */

        public static string type_of_target<T>(T[] y)
        {
            // XXX: is there a way to duck-type this condition?

            // check float and contains non-integer float values:
            if (typeof (T) == typeof (float) || typeof (T) == typeof (double))
                return "continous";

            if (y.Distinct().Count() <= 2)
                return "binary";

            return "multiclass";
        }
    }
}
