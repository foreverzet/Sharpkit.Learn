// -----------------------------------------------------------------------
// <copyright file="CrossValidationTest.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn.Test
{
    using System;

    /// <summary>
    /// TODO: Update summary.
    /// </summary>
    /// <remarks>
    /// https://github.com/scikit-learn/scikit-learn/tree/c63bb62f6fb430046495f426e39572b2c26913f7/sklearn/tests/test_cross_validation.py
    /// </remarks>
    public class CrossValidationTest
    {
        def test_train_test_split_errors():
    assert_raises(ValueError, cval.train_test_split)
    assert_raises(ValueError, cval.train_test_split, range(3), train_size=1.1)
    assert_raises(ValueError, cval.train_test_split, range(3), test_size=0.6,
                  train_size=0.6)
    assert_raises(ValueError, cval.train_test_split, range(3),
                  test_size=np.float32(0.6), train_size=np.float32(0.6))
    assert_raises(ValueError, cval.train_test_split, range(3),
                  test_size="wrong_type")
    assert_raises(ValueError, cval.train_test_split, range(3), test_size=2,
                  train_size=4)
    assert_raises(TypeError, cval.train_test_split, range(3),
                  some_argument=1.1)
    assert_raises(ValueError, cval.train_test_split, range(3), range(42))




def test_train_test_split():
    X = np.arange(100).reshape((10, 10))
    X_s = coo_matrix(X)
    y = range(10)
    split = cval.train_test_split(X, X_s, y)
    X_train, X_test, X_s_train, X_s_test, y_train, y_test = split
    assert_array_equal(X_train, X_s_train.toarray())
    assert_array_equal(X_test, X_s_test.toarray())
    assert_array_equal(X_train[:, 0], y_train * 10)
    assert_array_equal(X_test[:, 0], y_test * 10)
    split = cval.train_test_split(X, y, test_size=None, train_size=.5)
    X_train, X_test, y_train, y_test = split
    assert_equal(len(y_test), len(y_train))

    }
}
