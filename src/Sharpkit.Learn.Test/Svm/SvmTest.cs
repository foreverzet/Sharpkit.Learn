// -----------------------------------------------------------------------
// <copyright file="SvmTest.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

using Sharpkit.Learn.Datasets;

namespace Sharpkit.Learn.Test.Svm
{
    using System;
    using System.Linq;
    using MathNet.Numerics.LinearAlgebra.Double;
    using Microsoft.VisualStudio.TestTools.UnitTesting;
    using Sharpkit.Learn.Svm;

    /// <summary>
    /// TODO: Update summary.
    /// </summary>
    [TestClass]
    public class SvmTest
    {
        private DenseMatrix X = DenseMatrix.OfArray(new double[,] {{-2, -1}, {-1, -1}, {-1, -2}, {1, 1}, {1, 2}, {2, 1}});
        private int[] Y = new [] {1, 1, 1, 2, 2, 2};
        private DenseMatrix T = DenseMatrix.OfArray(new double[,] {{-1, -1}, {2, 2}, {3, 2}});
        private int[] true_result = new[] {1, 2, 2};

        private DenseMatrix X2 =
            DenseMatrix.OfArray(new double[,] {{0, 0, 0}, {1, 1, 1}, {2, 0, 0,}, {0, 0, 2}, {3, 3, 3}});

        private int[] Y2 = new[] {1, 2, 2, 2, 3};
        private DenseMatrix T2 = DenseMatrix.OfArray(new double[,] {{-1, -1, -1}, {1, 1, 1}, {2, 2, 2}});
        private int[] true_result2 = new[] {1, 2, 3};

        private IrisDataset iris = IrisDataset.Load();

        /// <summary>
        /// Test parameters on classes that make use of libsvm.
        /// </summary>
        [TestMethod]
        public void test_libsvm_parameters()
        {
            var clf = new Svc<int>(kernel: Kernel.FromSparseKernel(SparseKernel.Linear));
            clf.Fit(X, Y);
            Assert.IsTrue(clf.DualCoef.AlmostEquals(DenseMatrix.OfArray(new[,] {{0.25, -.25}})));
            //Assert.IsTrue(clf.Support.SequenceEqual(new[] {1, 3}));
            Assert.IsTrue(
                clf.SupportVectors.AlmostEquals(DenseMatrix.OfRows(2, X.ColumnCount, new[] {X.Row(1), X.Row(3)})));
            Assert.IsTrue(clf.Intercept.SequenceEqual(new[]{0.0}));
            Assert.IsTrue(clf.Predict(X).SequenceEqual(Y));
        }

        /// <summary>
        /// Check that sparse SVC gives the same result as SVC"
        /// </summary>
        [TestMethod]
        public void test_svc()
        {
            var clf = new Svc<int>(kernel : Kernel.FromSparseKernel(SparseKernel.Linear), probability : true);
            clf.Fit(X, Y);

            var sp_clf = new Svc<int>(kernel : Kernel.FromSparseKernel(SparseKernel.Linear), probability : true);
            sp_clf.Fit(SparseMatrix.OfMatrix(X), Y);

            Assert.IsTrue(sp_clf.Predict(T).SequenceEqual(true_result));

            Assert.IsTrue(sp_clf.SupportVectors is SparseMatrix);
            Assert.IsTrue(clf.SupportVectors.AlmostEquals(sp_clf.SupportVectors));

            Assert.IsTrue(sp_clf.DualCoef is SparseMatrix);
            Assert.IsTrue(clf.DualCoef.AlmostEquals(sp_clf.DualCoef));

            Assert.IsTrue(sp_clf.Coef is SparseMatrix);
            Assert.IsTrue(clf.Coef.AlmostEquals(sp_clf.Coef));
            Assert.IsTrue(clf.Support.SequenceEqual(sp_clf.Support));
            Assert.IsTrue(clf.Predict(T).SequenceEqual(sp_clf.Predict(T)));

            // refit with a different dataset

            clf.Fit(X2, Y2);
            sp_clf.Fit(SparseMatrix.OfMatrix(X2), Y2);
            Assert.IsTrue(clf.SupportVectors.AlmostEquals(sp_clf.SupportVectors));
            Assert.IsTrue(clf.DualCoef.AlmostEquals(sp_clf.DualCoef));
            Assert.IsTrue(clf.Coef.AlmostEquals(sp_clf.Coef));
            Assert.IsTrue(clf.Support.SequenceEqual(sp_clf.Support));
            Assert.IsTrue(clf.Predict(T2).SequenceEqual(sp_clf.Predict(T2)));
            Assert.IsTrue(clf.PredictProba(T2).AlmostEquals(sp_clf.PredictProba(T2)));
        }


        /// <summary>
       /// Check consistency on dataset iris.
       /// </summary>
        [TestMethod]
        public void test_libsvm_iris()
        {
            // shuffle the dataset so that labels are not ordered
            foreach (var k in new[]{SparseKernel.Linear, SparseKernel.Rbf})
            {
                var clf = new Svc<int>(kernel : Kernel.FromSparseKernel(k));
                clf.Fit(iris.Data, iris.Target);
                var pred = clf.Predict(iris.Data);
                var matchingN = pred.Zip(iris.Target, Tuple.Create).Where(t => t.Item1 == t.Item2).Count();
                Assert.IsTrue(1.0 * matchingN / pred.Length > 0.9);
                Assert.IsTrue(clf.Classes.SequenceEqual(clf.Classes.OrderBy(v => v)));
            }

            /*
            # check also the low-level API
            model = svm.libsvm.fit(iris.data, iris.target.astype(np.float64))
            pred = svm.libsvm.predict(iris.data, *model)
            assert_greater(np.mean(pred == iris.target), .95)

            model = svm.libsvm.fit(iris.data, iris.target.astype(np.float64),
                          kernel='linear')
            pred = svm.libsvm.predict(iris.data, *model, kernel='linear')
            assert_greater(np.mean(pred == iris.target), .95)
        
            pred = svm.libsvm.cross_validation(iris.data,
                                       iris.target.astype(np.float64), 5,
                                       kernel='linear')
            assert_greater(np.mean(pred == iris.target), .95)
             * */
        }

        /// <summary>
        /// Test whether SVCs work on a single sample given as a 1-d array.
        /// </summary>
        [TestMethod]
        public void test_single_sample_1d()
        {
            var clf = new Svc<int>();
            clf.Fit(X, Y);
            var p = clf.Predict(X.Row(0).ToRowMatrix());
            Assert.AreEqual(Y[0], p[0]);

            //todo:
            //clf = svm.LinearSVC(random_state=0).fit(X, Y)
            //clf.predict(X[0])
        }

        /// <summary>
        /// SVC with a precomputed kernel.
        ///
        /// We test it with a toy dataset and with iris.
        /// </summary>
        [TestMethod]
        public void test_precomputed()
        {
            var clf = new Svc<int>(kernel: Kernel.FromSparseKernel(SparseKernel.Precomputed));
            // Gram matrix for train data (square matrix)
            // (we use just a linear kernel)
            var K = X*(X.Transpose());
            clf.Fit(K, Y);
            // Gram matrix for test data (rectangular matrix)
            var KT = T*X.Transpose();
            var pred = clf.Predict(KT);
            try
            {
                clf.Predict(KT.Transpose());
                Assert.Fail();
            }
            catch (ArgumentException)
            {
            }

            Assert.IsTrue(clf.DualCoef.AlmostEquals(DenseMatrix.OfArray(new[,] {{0.25, -.25}})));
            Assert.IsTrue(clf.Support.SequenceEqual(new[] {1, 3}));
            Assert.IsTrue(clf.Intercept.SequenceEqual(new[]{0.0}));
            Assert.IsTrue(pred.SequenceEqual(true_result));

            // Gram matrix for test data but compute KT[i,j]
            // for support vectors j only.
            KT = KT.CreateMatrix(KT.RowCount, KT.ColumnCount);
            for (int i=0; i< T.RowCount; i++)
            {
                foreach (var j in clf.Support)
                {
                    KT[i, j] = T.Row(i)*X.Row(j);
                }
            }

            pred = clf.Predict(KT);
            Assert.IsTrue(pred.SequenceEqual(true_result));

            // same as before, but using a callable function instead of the kernel
            // matrix. kernel is just a linear kernel

            clf = new Svc<int>(kernel: Kernel.FromFunction((x, y) => x*y.Transpose()));
            clf.Fit(X, Y);
            pred = clf.Predict(T);

            Assert.IsTrue(clf.DualCoef.AlmostEquals(DenseMatrix.OfArray(new[,] {{0.25, -.25}})));
            Assert.IsTrue(clf.Support.SequenceEqual(new[] {1, 3}));
            Assert.IsTrue(clf.Intercept.SequenceEqual(new[]{0.0}));
            Assert.IsTrue(pred.SequenceEqual(true_result));

            // test a precomputed kernel with the iris dataset
            // and check parameters against a linear SVC
            clf = new Svc<int>(kernel: Kernel.FromSparseKernel(SparseKernel.Precomputed));
            var clf2 = new Svc<int>(kernel : Kernel.FromSparseKernel(SparseKernel.Linear));
            K = iris.Data*iris.Data.Transpose();
            clf.Fit(K, iris.Target);
            clf2.Fit(iris.Data, iris.Target);
            pred = clf.Predict(K);
            Assert.IsTrue(clf.Support.SequenceEqual(clf2.Support));
            Assert.IsTrue(clf.DualCoef.AlmostEquals(clf2.DualCoef));
            Assert.IsTrue(clf.Intercept.AlmostEquals(clf2.Intercept));
            
            var matchingN = pred.Zip(iris.Target, Tuple.Create).Where(t => t.Item1 == t.Item2).Count();
            Assert.IsTrue(1.0 * matchingN / pred.Length > 0.99);

            // Gram matrix for test data but compute KT[i,j]
            // for support vectors j only.
            K = K.CreateMatrix(K.RowCount, K.ColumnCount);
            for (int i=0; i<iris.Data.RowCount; i++)
            {
                foreach (var j in clf.Support)
                    K[i, j] = iris.Data.Row(i)*iris.Data.Row(j);
            }

            pred = clf.Predict(K);
            matchingN = pred.Zip(iris.Target, Tuple.Create).Where(t => t.Item1 == t.Item2).Count();
            Assert.IsTrue(1.0 * matchingN / pred.Length > 0.99);

            clf = new Svc<int>(kernel: Kernel.FromFunction((x, y) => x*y.Transpose()));
            clf.Fit(iris.Data, iris.Target);
            matchingN = pred.Zip(iris.Target, Tuple.Create).Where(t => t.Item1 == t.Item2).Count();
            Assert.IsTrue(1.0 * matchingN / pred.Length > 0.99);
        }

        /*
         * 
def test_tweak_params():
    """
    Make sure some tweaking of parameters works.

    We change clf.dual_coef_ at run time and expect .predict() to change
    accordingly. Notice that this is not trivial since it involves a lot
    of C/Python copying in the libsvm bindings.

    The success of this test ensures that the mapping between libsvm and
    the python classifier is complete.
    """
    clf = svm.SVC(kernel='linear', C=1.0)
    clf.fit(X, Y)
    assert_array_equal(clf.dual_coef_, [[.25, -.25]])
    assert_array_equal(clf.predict([[-.1, -.1]]), [1])
    clf.dual_coef_ = np.array([[.0, 1.]])
    assert_array_equal(clf.predict([[-.1, -.1]]), [2])
    }
         * 
         * def test_probability():
    """
    Predict probabilities using SVC

    This uses cross validation, so we use a slightly bigger testing set.
    """

    for clf in (svm.SVC(probability=True, C=1.0),
                svm.NuSVC(probability=True)):

        clf.fit(iris.data, iris.target)

        prob_predict = clf.predict_proba(iris.data)
        assert_array_almost_equal(
            np.sum(prob_predict, 1), np.ones(iris.data.shape[0]))
        assert_true(np.mean(np.argmax(prob_predict, 1)
                    == clf.predict(iris.data)) > 0.9)

        assert_almost_equal(clf.predict_proba(iris.data),
                            np.exp(clf.predict_log_proba(iris.data)), 8)


def test_decision_function():
    """
    Test decision_function

    Sanity check, test that decision_function implemented in python
    returns the same as the one in libsvm

    """
    # multi class:
    clf = svm.SVC(kernel='linear', C=0.1).fit(iris.data, iris.target)

    dec = np.dot(iris.data, clf.coef_.T) + clf.intercept_

    assert_array_almost_equal(dec, clf.decision_function(iris.data))

    # binary:
    clf.fit(X, Y)
    dec = np.dot(X, clf.coef_.T) + clf.intercept_
    prediction = clf.predict(X)
    assert_array_almost_equal(dec, clf.decision_function(X))
    assert_array_almost_equal(
        prediction,
        clf.classes_[(clf.decision_function(X) > 0).astype(np.int).ravel()])
    expected = np.array([[-1.], [-0.66], [-1.], [0.66], [1.], [1.]])
    assert_array_almost_equal(clf.decision_function(X), expected, 2)


def test_weight():
    """
    Test class weights
    """
    clf = svm.SVC(class_weight={1: 0.1})
    # we give a small weights to class 1
    clf.fit(X, Y)
    # so all predicted values belong to class 2
    assert_array_almost_equal(clf.predict(X), [2] * 6)

    X_, y_ = make_classification(n_samples=200, n_features=10,
                                 weights=[0.833, 0.167], random_state=2)

    for clf in (linear_model.LogisticRegression(),
                svm.LinearSVC(random_state=0), svm.SVC()):
        clf.set_params(class_weight={0: .1, 1: 10})
        clf.fit(X_[:100], y_[:100])
        y_pred = clf.predict(X_[100:])
        assert_true(f1_score(y_[100:], y_pred) > .3)


def test_sample_weights():
    """
    Test weights on individual samples
    """
    # TODO: check on NuSVR, OneClass, etc.
    clf = svm.SVC()
    clf.fit(X, Y)
    assert_array_equal(clf.predict(X[2]), [1.])

    sample_weight = [.1] * 3 + [10] * 3
    clf.fit(X, Y, sample_weight=sample_weight)
    assert_array_equal(clf.predict(X[2]), [2.])


def test_auto_weight():
    """Test class weights for imbalanced data"""
    from sklearn.linear_model import LogisticRegression
    # We take as dataset the two-dimensional projection of iris so
    # that it is not separable and remove half of predictors from
    # class 1.
    # We add one to the targets as a non-regression test: class_weight="auto"
    # used to work only when the labels where a range [0..K).
    from sklearn.utils import compute_class_weight
    X, y = iris.data[:, :2], iris.target + 1
    unbalanced = np.delete(np.arange(y.size), np.where(y > 2)[0][::2])

    classes, y_ind = unique(y[unbalanced], return_inverse=True)
    class_weights = compute_class_weight('auto', classes, y_ind)
    assert_true(np.argmax(class_weights) == 2)

    for clf in (svm.SVC(kernel='linear'), svm.LinearSVC(random_state=0),
                LogisticRegression()):
        # check that score is better when class='auto' is set.
        y_pred = clf.fit(X[unbalanced], y[unbalanced]).predict(X)
        clf.set_params(class_weight='auto')
        y_pred_balanced = clf.fit(X[unbalanced], y[unbalanced],).predict(X)
        assert_true(metrics.f1_score(y, y_pred)
                    <= metrics.f1_score(y, y_pred_balanced))

def test_svc_clone_with_callable_kernel():
    # create SVM with callable linear kernel, check that results are the same
    # as with built-in linear kernel
    svm_callable = svm.SVC(kernel=lambda x, y: np.dot(x, y.T),
                           probability=True)
    # clone for checking clonability with lambda functions..
    svm_cloned = base.clone(svm_callable)
    svm_cloned.fit(X, Y)

    svm_builtin = svm.SVC(kernel='linear', probability=True)
    svm_builtin.fit(X, Y)

    assert_array_almost_equal(svm_cloned.dual_coef_,
                              svm_builtin.dual_coef_)
    assert_array_almost_equal(svm_cloned.intercept_,
                              svm_builtin.intercept_)
    assert_array_equal(svm_cloned.predict(X), svm_builtin.predict(X))

    assert_array_almost_equal(svm_cloned.predict_proba(X),
                              svm_builtin.predict_proba(X))
    assert_array_almost_equal(svm_cloned.decision_function(X),
                              svm_builtin.decision_function(X))


def test_svc_bad_kernel():
    svc = svm.SVC(kernel=lambda x, y: x)
    assert_raises(ValueError, svc.fit, X, Y)


def test_timeout():
    a = svm.SVC(kernel=lambda x, y: np.dot(x, y.T), probability=True,
                max_iter=1)
    with warnings.catch_warnings(record=True) as foo:
        # Hackish way to reset the  warning counter
        from sklearn.svm import base
        base.__warningregistry__ = {}
        warnings.simplefilter("always")
        a.fit(X, Y)
        assert_equal(len(foo), 1, msg=foo)
        assert_equal(foo[0].category, ConvergenceWarning, msg=foo[0].category)

def test_unsorted_indices():
    # test that the result with sorted and unsorted indices in csr is the same
    # we use a subset of digits as iris, blobs or make_classification didn't
    # show the problem
    digits = load_digits()
    X, y = digits.data[:50], digits.target[:50]
    X_test = sparse.csr_matrix(digits.data[50:100])

    X_sparse = sparse.csr_matrix(X)
    coef_dense = svm.SVC(kernel='linear', probability=True).fit(X, y).coef_
    sparse_svc = svm.SVC(kernel='linear', probability=True).fit(X_sparse, y)
    coef_sorted = sparse_svc.coef_
    # make sure dense and sparse SVM give the same result
    assert_array_almost_equal(coef_dense, coef_sorted.toarray())

    X_sparse_unsorted = X_sparse[np.arange(X.shape[0])]
    X_test_unsorted = X_test[np.arange(X_test.shape[0])]

    # make sure we scramble the indices
    assert_false(X_sparse_unsorted.has_sorted_indices)
    assert_false(X_test_unsorted.has_sorted_indices)

    unsorted_svc = svm.SVC(kernel='linear',
                           probability=True).fit(X_sparse_unsorted, y)
    coef_unsorted = unsorted_svc.coef_
    # make sure unsorted indices give same result
    assert_array_almost_equal(coef_unsorted.toarray(), coef_sorted.toarray())
    assert_array_almost_equal(sparse_svc.predict_proba(X_test_unsorted),
                              sparse_svc.predict_proba(X_test))


def test_svc_with_custom_kernel():
    kfunc = lambda x, y: safe_sparse_dot(x, y.T)
    clf_lin = svm.SVC(kernel='linear').fit(X_sp, Y)
    clf_mylin = svm.SVC(kernel=kfunc).fit(X_sp, Y)
    assert_array_equal(clf_lin.predict(X_sp), clf_mylin.predict(X_sp))


def test_svc_iris():
    """Test the sparse SVC with the iris dataset"""
    for k in ('linear', 'poly', 'rbf'):
        sp_clf = svm.SVC(kernel=k).fit(iris.data, iris.target)
        clf = svm.SVC(kernel=k).fit(iris.data.todense(), iris.target)

        assert_array_almost_equal(clf.support_vectors_,
                                  sp_clf.support_vectors_.todense())
        assert_array_almost_equal(clf.dual_coef_, sp_clf.dual_coef_.todense())
        assert_array_almost_equal(
            clf.predict(iris.data.todense()), sp_clf.predict(iris.data))
        if k == 'linear':
            assert_array_almost_equal(clf.coef_, sp_clf.coef_.todense())


def test_error():
    """
    Test that it gives proper exception on deficient input
    """
    # impossible value of C
    assert_raises(ValueError, svm.SVC(C=-1).fit, X, Y)

    # impossible value of nu
    clf = svm.NuSVC(nu=0.0)
    assert_raises(ValueError, clf.fit, X_sp, Y)

    Y2 = Y[:-1]  # wrong dimensions for labels
    assert_raises(ValueError, clf.fit, X_sp, Y2)

    clf = svm.SVC()
    clf.fit(X_sp, Y)
    assert_array_equal(clf.predict(T), true_result)

def test_weight():
    """
    Test class weights
    """
    X_, y_ = make_classification(n_samples=200, n_features=100,
                                 weights=[0.833, 0.167], random_state=0)

    X_ = sparse.csr_matrix(X_)
    for clf in (linear_model.LogisticRegression(),
                svm.LinearSVC(random_state=0),
                svm.SVC()):
        clf.set_params(class_weight={0: 5})
        clf.fit(X_[:180], y_[:180])
        y_pred = clf.predict(X_[180:])
        assert_true(np.sum(y_pred == y_[180:]) >= 11)


def test_sample_weights():
    """
    Test weights on individual samples
    """
    clf = svm.SVC()
    clf.fit(X_sp, Y)
    assert_array_equal(clf.predict(X[2]), [1.])

    sample_weight = [.1] * 3 + [10] * 3
    clf.fit(X_sp, Y, sample_weight=sample_weight)
    assert_array_equal(clf.predict(X[2]), [2.])

         * def test_sparse_realdata():
    """
    Test on a subset from the 20newsgroups dataset.

    This catchs some bugs if input is not correctly converted into
    sparse format or weights are not correctly initialized.
    """

    data = np.array([0.03771744,  0.1003567,  0.01174647,  0.027069])
    indices = np.array([6, 5, 35, 31])
    indptr = np.array(
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4])
    X = sparse.csr_matrix((data, indices, indptr))
    y = np.array(
        [1.,  0.,  2.,  2.,  1.,  1.,  1.,  2.,  2.,  0.,  1.,  2.,  2.,
         0.,  2.,  0.,  3.,  0.,  3.,  0.,  1.,  1.,  3.,  2.,  3.,  2.,
         0.,  3.,  1.,  0.,  2.,  1.,  2.,  0.,  1.,  0.,  2.,  3.,  1.,
         3.,  0.,  1.,  0.,  0.,  2.,  0.,  1.,  2.,  2.,  2.,  3.,  2.,
         0.,  3.,  2.,  1.,  2.,  3.,  2.,  2.,  0.,  1.,  0.,  1.,  2.,
         3.,  0.,  0.,  2.,  2.,  1.,  3.,  1.,  1.,  0.,  1.,  2.,  1.,
         1.,  3.])

    clf = svm.SVC(kernel='linear').fit(X.todense(), y)
    sp_clf = svm.SVC(kernel='linear').fit(sparse.coo_matrix(X), y)

    assert_array_equal(clf.support_vectors_, sp_clf.support_vectors_.todense())
    assert_array_equal(clf.dual_coef_, sp_clf.dual_coef_.todense())


def test_sparse_svc_clone_with_callable_kernel():
    # Test that the "dense_fit" is called even though we use sparse input
    # meaning that everything works fine.
    a = svm.SVC(C=1, kernel=lambda x, y: x * y.T, probability=True)
    b = base.clone(a)

    b.fit(X_sp, Y)
    pred = b.predict(X_sp)
    b.predict_proba(X_sp)

    dense_svm = svm.SVC(C=1, kernel=lambda x, y: np.dot(x, y.T),
                        probability=True)
    pred_dense = dense_svm.fit(X, Y).predict(X)
    assert_array_equal(pred_dense, pred)
    # b.decision_function(X_sp)  # XXX : should be supported


def test_timeout():
    sp = svm.SVC(C=1, kernel=lambda x, y: x * y.T, probability=True,
                 max_iter=1)
    with warnings.catch_warnings(record=True) as foo:
        sp.fit(X_sp, Y)
        nose_assert_equal(len(foo), 1, msg=foo)
        nose_assert_equal(foo[0].category, ConvergenceWarning,
                          msg=foo[0].category)

         * */
}
