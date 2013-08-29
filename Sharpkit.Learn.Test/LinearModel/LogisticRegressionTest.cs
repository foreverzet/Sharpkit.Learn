// -----------------------------------------------------------------------
// <copyright file="LogisticRegressionTest.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

using System.IO;
using System.Text;
using Liblinear;
using MathNet.Numerics.LinearAlgebra.Double;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Sharpkit.Learn.Datasets;
using Sharpkit.Learn.LinearModel;

namespace Sharpkit.Learn.Test.LinearModel
{
    using System;
    using System.Linq;

    /// <summary>
    /// TODO: Update summary.
    /// </summary>
    [TestClass]
    public class LogisticRegressionTest
    {
        [ClassInitialize]
        public static void a(TestContext t)
        {
            Linear.disableDebugOutput();
        }

        private Matrix X = DenseMatrix.OfArray(new double[,] { { -1, 0 }, { 0, 1 }, { 1, 1 } });
        private Matrix X_sp = SparseMatrix.OfArray(new double[,] {{-1, 0}, {0, 1}, {1, 1}});

        private int[] Y1 = new[] {0, 1, 1};
        private int[] Y2 = new[] {2, 1, 0};
        private IrisDataset iris = IrisDataset.Load();

        /// <summary>
        /// Check that the model is able to fit the classification data
        /// </summary>
        private void check_predictions(LogisticRegression<int> clf, Matrix X, int[] y)
        {
            int n_samples = y.Length;
            int[] classes = y.Distinct().OrderBy(v => v).ToArray();
            int n_classes = classes.Length;

            var predicted = clf.Fit(X, y).Predict(X);
            Assert.IsTrue(classes.SequenceEqual(clf.Classes));

            Assert.AreEqual(n_samples, predicted.Length);
            Assert.IsTrue(y.SequenceEqual(predicted));
                
            Matrix probabilities = clf.PredictProba(X);
            Assert.AreEqual(Tuple.Create(n_samples, n_classes), probabilities.Shape());
            Assert.IsTrue(probabilities.SumColumns().AlmostEquals(DenseVector.Create(probabilities.RowCount, i => 1.0)));
            Assert.IsTrue(y.SequenceEqual(probabilities.ArgmaxColumns()));
        }
        
        /// <summary>
        /// Simple sanity check on a 2 classes dataset
        /// Make sure it predicts the correct result on simple datasets.
        /// </summary>
        [TestMethod]
        public void test_predict_2_classes()
        {
            check_predictions(new LogisticRegression<int>(random: new Random(0)), X, Y1);
            check_predictions(new LogisticRegression<int>(random: new Random(0)), X_sp, Y1);

            check_predictions(new LogisticRegression<int>(C : 100, random : new Random(0)), X, Y1);
            check_predictions(new LogisticRegression<int>(C : 100, random : new Random(0)), X_sp, Y1);

            check_predictions(new LogisticRegression<int>(fitIntercept: false, random : new Random(0)), X, Y1);
            check_predictions(new LogisticRegression<int>(fitIntercept: false, random :new Random(0)), X_sp, Y1);
        }
        /*
        public void test_error()
        {
            //"""Test for appropriate exception on errors"""
            assert_raises(ValueError, logistic.LogisticRegression(C = -1).fit, X, Y1)
        }
        */
        [TestMethod]
        public void test_predict_3_classes()
        {
            check_predictions(new LogisticRegression<int>(C: 10), X, Y2);
            check_predictions(new LogisticRegression<int>(C: 10), X_sp, Y2);
        }
        
        /// <summary>
        /// Test logistic regression with the iris dataset.
        /// </summary>
        [TestMethod]
        public void test_predict_iris()
        {
            int n_samples = iris.Data.RowCount;

            string[] target = iris.Target.Select(v => iris.TargetNames[v]).ToArray();
            var clf = new LogisticRegression<string>(C: iris.Data.RowCount);
            clf.Fit(iris.Data, target);
            Assert.IsTrue(target.Distinct().OrderBy(t => t).SequenceEqual(clf.Classes));

            var pred = clf.Predict(iris.Data);
            var matchingN = pred.Zip(target, Tuple.Create).Where(t => t.Item1 == t.Item2).Count();
            Assert.IsTrue(1.0*matchingN/pred.Length > 0.95);

            var probabilities = clf.PredictProba(iris.Data);
            Assert.IsTrue(probabilities.SumColumns().AlmostEquals(DenseVector.Create(n_samples, i => 1.0)));

            pred = probabilities.RowEnumerator().Select(r => iris.TargetNames[r.Item2.MaximumIndex()]).ToArray();
            matchingN = pred.Zip(target, Tuple.Create).Where(t => t.Item1 == t.Item2).Count();
            Assert.IsTrue(1.0 * matchingN / pred.Length > 0.95);
        }
        
        /// <summary>
        /// Test sparsify and densify members.
        /// </summary>
        [TestMethod]
        public void test_sparsify()
        {
            int n_samples = iris.Data.RowCount;
            int n_features = iris.Data.ColumnCount;
            string[] target = iris.Target.Select(t => iris.TargetNames[t]).ToArray();
            var clf = new LogisticRegression<string>(random: new Random(0));
            clf.Fit(iris.Data, target);

            Matrix pred_d_d = clf.DecisionFunction(iris.Data);

            clf.Sparsify();
            Assert.IsInstanceOfType(clf.CoefMatrix, typeof(SparseMatrix));
            Matrix pred_s_d = clf.DecisionFunction(iris.Data);

            Matrix sp_data = new SparseMatrix(iris.Data);
            Matrix pred_s_s = clf.DecisionFunction(sp_data);

            clf.Densify();
            Matrix pred_d_s = clf.DecisionFunction(sp_data);

            Assert.IsTrue(pred_d_d.AlmostEquals(pred_s_d));
            Assert.IsTrue(pred_d_d.AlmostEquals(pred_s_s));
            Assert.IsTrue(pred_d_d.AlmostEquals(pred_d_s));
        }

        /*
        /// <summary>
        /// Test that an exception is raised on inconsistent input.
        /// </summary>
        [TestMethod]
        public void test_inconsistent_input()
        {
            rng = np.random.RandomState(0)
            X_ = rng.random_sample((5, 10))
            y_ = np.ones(X_.shape[0])
            y_[0] = 0

            clf = logistic.LogisticRegression(random_state = 0)

            //# Wrong dimensions for training data
            y_wrong = y_[:
            -1]
            assert_raises(ValueError, clf.fit, X, y_wrong)

            //# Wrong dimensions for test data
            assert_raises(ValueError, clf.fit(X_, y_).predict,
                          rng.random_sample((3, 12)))
        }

        /// <summary>
        /// Test that we can write to coef_ and intercept_.
        /// </summary>
        [TestMethod]
        public void test_write_parameters()
        {
            clf = logistic.LogisticRegression(random_state = 0)
            clf.fit(X, Y1)
            clf.coef_[:] = 0
            clf.intercept_[:] = 0
            assert_array_equal(clf.decision_function(X), 0)
        }

        /// <summary>
        /// Test proper NaN handling.
        /// </summary>
        [TestMethod]
        [ExpectedException(typeof (ArgumentException))]
        public void test_nan()
        {
            Xnan = np.array(X, dtype = np.float64)
            Xnan[0, 1] = np.nan
            logistic.LogisticRegression(random_state = 0).fit(Xnan, Y1)
        }

        [TestMethod]
        public void test_liblinear_random_state()
        {
            X, y = datasets.make_classification(n_samples = 20)
            lr1 = logistic.LogisticRegression(random_state = 0)
            lr1.fit(X, y)
            lr2 = logistic.LogisticRegression(random_state = 0)
            lr2.fit(X, y)
            assert_array_almost_equal(lr1.coef_, lr2.coef_)
        }
         * */
    }
}
