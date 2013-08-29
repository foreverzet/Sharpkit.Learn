using System;
using MathNet.Numerics;
using MathNet.Numerics.Algorithms.LinearAlgebra.Mkl;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra.Double;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Sharpkit.Learn.Datasets;
using Sharpkit.Learn.LinearModel;

namespace Sharpkit.Learn.Test
{
    [TestClass]
    public class LinearRegressionTest
    {
        /// <summary>
        /// Test LinearRegression on a simple dataset.
        /// </summary>
        [TestMethod]
        public void test_linear_regression()
        {
            Control.LinearAlgebraProvider = new MklLinearAlgebraProvider();
            // a simple dataset
            var X = DenseMatrix.OfArray(new double[,] {{1}, {2}});
            var Y = DenseVector.OfEnumerable(new double[] {1, 2});

            var clf = new LinearRegression();
            clf.Fit(X, Y);

            Assert.AreEqual(1.0, clf.Coef[0], 1E-5);
            //Assert.AreEqual(0.0, clf.Intercept[0]);
            Assert.IsTrue(((Vector)(DenseVector.OfEnumerable(new double[] {1, 2}) - clf.Predict(X).Column(0))).FrobeniusNorm() <
                          0.0001);

            // test it also for degenerate input
            X = DenseMatrix.OfArray(new double[,] {{1}});
            Y = DenseVector.OfEnumerable(new double[] {0});


            clf = new LinearRegression(fitIntercept: false);
            clf.Fit(X, Y);
            Assert.AreEqual(0.0, clf.Coef[0]);
            //assert_array_almost_equal(clf.intercept_, [0])
            Assert.AreEqual(0.0, clf.Predict(X).Column(0)[0]);
        }

        /// <summary>
        /// Test assertions on betas shape.
        /// </summary>
        [TestMethod]
        public void test_fit_intercept()
        {
            Control.LinearAlgebraProvider = new MklLinearAlgebraProvider();
            var X2 = DenseMatrix.OfArray(new[,]
                                             {
                                                 {0.38349978, 0.61650022},
                                                 {0.58853682, 0.41146318}
                                             });
            var X3 = DenseMatrix.OfArray(new[,]
                                             {
                                                 {0.27677969, 0.70693172, 0.01628859},
                                                 {0.08385139, 0.20692515, 0.70922346}
                                             });
            var y = DenseVector.OfEnumerable(new double[] {1, 1});


            var lr2_without_intercept = new LinearRegression(fitIntercept: false).Fit(X2, y);
            var lr2_with_intercept = new LinearRegression(fitIntercept: true).Fit(X2, y);

            var lr3_without_intercept = new LinearRegression(fitIntercept: false).Fit(X3, y);
            var lr3_with_intercept = new LinearRegression(fitIntercept: true).Fit(X3, y);


            Assert.AreEqual(lr2_with_intercept.Coef.Count,
                            lr2_without_intercept.Coef.Count);
            Assert.AreEqual(lr3_with_intercept.Coef.Count,
                            lr3_without_intercept.Coef.Count);
        }


        /// <summary>
        /// Test that linear regression also works with sparse data.
        /// </summary>
        /// <param name="?"></param>
        [TestMethod]
        public void test_linear_regression_sparse()
        {
            int n = 100;
            Matrix X = SparseMatrix.Identity(n);
            var beta = DenseVector.CreateRandom(n, new Normal());
            Vector y = (Vector)(X*beta);


            var ols = new LinearRegression(fitIntercept: true).Fit(X, y);
            Assert.IsTrue(((Vector)(ols.Coef + ols.Intercept - beta)).FrobeniusNorm() < 0.0001);
        }

        //"Test multiple-outcome linear regressions"
        [TestMethod]
        public void test_linear_regression_multiple_outcome()
        {
            var result = SampleGenerator.MakeRegression(shuffle:false, random : new Random(0));

            Matrix y = DenseMatrix.OfColumns(
                result.Y.RowCount,
                2,
                new[]{ result.Y.Column(0), result.Y.Column(0) });
            var numFeatures = result.X.RowCount;

            var clf = new LinearRegression(fitIntercept: true);
            clf.Fit(result.X, y);
            Assert.AreEqual(2, clf.CoefMatrix.ColumnCount);
            Assert.AreEqual(numFeatures, clf.CoefMatrix.RowCount);
            
            Matrix Y_pred = clf.Predict(result.X);
            clf.Fit(result.X, result.Y);
            Matrix y_pred = clf.Predict(result.X);

            Assert.AreEqual(((Vector)(y_pred.Column(0) - Y_pred.Column(0))).FrobeniusNorm(), 0.0,  1E-10);
            Assert.AreEqual(((Vector)(y_pred.Column(0) - Y_pred.Column(1))).FrobeniusNorm(), 0.0,  1E-10);
        }
        
        /// <summary>
        /// Test multiple-outcome linear regressions with sparse data
        /// </summary>
        [TestMethod]
        public void test_linear_regression_sparse_multiple_outcome()
        {
            var random = new Random(0);
            var r = SampleGenerator.MakeSparseUncorrelated(random : random);
            Matrix X = SparseMatrix.OfMatrix(r.X);
            Vector y = (Vector)r.Y.Column(0);
            Matrix Y = DenseMatrix.OfColumns(y.Count, 2, new[] {y, y});
            int n_features = X.ColumnCount;

            var ols = new LinearRegression();
            ols.Fit(X, Y);
            Assert.AreEqual(2, ols.CoefMatrix.ColumnCount);
            Assert.AreEqual(n_features, ols.CoefMatrix.RowCount);
            Matrix Y_pred = ols.Predict(X);
            ols.Fit(X, y);
            Matrix y_pred = ols.Predict(X);
            
            Assert.AreEqual(((Vector)(y_pred.Column(0) - Y_pred.Column(0))).FrobeniusNorm(), 0.0,  1E-10);
            Assert.AreEqual(((Vector)(y_pred.Column(0) - Y_pred.Column(1))).FrobeniusNorm(), 0.0, 1E-10);
        }
    }
}
