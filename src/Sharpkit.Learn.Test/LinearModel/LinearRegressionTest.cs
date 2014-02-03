namespace Sharpkit.Learn.Test.LinearModel
{
    using System;
    using MathNet.Numerics;
    using MathNet.Numerics.Algorithms.LinearAlgebra.Mkl;
    using MathNet.Numerics.Distributions;
    using MathNet.Numerics.LinearAlgebra.Double;
    using Microsoft.VisualStudio.TestTools.UnitTesting;
    using Sharpkit.Learn.Datasets;
    using Sharpkit.Learn.LinearModel;
    using MathNet.Numerics.LinearAlgebra.Generic;

    [TestClass]
    public class LinearRegressionTest
    {
        /// <summary>
        /// Test LinearRegression on a simple dataset.
        /// </summary>
        [TestMethod]
        public void TestLinearRegression()
        {
            // Control.LinearAlgebraProvider = new MklLinearAlgebraProvider();
            // a simple dataset
            var x = DenseMatrix.OfArray(new double[,] {{1}, {2}});
            var y = DenseVector.OfEnumerable(new double[] {1, 2});

            var clf = new LinearRegression();
            clf.Fit(x, y);

            Assert.AreEqual(1.0, clf.Coef.Column(0)[0], 1E-5);
            //Assert.AreEqual(0.0, clf.Intercept[0]);
            Assert.IsTrue(DenseVector.OfEnumerable(new double[] {1, 2}).AlmostEquals(clf.Predict(x).Column(0)));

            // test it also for degenerate input
            x = DenseMatrix.OfArray(new double[,] {{1}});
            y = DenseVector.OfEnumerable(new double[] {0});


            clf = new LinearRegression(fitIntercept: false);
            clf.Fit(x, y);
            Assert.AreEqual(0.0, clf.Coef.Column(0)[0]);
            //assert_array_almost_equal(clf.intercept_, [0])
            Assert.AreEqual(0.0, clf.Predict(x).Column(0)[0]);
        }

        /// <summary>
        /// Test assertions on betas shape.
        /// </summary>
        [TestMethod]
        public void TestFitIntercept()
        {
            var x2 = DenseMatrix.OfArray(new[,]
                                             {
                                                 {0.38349978, 0.61650022},
                                                 {0.58853682, 0.41146318}
                                             });
            var x3 = DenseMatrix.OfArray(new[,]
                                             {
                                                 {0.27677969, 0.70693172, 0.01628859},
                                                 {0.08385139, 0.20692515, 0.70922346}
                                             });
            var y = DenseVector.OfEnumerable(new double[] {1, 1});


            var lr2WithoutIntercept = new LinearRegression(fitIntercept: false);
            lr2WithoutIntercept.Fit(x2, y);
            var lr2WithIntercept = new LinearRegression(fitIntercept: true);
            lr2WithIntercept.Fit(x2, y);

            var lr3WithoutIntercept = new LinearRegression(fitIntercept: false);
            lr3WithoutIntercept.Fit(x3, y);
            var lr3WithIntercept = new LinearRegression(fitIntercept: true);
            lr3WithIntercept.Fit(x3, y);

            Assert.AreEqual(lr2WithIntercept.Coef.Column(0).Count,
                            lr2WithoutIntercept.Coef.Column(0).Count);
            Assert.AreEqual(lr3WithIntercept.Coef.Column(0).Count,
                            lr3WithoutIntercept.Coef.Column(0).Count);
        }


        /// <summary>
        /// Test that linear regression also works with sparse data.
        /// </summary>
        [TestMethod]
        public void TestLinearRegressionSparse()
        {
            const int n = 100;
            Matrix x = SparseMatrix.Identity(n);
            var beta = DenseVector.CreateRandom(n, new Normal());
            var y = x*beta;

            var ols = new LinearRegression(fitIntercept: true);
            ols.Fit(x, y);
            Assert.IsTrue((ols.Coef.Row(0) + ols.Intercept[0]).AlmostEquals(beta));
        }

        /// <summary>
        /// Test multiple-outcome linear regressions.
        /// </summary>
        [TestMethod]
        public void TestLinearRegressionMultipleOutcome()
        {
            var result = SampleGenerator.MakeRegression(shuffle:false, random : new Random(0));

            Matrix y = DenseMatrix.OfColumns(
                result.Y.RowCount,
                2,
                new[]{ result.Y.Column(0), result.Y.Column(0) });
            var numFeatures = result.X.RowCount;

            var clf = new LinearRegression(fitIntercept: true);
            clf.Fit(result.X, y);
            Assert.AreEqual(Tuple.Create(2, numFeatures), clf.Coef.Shape());
            
            Matrix<double> yPred = clf.Predict(result.X);
            clf.Fit(result.X, result.Y);
            Matrix<double> yPred1 = clf.Predict(result.X);

            Assert.IsTrue(yPred1.Column(0).AlmostEquals(yPred.Column(0)));
            Assert.IsTrue(yPred1.Column(0).AlmostEquals(yPred.Column(1)));
        }
        
        /// <summary>
        /// Test multiple-outcome linear regressions with sparse data
        /// </summary>
        [TestMethod]
        public void TestLinearRegressionSparseMultipleOutcome()
        {
            var random = new Random(0);
            var r = SampleGenerator.MakeSparseUncorrelated(random : random);
            Matrix x = SparseMatrix.OfMatrix(r.X);
            Vector<double> y = r.Y.Column(0);
            Matrix y1 = DenseMatrix.OfColumns(y.Count, 2, new[] {y, y});
            int nFeatures = x.ColumnCount;

            var ols = new LinearRegression();
            ols.Fit(x, y1);
            Assert.AreEqual(Tuple.Create(2, nFeatures), ols.Coef.Shape());
            Assert.AreEqual(Tuple.Create(2, nFeatures), ols.Coef.Shape());
            Matrix<double> yPred = ols.Predict(x);
            ols.Fit(x, y);
            Matrix<double> yPred1 = ols.Predict(x);
            
            Assert.IsTrue(yPred1.Column(0).AlmostEquals(yPred.Column(0)));
            Assert.IsTrue(yPred1.Column(0).AlmostEquals(yPred.Column(1)));
        }
    }
}
