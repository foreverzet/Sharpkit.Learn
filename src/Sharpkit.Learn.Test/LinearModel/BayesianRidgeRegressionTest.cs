// -----------------------------------------------------------------------
// <copyright file="BayesianRidgeRegressionTest.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn.Test.LinearModel
{
    using System;
    using System.Linq;
    using MathNet.Numerics.LinearAlgebra.Double;
    using Microsoft.VisualStudio.TestTools.UnitTesting;
    using Sharpkit.Learn.Datasets;
    using Sharpkit.Learn.LinearModel;

    /// <summary>
    /// Tests for <see cref="BayesianRidgeRegression"/>.
    /// </summary>
    [TestClass]
    public class BayesianRidgeRegressionTest
    {
        /// <summary>
        /// Test BayesianRidge on diabetes.
        /// </summary>
        [TestMethod]
        [Ignore] //This test is ignored in scikit-learn also.
        public void TestBayesianOnDiabetes()
        {
            //raise SkipTest("XFailed Test")
            var diabetes = DiabetesDataset.Load();
            var x = diabetes.Data;
            var y = diabetes.Target;

            var clf = new BayesianRidgeRegression(computeScore : true);

           // Test with more samples than features
            clf.Fit(x, y);
            // Test that scores are increasing at each iteration
            Assert.AreEqual(clf.Scores.Count - 1, clf.Scores.Diff().Count(v =>v > 0));
    
            // Test with more features than samples
            x = x.SubMatrix(0, 5, 0, x.ColumnCount);
            y = y.SubVector(0, 5);
            clf.Fit(x, y);
            // Test that scores are increasing at each iteration
            Assert.AreEqual(clf.Scores.Count - 1, clf.Scores.Diff().Count(v => v > 0));
        }

        /// <summary>
        /// Test BayesianRidge on toy
        /// </summary>
        [TestMethod]
        public void TestToyBayesianRidgeObject()
        {
            var x = new double[,] {{1}, {2}, {6}, {8}, {10}};
            var y = new double[] {1, 2, 6, 8, 10};
            var clf = new BayesianRidgeRegression(computeScore: true);
            clf.Fit(x, y);
            var xTest = DenseMatrix.OfArray(new double[,] {{1}, {3}, {4}});
            Assert.IsTrue(
                DenseVector.OfEnumerable(new double[] {1, 3, 4}).AlmostEquals(clf.Predict(xTest).Column(0), 1E-5));
        }
    }
}
