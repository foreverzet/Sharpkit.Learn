// -----------------------------------------------------------------------
// <copyright file="SampleGeneratorTest.cs" company="No company">
//  Authors: B. Thirion, G. Varoquaux, A. Gramfort, V. Michel, O. Grisel,
//          G. Louppe, S. Zyuzin
//  License: BSD 3 clause
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn.Test.Datasets
{
    using System;
    using System.Linq;
    using MathNet.Numerics.LinearAlgebra.Double;
    using MathNet.Numerics.LinearAlgebra.Double.Factorization;
    using MathNet.Numerics.LinearAlgebra.Generic;
    using MathNet.Numerics.Statistics;
    using Microsoft.VisualStudio.TestTools.UnitTesting;
    using Learn.Datasets;

    /// <summary>
    /// Tests <see cref="SampleGenerator"/>.
    /// </summary>
    [TestClass]
    public class SampleGeneratorTest
    {
        [TestMethod]
        public void test_make_classification()
        {
            // todo:
    /*X, y = make_classification(n_samples=100, n_features=20, n_informative=5,
                               n_redundant=1, n_repeated=1, n_classes=3,
                               n_clusters_per_class=1, hypercube=False,
                               shift=None, scale=None, weights=[0.1, 0.25],
                               random_state=0)

    assert_equal(X.shape, (100, 20), "X shape mismatch")
    assert_equal(y.shape, (100,), "y shape mismatch")
    assert_equal(np.unique(y).shape, (3,), "Unexpected number of classes")
    assert_equal(sum(y == 0), 10, "Unexpected number of samples in class #0")
    assert_equal(sum(y == 1), 25, "Unexpected number of samples in class #1")
    assert_equal(sum(y == 2), 65, "Unexpected number of samples in class #2")
     * */
        }

        [TestMethod]
        public void test_make_multilabel_classification()
        {
         //todo:
            /*
            for allow_unlabeled, min_length in zip((True, False), (0, 1)):
            X, Y = make_multilabel_classification(n_samples=100, n_features=20,
                                                  n_classes=3, random_state=0,
                                                  allow_unlabeled=allow_unlabeled)
            assert_equal(X.shape, (100, 20), "X shape mismatch")
            if not allow_unlabeled:
                assert_equal(max([max(y) for y in Y]), 2)
            assert_equal(min([len(y) for y in Y]), min_length)
            assert_true(max([len(y) for y in Y]) <= 3)
            */
        }

        /// <summary>
        /// Tests <see cref="SampleGenerator.MakeRegression"/>.
        /// </summary>
        [TestMethod]
        public void TestMakeRegression()
        {
            var r = SampleGenerator.MakeRegression(
                numSamples: 100,
                numFeatures: 10,
                numInformative: 3,
                shuffle: false,
                effectiveRank: 5,
                coef: true,
                bias: 0.0,
                noise: 1.0,
                random: new Random(0));

            Assert.AreEqual(Tuple.Create(100, 10), r.X.Shape(), "X shape mismatch");
            Assert.AreEqual(Tuple.Create(100, 1), r.Y.Shape(), "y shape mismatch");
            Assert.AreEqual(Tuple.Create(10, 1), r.Coef.Shape(), "coef shape mismatch");
            Assert.AreEqual(3, r.Coef.Column(0).Count(v => v != 0), "Unexpected number of informative features");

            // Test that y ~= np.dot(X, c) + bias + N(0, 1.0)
            Matrix<double> matrix = r.Y - (r.X * r.Coef);
            Assert.IsTrue(Math.Abs(matrix.Column(0).StandardDeviation() - 1.0) < 0.2);
        }

        /// <summary>
        /// Tests <see cref="SampleGenerator.MakeLowRankMatrix"/>.
        /// </summary>
        [TestMethod]
        public void TestMakeLowRankMatrix()
        {
            Matrix<double> x = SampleGenerator.MakeLowRankMatrix(
                numSamples: 50,
                numFeatures: 25,
                effectiveRank: 5,
                tailStrength: 0.01,
                randomState: new Random(0));

            Assert.AreEqual(50, x.RowCount, "X shape mismatch");
            Assert.AreEqual(25, x.ColumnCount, "X shape mismatch");

            Svd svd = x.Svd(true);
            double sum = svd.S().Sum();
            Assert.IsTrue(Math.Abs(sum - 5) < 0.1, "X rank is not approximately 5");
        }
    }
}
