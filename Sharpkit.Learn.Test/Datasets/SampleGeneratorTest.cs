// -----------------------------------------------------------------------
// <copyright file="SampleGeneratorTest.cs" company="No company">
//  Authors: B. Thirion, G. Varoquaux, A. Gramfort, V. Michel, O. Grisel,
//          G. Louppe, S. Zyuzin
//  License: BSD 3 clause
// </copyright>
// -----------------------------------------------------------------------

using System.Linq;

namespace Sharpkit.Learn.Test.Datasets
{
    using System;
    using MathNet.Numerics.LinearAlgebra.Double;
    using MathNet.Numerics.LinearAlgebra.Double.Factorization;
    using MathNet.Numerics.LinearAlgebra.Generic;
    using MathNet.Numerics.Statistics;
    using Microsoft.VisualStudio.TestTools.UnitTesting;
    using Sharpkit.Learn.Datasets;

    /// <summary>
    /// Tests <see cref="SampleGenerator"/>.
    /// </summary>
    [TestClass]
    public class SampleGeneratorTest
    {
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
            Matrix x = SampleGenerator.MakeLowRankMatrix(
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
