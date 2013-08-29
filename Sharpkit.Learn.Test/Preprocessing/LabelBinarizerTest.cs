// -----------------------------------------------------------------------
// <copyright file="LabelBinarizerTest.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

using Sharpkit.Learn.Preprocessing;

namespace Sharpkit.Learn.Test.Preprocessing
{
    using MathNet.Numerics.LinearAlgebra.Double;
    using Microsoft.VisualStudio.TestTools.UnitTesting;
    using System;
    using System.Linq;

    /// <summary>
    /// TODO: Update summary.
    /// </summary>
    [TestClass]
    public class LabelBinarizerTest
    {
        [TestMethod]
        public void TestLabelBinarizer()
        {
            var lb = new LabelBinarizer<string>();

            // two-class case
            var inp = new[] {"neg", "pos", "pos", "neg"};
            var expected = new double[,] {{0, 1, 1, 0}}.ToDenseMatrix().Transpose();
            var got = lb.Fit(inp).Transform(inp);
            Assert.AreEqual(expected, got);
            Assert.IsTrue(lb.InverseTransform(got).SequenceEqual(inp));

            // multi-class case
            inp = new[] {"spam", "ham", "eggs", "ham", "0"};
            expected = new double[,]
                           {
                               {0, 0, 0, 1},
                               {0, 0, 1, 0},
                               {0, 1, 0, 0},
                               {0, 0, 1, 0},
                               {1, 0, 0, 0}
                           }.ToDenseMatrix();

            got = lb.Fit(inp).Transform(inp);
            Assert.AreEqual(expected, got);
            Assert.IsTrue(lb.InverseTransform(got).SequenceEqual(inp));
        }

        [TestMethod]
        public void TestLabelBinarizerSetLabelEncoding()
        {
            var lb = new LabelBinarizer<int>(negLabel: -2, posLabel: 2);

            // two-class case
            var inp = new[] {0, 1, 1, 0};
            var expected = new double[,] {{-2, 2, 2, -2}}.ToDenseMatrix().Transpose();
            var got = lb.Fit(inp).Transform(inp);
            Assert.AreEqual(expected, got);
            Assert.IsTrue(lb.InverseTransform(got).SequenceEqual(inp));

            // multi-class case
            inp = new[] {3, 2, 1, 2, 0};
            expected = new double[,]
                           {
                               {-2, -2, -2, +2},
                               {-2, -2, +2, -2},
                               {-2, +2, -2, -2},
                               {-2, -2, +2, -2},
                               {+2, -2, -2, -2}
                           }.ToDenseMatrix();

            got = lb.Fit(inp).Transform(inp);
            Assert.AreEqual(expected, got);
            Assert.IsTrue(lb.InverseTransform(got).SequenceEqual(inp));
        }

        /// <summary>
        /// Check that invalid arguments yield ArgumentException.
        /// </summary>
        [TestMethod]
        public void TestLabelBinarizerErrors()
        {
            var oneClass = new[] {0, 0, 0, 0};
            var lb = new LabelBinarizer<int>().Fit(oneClass);

            lb = new LabelBinarizer<int>();
            try
            {
                lb.Transform(new int[0]);
                Assert.Fail();
            }
            catch (InvalidOperationException)
            {
            }

            try
            {
                lb.InverseTransform(new DenseMatrix(0, 0));
                Assert.Fail();
            }
            catch (ArgumentException)
            {
            }

            try
            {
                new LabelBinarizer<int>(negLabel: 2, posLabel: 1);
                Assert.Fail();
            }
            catch (ArgumentException)
            {
            }

            try
            {
                new LabelBinarizer<int>(negLabel: 2, posLabel: 2);
                Assert.Fail();
            }
            catch (ArgumentException)
            {
            }
        }
    }
}
