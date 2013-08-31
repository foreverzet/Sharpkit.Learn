// -----------------------------------------------------------------------
// <copyright file="ClassWeightTest.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

using MathNet.Numerics.LinearAlgebra.Double;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Sharpkit.Learn.Test
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;

    /// <summary>
    /// TODO: Update summary.
    /// </summary>
    [TestClass]
    public class ClassWeightTest
    {
        //  """Test (and demo) compute_class_weight."""
        [TestMethod]
        public void test_compute_class_weight()
        {
            int[] classes = new[] {2, 3, 4};
            int[] y_ind = new[] {0, 0, 0, 1, 1, 2};
            Vector cw = ClassWeight<int>.Auto.ComputeWeights(classes, y_ind);
            Assert.AreEqual(cw.Sum(), classes.Length);
            Assert.IsTrue(cw[0] < cw[1] && cw[1] < cw[2]);
        }

        //"""Test compute_class_weight in case y doesn't contain all classes."""
        [TestMethod]
        public void test_compute_class_weight_not_present()
        {
            int[] classes = new[] {0, 1, 2, 3};
            int[] y = new[] {0, 0, 0, 1, 1, 2};
            Vector cw = ClassWeight<int>.Auto.ComputeWeights(classes, y);
            Assert.AreEqual(cw.Sum(), classes.Length);
            Assert.AreEqual(cw.Count, classes.Length);
            Assert.IsTrue(cw[0] < cw[1] && cw[1] < cw[2] && cw[2] <= cw[3]);
        }
    }
}
