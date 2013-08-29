// -----------------------------------------------------------------------
// <copyright file="FeatureNodeTest.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Liblinear.Test
{
    using System;

    /// <summary>
    /// TODO: Update summary.
    /// </summary>
    [TestClass]
    public class FeatureNodeTest {


    [TestMethod]
    public void testConstructorIndexZero() {
        // since 1.5 there's no more exception here
        new Feature(0, 0);
    }


    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void testConstructorIndexNegative() {
        new Feature(-1, 0);
    }


    [TestMethod]
    public void testConstructorHappy() {
        Feature fn = new Feature(25, 27.39);
        Assert.AreEqual(25, fn.Index);
        Assert.AreEqual(27.39, fn.Value);

        fn = new Feature(1, -0.22222);
        Assert.AreEqual(1, fn.Index);
        Assert.AreEqual(-0.22222, fn.Value);
    }
}

}
