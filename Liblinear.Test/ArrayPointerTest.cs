// -----------------------------------------------------------------------
// <copyright file="ArrayPointerTest.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

namespace Liblinear.Test
{
    using System;
    using System.Linq;
    using Microsoft.VisualStudio.TestTools.UnitTesting;

    /// <summary>
    /// TODO: Update summary.
    /// </summary>

    [TestClass]
    public class ArrayPointerTest {


    [TestMethod]
    public void testGetIntArrayPointer() {
        int[] foo = new [] {1, 2, 3, 4, 6};
        ArrayPointer<int> pFoo = new ArrayPointer<int>(foo, 2);
        Assert.AreEqual(3, pFoo[0]);
        Assert.AreEqual(4, pFoo[1]);
        Assert.AreEqual(6, pFoo[2]);
        try {
            var i = pFoo[3];
            Assert.Fail("IndexOutOfRangeException expected");
        } catch (IndexOutOfRangeException) {}
    }


    [TestMethod]
    public void testSetIntArrayPointer() {
        int[] foo = new int[] {1, 2, 3, 4, 6};
        ArrayPointer<int> pFoo = new ArrayPointer<int>(foo, 2);
        pFoo[2] = 5;
        Assert.IsTrue(new [] {1, 2, 3, 4, 5}.SequenceEqual(foo));
        try {
            pFoo[3] = 0;
            Assert.Fail("IndexOutOfRangeException expected");
        } catch (IndexOutOfRangeException) {}
    }


    [TestMethod]
    public void testGetDoubleArrayPointer() {
        double[] foo = new double[] {1, 2, 3, 4, 6};
        ArrayPointer<double> pFoo = new ArrayPointer<double>(foo, 2);
        Assert.AreEqual(3, pFoo[0]);
        Assert.AreEqual(4, pFoo[1]);
        Assert.AreEqual(6, pFoo[2]);
        try {
            var i = pFoo[3];
            Assert.Fail("IndexOutOfRangeException expected");
        } catch (IndexOutOfRangeException) {}
    }


    [TestMethod]
    public void testSetDoubleArrayPointer() {
        double[] foo = new double[] {1, 2, 3, 4, 6};
        ArrayPointer<double> pFoo = new ArrayPointer<double>(foo, 2);
        pFoo[2] = 5;
        Assert.IsTrue(new double[] {1, 2, 3, 4, 5}.SequenceEqual(foo));
        try {
            pFoo[3] = 0;
            Assert.Fail("IndexOutOfRangeException expected");
        } catch (IndexOutOfRangeException) {}
    }
}

 
 


 
   

 
}
