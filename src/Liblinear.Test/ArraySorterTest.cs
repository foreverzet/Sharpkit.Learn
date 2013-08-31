// -----------------------------------------------------------------------
// <copyright file="ArraySorterTest.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Liblinear.Test
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;

    /// <summary>
    /// TODO: Update summary.
    /// </summary>

    [TestClass]
    public class ArraySorterTest {


    private Random random = new Random();


    private void assertDescendingOrder(double[] array) {
        double before = array[0];
        foreach (double d in array) {
            // accept that case
            if (d == 0.0 && before == -0.0) continue;


            Assert.IsTrue(d <= before);
            before = d;
        }
    }


    private void shuffleArray(double[] array) {


        for (int i = 0; i < array.Length; i++) {
            int j = random.Next(array.Length);
            double temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }


    [TestMethod]
    public void testReversedMergesort() {


        for (int k = 1; k <= 16 * 8096; k *= 2) {
            // create random array
            double[] array = new double[k];
            for (int i = 0; i < array.Length; i++) {
                array[i] = random.NextDouble();
            }


            ArraySorter.reversedMergesort(array);
            assertDescendingOrder(array);
        }
    }


    [TestMethod]
    public void testReversedMergesortWithMeanValues() {
        double[] array = new double[] {1.0, -0.0, -1.1, 2.0, 3.0, 0.0, 4.0, -0.0, 0.0};
        shuffleArray(array);
        ArraySorter.reversedMergesort(array);
        assertDescendingOrder(array);
    }
}

 
 


 
   

 
}
