// -----------------------------------------------------------------------
// <copyright file="LinearRegression.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn.Samples.CSharp
{
    using System;
    using Microsoft.VisualStudio.TestTools.UnitTesting;

    /// <summary>
    /// TODO: Update summary.
    /// </summary>
    [TestClass]
    public class LinearRegressionSamples
    {
        [TestMethod]
        public void LinearRegressionSample()
        {
            // Learn
            var clf = new Sharpkit.Learn.LinearModel.LinearRegression();
            clf.Fit(new double[,] {{0, 0}, {1, 1}, {2, 2}}, new double[] {0, 1, 2});
            Console.WriteLine(clf.Coef.ToString());

            // Predict
            var prediction = clf.Predict(new double[] {3, 3});
            Console.WriteLine(prediction);
        }
    }
}
