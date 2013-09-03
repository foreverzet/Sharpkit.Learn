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
            var clf = new Sharpkit.Learn.LinearModel.LinearRegression();
            clf.Fit(new double[,] {{0, 0}, {1, 1}, {2, 2}}, new double[] {0, 1, 2});
            Console.Write(clf.Coef.ToString());
        }
    }
}
