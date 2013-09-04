// -----------------------------------------------------------------------
// <copyright file="RidgeRegression.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

using Microsoft.VisualStudio.TestTools.UnitTesting;
using Sharpkit.Learn.LinearModel;

namespace Sharpkit.Learn.Samples.CSharp
{
    using System;

    [TestClass]
    public class Ridge
    {
        [TestMethod]
        public void RidgeRegressionSample()
        {
            var clf = new RidgeRegression(alpha: 0.5);
            clf.Fit(new[,] {{0.0, 0.0}, {0.0, 0.0}, {1.0, 1.0}}, new[] {0.0, 0.1, 1.0});
            Console.WriteLine(clf.Coef);
            Console.WriteLine(clf.Intercept);

            var prediction = clf.Predict(new[] {5.0, 6.0});
            Console.WriteLine(prediction);
        }

        [TestMethod]
        public void RidgeClassifierSample()
        {
            var clf = new RidgeClassifier<string>(alpha: 0.5);
            clf.Fit(new[,] { { 0.0, 0.0 }, { 0.0, 0.0 }, { 1.0, 1.0 } }, new[] { "a", "b", "c" });
            Console.WriteLine(clf.Coef);
            Console.WriteLine(clf.Intercept);

            var prediction = clf.Predict(new[] { 5.0, 6.0 });
            Console.WriteLine(prediction);
        }
    }
}
