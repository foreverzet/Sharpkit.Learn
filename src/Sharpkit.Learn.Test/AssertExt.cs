// -----------------------------------------------------------------------
// <copyright file="AssertExt.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

using MathNet.Numerics.LinearAlgebra.Generic;
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
    public class AssertExt
    {
        public static void ArrayEqual<T>(T[] left, T[] right, string message)
        {
            Assert.IsTrue(left.SequenceEqual(right), message);
        }

        public static void AlmostEqual(double[] left, double[] right, string message, double precision = 1E-10)
        {
            Assert.IsTrue(left.AlmostEquals(right), message, precision);
        }

        public static void AlmostEqual(double[] left, double[] right, double precision = 1E-10)
        {
            Assert.IsTrue(left.AlmostEquals(right, precision));
        }

        public static void AlmostEqual(Matrix<double> left, Matrix<double> right, string message, double precision = 1E-10)
        {
            Assert.IsTrue(left.AlmostEquals(right), message, precision);
        }

        public static void AlmostEqual(Vector<double> left, Vector<double> right, string message, double precision = 1E-10)
        {
            Assert.IsTrue(left.AlmostEquals(right), message, precision);
        }
    }
}
