namespace Sharpkit.Learn.Test.LeastSquares
{
    using MathNet.Numerics.LinearAlgebra.Double;
    using Microsoft.VisualStudio.TestTools.UnitTesting;
    using Sharpkit.Learn.LeastSquares;
    using MathNet.Numerics.LinearAlgebra.Generic;

    [TestClass]
    public class LsqrTest
    {
        const int n = 35;
        private Vector b = Normal(size: n);
        private Matrix g;

        [TestInitialize]
        public void TestInitialize()
        {
            //dx Set up a test problem

            g = DenseMatrix.Identity(n);

            for (int jj = 0; jj <5; jj++)
            {
                Vector gg = Normal(size : n);
                var hh = gg.PointwiseMultiply(gg);
                g.AddRowVector(hh, g);
                g.AddRowVector(Normal(size : n).PointwiseMultiply( Normal(size : n)), g);
            }
        }

        private static Vector Normal(int size)
        {
            return DenseVector.CreateRandom(size, new MathNet.Numerics.Distributions.Normal());
        }

        [TestMethod]
        public void TestBasic()
        {
            var svx = g.Svd(true).Solve(b);
            Vector<double> x = Lsqr.lsqr(a: g, b: b, show: false, atol: 1E-10, btol: 1E-10).X;
            var r = svx - x;
            Assert.IsTrue(r.FrobeniusNorm() < 1e-5);
        }
    }
}
