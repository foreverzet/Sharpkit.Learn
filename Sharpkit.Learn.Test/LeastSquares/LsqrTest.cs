using MathNet.Numerics.LinearAlgebra.Double;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Sharpkit.Learn.LeastSquares;

namespace Sharpkit.Learn.Test.LeastSquares
{
    [TestClass]
    public class LsqrTest
    {
        const int n = 35;
        private Vector b = Normal(size: n);
        private Matrix G;

        [TestInitialize]
        public void TestInitialize()
        {
            //dx Set up a test problem

            G = DenseMatrix.Identity(n);

            for (int jj = 0; jj <5; jj++)
            {
                Vector gg = Normal(size : n);
                Vector hh = (Vector)gg.PointwiseMultiply(gg);
                G.AddRowVector(hh, G);
                G.AddRowVector((Vector)(Normal(size : n).PointwiseMultiply( Normal(size : n))), G);
            }
        }

        private static Vector Normal(int size)
        {
            return DenseVector.CreateRandom(size, new MathNet.Numerics.Distributions.Normal());
        }

        [TestMethod]
        public void test_basic()
        {
            Vector svx = (Vector)G.Svd(true).Solve(b);
            Vector X = Lsqr.lsqr(a: G, b: b, show: false, atol: 1E-10, btol: 1E-10).X;
            Vector r = (Vector)(svx - X);
            Assert.IsTrue(r.FrobeniusNorm() < 1e-5);
        }
    }
}
