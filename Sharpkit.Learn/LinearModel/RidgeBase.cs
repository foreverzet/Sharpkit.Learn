
namespace Sharpkit.Learn.LinearModel
{
    using System;
    using MathNet.Numerics.LinearAlgebra.Double;
    using LeastSquares;
    using MathNet.Numerics.LinearAlgebra.Generic;

    internal class RidgeBase
    {
        private readonly double alpha;
        private readonly bool normalize;
        private readonly int? maxIter;
        private readonly double tol;
        private readonly RidgeSolver solver;
        private readonly LinearModel model;

        public RidgeBase(
            LinearModel model,
            double alpha = 1.0,
            bool normalize = false,
            int? maxIter = null,
            double tol = 1e-3,
            RidgeSolver solver = RidgeSolver.Auto)
        {
            this.model = model;
            this.alpha = alpha;
            this.normalize = normalize;
            this.maxIter = maxIter;
            this.tol = tol;
            this.solver = solver;
        }

        public void Fit(Matrix<double> x, Matrix<double> y, Vector<double> sampleWeight = null)
        {
            var t = model.CenterData(x, y, model.FitIntercept, this.normalize, sampleWeight);

            this.model.Coef = RidgeRegression(t.X, t.Y,
                                                     alpha: this.alpha,
                                                     maxIter: this.maxIter,
                                                     tol: this.tol,
                                                     sampleWeight: sampleWeight,
                                                     solver: this.solver);
            this.model.SetIntercept(t.xMean, t.yMean, t.xStd);
        }

        /// <summary>
        /// Solve the ridge equation by the method of normal equations.
        /// </summary>
        /// <param name="x">[n_samples, n_features]
        /// Training data</param>
        /// <param name="y">[n_samples, n_targets]
        /// Target values</param>
        /// <param name="alpha"></param>
        /// <param name="sampleWeight">Individual weights for each sample.</param>
        /// <param name="solver">Solver to use in the computational routines.</param>
        /// <param name="maxIter">Maximum number of iterations for least squares solver. </param>
        /// <param name="tol">Precision of the solution.</param>
        /// <returns>[n_targets, n_features]
        /// Weight vector(s)</returns>
        /// <remarks>
        /// This function won't compute the intercept;
        /// </remarks>
        public static Matrix<double> RidgeRegression(
            Matrix<double> x,
            Matrix<double> y,
            double alpha,
            Vector<double> sampleWeight = null,
            RidgeSolver solver = RidgeSolver.Auto,
            int? maxIter = null,
            double tol = 1E-3)
        {
            int nSamples = x.RowCount;
            int nFeatures = x.ColumnCount;

            if (solver == RidgeSolver.Auto)
            {
                // cholesky if it's a dense array and lsqr in
                // any other case
                if (x is DenseMatrix)
                {
                    solver = RidgeSolver.DenseCholesky;
                }
                else
                {
                    solver = RidgeSolver.Lsqr;
                }
            }

            if (sampleWeight != null)
            {
                solver = RidgeSolver.DenseCholesky;
            }

            if (solver == RidgeSolver.Lsqr)
            {
                // According to the lsqr documentation, alpha = damp^2.
                double sqrtAlpha = Math.Sqrt(alpha);
                Matrix coefs = new DenseMatrix(y.ColumnCount, x.ColumnCount);
                foreach (var column in y.ColumnEnumerator())
                {
                    Vector<double> c = Lsqr.lsqr(
                        x,
                        column.Item2,
                        damp: sqrtAlpha,
                        atol: tol,
                        btol: tol,
                        iterLim: maxIter).X;

                    coefs.SetRow(column.Item1, c);
                }

                return coefs;
            }

            if (solver == RidgeSolver.DenseCholesky)
            {
                //# normal equations (cholesky) method
                if (nFeatures > nSamples || sampleWeight != null)
                {
                    // kernel ridge
                    // w = X.T * inv(X X^t + alpha*Id) y
                    var k = x.TransposeAndMultiply(x);
                    Vector<double> sw = null;
                    if (sampleWeight != null)
                    {
                        sw = sampleWeight.Sqrt();
                        // We are doing a little danse with the sample weights to
                        // avoid copying the original X, which could be big

                        y = y.MulColumnVector(sw);

                        k = k.PointwiseMultiply(sw.Outer(sw));
                    }

                    k.Add(DenseMatrix.Identity(k.RowCount)*alpha, k);
                    try
                    {
                        var dualCoef = k.Cholesky().Solve(y);
                        if (sampleWeight != null)
                            dualCoef = dualCoef.MulColumnVector(sw);

                        return x.TransposeThisAndMultiply(dualCoef).Transpose();
                    }
                    catch (Exception) //todo:
                    {
                        // use SVD solver if matrix is singular
                        solver = RidgeSolver.Svd;
                    }
                }
                else
                {
                    // ridge
                    // w = inv(X^t X + alpha*Id) * X.T y
                    var a = x.TransposeThisAndMultiply(x);
                    a.Add(DenseMatrix.Identity(a.ColumnCount)*alpha, a);

                    var xy = x.TransposeThisAndMultiply(y);

                    try
                    {
                        return a.Cholesky().Solve(xy).Transpose();
                    }
                    catch (Exception) //todo:
                    {
                        // use SVD solver if matrix is singular
                        solver = RidgeSolver.Svd;
                    }
                }
            }

            if (solver == RidgeSolver.Svd)
            {
                // slower than cholesky but does not break with
                // singular matrices
                var svd = x.Svd(true);
                //U, s, Vt = linalg.svd(X, full_matrices=False)
                int k = Math.Min(x.ColumnCount, x.RowCount);
                var d = svd.S().SubVector(0, k);
                d.MapInplace(v => v > 1e-15 ? v/(v*v + alpha) : 0.0);

                var ud = svd.U().SubMatrix(0, x.RowCount, 0, k).TransposeThisAndMultiply(y).Transpose();
                ud = ud.MulRowVector(d);
                return ud.Multiply(svd.VT().SubMatrix(0, k, 0, x.ColumnCount));
            }

            return null;
        }
    }
}