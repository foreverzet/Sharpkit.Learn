
namespace Sharpkit.Learn.LinearModel
{
    using System;
    using MathNet.Numerics.LinearAlgebra.Double;
    using Sharpkit.Learn.LeastSquares;

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

        public void Fit(Matrix X, Matrix y, Vector sampleWeight = null)
        {
            var t = model.CenterData(X, y, model.FitIntercept, this.normalize, sampleWeight);

            this.model.CoefMatrix = RidgeRegression(t.X, t.Y,
                                                     alpha: this.alpha,
                                                     maxIter: this.maxIter,
                                                     tol: this.tol,
                                                     sampleWeight: sampleWeight,
                                                     solver: this.solver);
            this.model.SetIntercept(t.xMean, t.yMean, t.xStd);
        }

        /*
         * """Solve the ridge equation by the method of normal equations.

    Parameters
    ----------
    X : {array-like, sparse matrix, LinearOperator},
        shape = [n_samples, n_features]
        Training data

    y : array-like, shape = [n_samples] or [n_samples, n_targets]
        Target values

    max_iter : int, optional
        Maximum number of iterations for conjugate gradient solver.
        The default value is determined by scipy.sparse.linalg.

    sample_weight : float or numpy array of shape [n_samples]
        Individual weights for each sample

    solver : {'auto', 'svd', 'dense_cholesky', 'lsqr', 'sparse_cg'}
        Solver to use in the computational routines:

        - 'auto' chooses the solver automatically based on the type of data.

        - 'svd' uses a Singular Value Decomposition of X to compute the Ridge
          coefficients. More stable for singular matrices than 'dense_cholesky'.

        - 'dense_cholesky' uses the standard scipy.linalg.solve function to
          obtain a closed-form solution via a Cholesky decomposition of dot(X.T, X)

        - 'sparse_cg' uses the conjugate gradient solver as found in
          scipy.sparse.linalg.cg. As an iterative algorithm, this solver is
          more appropriate than 'dense_cholesky' for large-scale data
          (possibility to set `tol` and `max_iter`).

        - 'lsqr' uses the dedicated regularized least-squares routine
          scipy.sparse.linalg.lsqr. It is the fatest but may not be available
          in old scipy versions. It also uses an iterative procedure.

        All three solvers support both dense and sparse data.

    tol: float
        Precision of the solution.

    Returns
    -------
    coef: array, shape = [n_features] or [n_targets, n_features]
        Weight vector(s).

    Notes
    -----
    This function won't compute the intercept.
    """
         * */

        public static Matrix RidgeRegression(
            Matrix x,
            Matrix y,
            double alpha,
            Vector sampleWeight = null,
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
                double sqrt_alpha = Math.Sqrt(alpha);
                Matrix coefs = new DenseMatrix(x.ColumnCount, y.ColumnCount);
                foreach (var column in y.ColumnEnumerator())
                {
                    Vector c = Lsqr.lsqr(
                        x,
                        (Vector)column.Item2,
                        damp: sqrt_alpha,
                        atol: tol,
                        btol: tol,
                        iterLim: maxIter).X;

                    coefs.SetColumn(column.Item1, c);
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
                    Matrix K = (Matrix)x.TransposeAndMultiply(x);
                    Vector sw = null;
                    if (sampleWeight != null)
                    {
                        sw = sampleWeight.Sqrt();
                        // We are doing a little danse with the sample weights to
                        // avoid copying the original X, which could be big

                        y = y.MulColumnVector(sw);

                        K = (Matrix)K.PointwiseMultiply(sw.Outer(sw));
                    }

                    K.Add(DenseMatrix.Identity(K.RowCount)*alpha, K);
                    try
                    {
                        Matrix dual_coef = (Matrix)K.Cholesky().Solve(y);
                        if (sampleWeight != null)
                            dual_coef = dual_coef.MulColumnVector(sw);

                        return (Matrix)x.TransposeThisAndMultiply(dual_coef);
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
                    Matrix A = (Matrix)x.TransposeThisAndMultiply(x);
                    A.Add(DenseMatrix.Identity(A.ColumnCount)*alpha, A);

                    Matrix Xy = (Matrix)x.TransposeThisAndMultiply(y);

                    try
                    {
                        return (Matrix)A.Cholesky().Solve(Xy);
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
                Vector d = (Vector)(svd.S()).SubVector(0, k);
                d.MapInplace(v => v > 1e-15 ? v/(v*v + alpha) : 0.0);

                Matrix Ud = (Matrix)svd.U().SubMatrix(0, x.RowCount, 0, k).TransposeThisAndMultiply(y).Transpose();
                Ud = Ud.MulRowVector(d);
                return (Matrix)Ud.Multiply(svd.VT().SubMatrix(0, k, 0, x.ColumnCount)).Transpose();
            }

            //todo:
            return null;
        }
    }
}