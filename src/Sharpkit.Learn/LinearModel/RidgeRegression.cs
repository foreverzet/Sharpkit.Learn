// -----------------------------------------------------------------------
// <copyright file="RidgeRegression.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn.LinearModel
{
    using System;
    using MathNet.Numerics.LinearAlgebra.Double;
    using MathNet.Numerics.LinearAlgebra.Generic;

    /// <summary>
    /// Linear least squares with l2 regularization.
    ///
    /// This model solves a regression model where the loss function is
    /// the linear least squares function and regularization is given by
    /// the l2-norm. Also known as Ridge Regression or Tikhonov regularization.
    /// This estimator has built-in support for multi-variate regression
    /// (i.e., when y is a 2d-array of shape [n_samples, n_targets]).
    /// </summary>
    public class RidgeRegression: LinearRegressor
    {
        private readonly RidgeBase ridgeBase;

        /// <summary>
        /// Initializes a new instance of the RidgeRegression class.
        /// </summary>
        /// <param name="alpha">
        /// Small positive values of alpha improve the conditioning of the problem
        /// and reduce the variance of the estimates.  Alpha corresponds to
        /// ``(2*C)^-1`` in other linear models such as LogisticRegression or
        /// LinearSVC.
        /// </param>
        /// <param name="fitIntercept">
        /// Whether to calculate the intercept for this model. If set
        /// to false, no intercept will be used in calculations
        /// (e.g. data is expected to be already centered).
        /// </param>
        /// <param name="normalize">
        /// If True, the regressors X will be normalized before regression.
        /// </param>
        /// <param name="maxIter">
        /// Maximum number of iterations for conjugate gradient solver.
        /// The default value is determined by Math.Net.
        /// </param>
        /// <param name="tol">Precision of the solution.</param>
        /// <param name="solver">Solver to use in the computational routines.</param>
        public RidgeRegression(
            double alpha = 1.0,
            bool fitIntercept = true,
            bool normalize = false,
            int? maxIter = null,
            double tol = 1e-3,
            RidgeSolver solver = RidgeSolver.Auto) : base(fitIntercept)
        {
            ridgeBase = new RidgeBase(this, alpha, normalize, maxIter, tol, solver);
        }

        public double Score(Matrix<double> x, Matrix<double> y)
        {
            return Metrics.Metrics.R2Score(y, this.Predict(x));
        }

        public double Score(Matrix<double> x, Vector<double> y)
        {
            return Metrics.Metrics.R2Score(y.ToColumnMatrix(), this.Predict(x));
        }

        public override LinearRegressor Fit(Matrix<double> x, Matrix<double> y, Vector<double> sampleWeight = null)
        {
            ridgeBase.Fit(x, y, sampleWeight);
            return this;
        }
    }
}
