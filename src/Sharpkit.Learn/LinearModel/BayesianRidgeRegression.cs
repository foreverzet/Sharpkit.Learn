// -----------------------------------------------------------------------
// <copyright file="BayesianRidgeRegression.cs" company="Sharpkit.Learn">
// Authors: V. Michel, F. Pedregosa, A. Gramfort, S.Zyuzin
// License: BSD 3 clause
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn.LinearModel
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

    using MathNet.Numerics.LinearAlgebra.Double;
    using MathNet.Numerics.LinearAlgebra.Generic;
    using MathNet.Numerics.Statistics;

    /// <summary>
    /// Bayesian ridge regression.
    /// <para>
    /// Fit a Bayesian ridge model and optimize the regularization parameters
    /// lambda (precision of the weights) and alpha (precision of the noise).
    /// </para>
    /// </summary>
    /// <example>
    /// >>> from sklearn import linear_model
    /// >>> clf = linear_model.BayesianRidge()
    /// >>> clf.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
    /// ... # doctest: +NORMALIZE_WHITESPACE
    /// BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False,
    ///        copy_X=True, fit_intercept=True, lambda_1=1e-06, lambda_2=1e-06,
    ///        n_iter=300, normalize=False, tol=0.001, verbose=False)
    /// >>> clf.predict([[1, 1]])
    /// array([ 1.])
    /// </example>
    public class BayesianRidgeRegression : LinearRegressor
    {
        /// <summary>
        /// Initializes a new instance of the BayesianRidgeRegression class.
        /// </summary>
        /// <param name="numIter">Maximum number of iterations. Default is 300.</param>
        /// <param name="tol">Stop the algorithm if w has converged. Default is 1E-3.</param>
        /// <param name="alpha1">
        /// Hyper-parameter : shape parameter for the Gamma distribution prior
        /// over the alpha parameter. Default is 1E-6.
        /// </param>
        /// <param name="alpha2">
        /// Hyper-parameter : inverse scale parameter (rate parameter) for the
        /// Gamma distribution prior over the alpha parameter.
        /// Default is 1.e-6.
        /// </param>
        /// <param name="lambda1">
        /// Hyper-parameter : shape parameter for the Gamma distribution prior
        /// over the lambda parameter. Default is 1Ee-6.
        /// </param>
        /// <param name="lambda2">
        /// Hyper-parameter : inverse scale parameter (rate parameter) for the
        /// Gamma distribution prior over the lambda parameter.
        /// Default is 1E-6.
        /// </param>
        /// <param name="computeScore">If True, compute the objective function at each step of the model.
        /// Default is false
        /// </param>
        /// <param name="fitIntercept">
        /// whether to calculate the intercept for this model. If set
        /// to false, no intercept will be used in calculations
        /// (e.g. data is expected to be already centered).
        /// Default is true.
        /// </param>
        /// <param name="normalize">If True, the regressors X will be normalized before regression.</param>
        /// <param name="verbose">Verbose mode when fitting the model.</param>
        public BayesianRidgeRegression(
            int numIter = 300,
            double tol = 1E-3,
            double alpha1 = 1E-6,
            double alpha2 = 1E-6,
            double lambda1 = 1E-6,
            double lambda2 = 1E-6,
            bool computeScore = false,
            bool fitIntercept = true,
            bool normalize = false,
            bool verbose = false)
            : base(fitIntercept)
        {
            this.NumIter = numIter;
            this.Tol = tol;
            this.Alpha1 = alpha1;
            this.Alpha2 = alpha2;
            this.Lambda1 = lambda1;
            this.Lambda2 = lambda2;
            this.Normalize = normalize;
            this.Verbose = verbose;
            this.ComputeScore = computeScore;
        }

        /// <summary>
        /// Gets a value indicating whether the regressors X will be normalized before regression.
        /// </summary>
        public bool Normalize { get; private set; }

        /// <summary>
        /// Gets a value indicating whether to compute the objective function at each step of the model.
        /// </summary>
        public bool ComputeScore { get; private set; }

        /// <summary>
        /// Gets hyper-parameter : inverse scale parameter (rate parameter) for the
        /// Gamma distribution prior over the lambda parameter.
        /// </summary>
        public double Lambda1 { get; private set; }

        /// <summary>
        /// Gets hyper-parameter : inverse scale parameter (rate parameter) for the
        /// Gamma distribution prior over the lambda parameter.
        /// </summary>
        public double Lambda2 { get; private set; }
        
        /// <summary>
        /// Gets Hyper-parameter : shape parameter for the Gamma distribution prior
        /// over the alpha parameter.
        /// </summary>
        public double Alpha1 { get; private set; }
        
        /// <summary>
        /// Gets Hyper-parameter : inverse scale parameter (rate parameter) for the
        /// Gamma distribution prior over the alpha parameter.
        /// </summary>
        public double Alpha2 { get; private set; }
        
        /// <summary>
        /// Gets precision.
        /// </summary>
        public double Tol { get; private set; }

        /// <summary>
        /// Gets maximum number of iterations.
        /// </summary>
        public int NumIter { get; private set; }

        /// <summary>
        /// Gets a value indicating whether verbose mode when fitting the model.
        /// </summary>
        public bool Verbose { get; private set; }

        /// <summary>
        /// Gets estimated precision of the noise.
        /// </summary>
        public double Alpha { get; private set; }

        /// <summary>
        /// Gets estimated precisions of the weights.
        /// (nFeatures).
        /// </summary>
        public double Lambda { get; private set; }

        /// <summary>
        /// Gets value of the objective function (to be maximized), If computed.
        /// </summary>
        public List<double> Scores { get; private set; }

        /// <summary>
        /// Fit linear model.
        /// </summary>
        /// <param name="x">Matrix of shape [n_samples,n_features]. Training data</param>
        /// <param name="y">Target values.[n_samples]</param>
        /// <param name="sampleWeight">Sample weights.[n_samples]</param>
        /// <returns>Instance of self.</returns>
        public override LinearRegressor Fit(Matrix<double> x, Matrix<double> y, Vector<double> sampleWeight = null)
        {
            if (y.ColumnCount != 1)
            {
                throw new ArgumentException("This classifier supports only one target.");
            }

            if (sampleWeight != null)
            {
                throw new ArgumentException("Sample weights are not supported by this classifier.");
            }

            var t = CenterData(x, y, FitIntercept, Normalize, sampleWeight);
            int nSamples = t.X.RowCount;
            int nFeatures = t.X.ColumnCount;

            // Initialization of the values of the parameters
            double alpha = 1.0 / t.Y.Column(0).PopulationVariance();
            double lambda = 1.0;

            Scores = new List<double>();
            Vector<double> coefOld = null;
            Vector<double> coef = null;

            Matrix<double> xTy = t.X.TransposeThisAndMultiply(t.Y);
            var svd = t.X.Svd(true);
            // U, S, Vh = linalg.svd(X, full_matrices=False)
            int k = Math.Min(t.X.ColumnCount, t.X.RowCount);
            var U = svd.U().SubMatrix(0, t.X.RowCount, 0, k);
            var S = svd.S().SubVector(0, k);
            var Vh = svd.VT().SubMatrix(0, k, 0, t.X.ColumnCount);

            Vector<double> eigenVals = S.Sqr();

            // Convergence loop of the bayesian ridge regression)
            for (int iter = 0; iter < this.NumIter; iter++)
            {
                // Compute mu and sigma
                // sigma_ = lambda_ / alpha_ * np.eye(n_features) + np.dot(X.T, X)
                // coef_ = sigma_^-1 * XT * y
                double logdetSigma = 0.0;
                if (nSamples > nFeatures)
                {
                    var coef1 = Vh.TransposeThisAndMultiply(Vh.DivColumnVector(eigenVals + (lambda / alpha)));
                    coef = (coef1 * xTy).Column(0);
                    if (this.ComputeScore)
                    {
                        logdetSigma = -(lambda + (alpha * eigenVals)).Log().Sum();
                    }
                }
                else
                {
                    var coef1 = t.X.TransposeThisAndMultiply(
                        U.DivRowVector(eigenVals + (lambda / alpha)) * U.Transpose());

                    coef = (coef1 * t.Y).Column(0);
                    if (this.ComputeScore)
                    {
                        var logdetSigma1 = lambda * DenseVector.Create(nFeatures, i => 1.0);
                        logdetSigma1.SetSubVector(
                            0,
                            nSamples,
                            logdetSigma1.SubVector(0, nSamples) + (alpha * eigenVals));

                        logdetSigma = -logdetSigma1.Log().Sum();
                    }
                }

                // Update alpha and lambda
                double rmse = (t.Y.Column(0) - (t.X * coef)).Sqr().Sum();
                double gamma = ((alpha * eigenVals) / (lambda + (alpha * eigenVals))).Sum();
                lambda = (gamma + (2 * Lambda1))
                           / (coef.Sqr().Sum() + (2 * Lambda2));
                alpha = (nSamples - gamma + (2 * Alpha1))
                          / (rmse + (2 * Alpha2));

                // Compute the objective function
                if (this.ComputeScore)
                {
                    double s = (Lambda1 * Math.Log(lambda)) - (Lambda2 * lambda);
                    s += (Alpha1 * Math.Log(alpha)) - (Alpha2 * alpha);
                    s += 0.5 * ((nFeatures * Math.Log(lambda))
                              + (nSamples * Math.Log(alpha))
                              - (alpha * rmse)
                              - (lambda * coef.Sqr().Sum())
                              - logdetSigma
                              - (nSamples * Math.Log(2 * Math.PI)));
                    this.Scores.Add(s);
                }

                // Check for convergence
                if (iter != 0 && (coefOld - coef).Abs().Sum() < this.Tol)
                {
                    if (Verbose)
                    {
                        Console.WriteLine("Convergence after {0} iterations", iter);
                    }

                    break;
                }

                coefOld = coef.Clone();
            }

            this.Alpha = alpha;
            this.Lambda = lambda;
            this.Coef = coef.ToRowMatrix();

            this.SetIntercept(t.xMean, t.yMean, t.xStd);
            return this;
        }
    }
}
