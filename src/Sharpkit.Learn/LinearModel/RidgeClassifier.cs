namespace Sharpkit.Learn.LinearModel
{
    using System;
    using System.Linq;
    using MathNet.Numerics.LinearAlgebra.Double;
    using Preprocessing;
    using MathNet.Numerics.LinearAlgebra.Generic;

    /// <summary>
    /// Classifier using Ridge regression.
    /// </summary>
    /// <remarks>
    /// For multi-class classification, n_class classifiers are trained in
    /// a one-versus-all approach. Concretely, this is implemented by taking
    /// advantage of the multi-variate response support in Ridge.
    /// </remarks>
    /// <typeparam name="TLabel"></typeparam>
    public class RidgeClassifier<TLabel> : LinearClassifier<TLabel> where TLabel:IEquatable<TLabel>
    {
        private LabelBinarizer<TLabel> labelBinarizer;
        private readonly RidgeBase ridgeBase;

        /// <summary>
        /// Initializes a new instance of the RidgeClassifier class.
        /// </summary>
        /// <param name="alpha"> Small positive values of alpha improve the conditioning of the problem
        /// and reduce the variance of the estimates.  Alpha corresponds to
        /// ``(2*C)^-1`` in other linear models such as LogisticRegression or
        /// LinearSVC.</param>
        /// <param name="fitIntercept">Whether to calculate the intercept for this model. If set to false, no
        /// intercept will be used in calculations (e.g. data is expected to be
        /// already centered).</param>
        /// <param name="normalize">If True, the regressors X will be normalized before regression.</param>
        /// <param name="maxIter">Maximum number of iterations for conjugate gradient solver.
        /// The default value is determined by Math.Net.</param>
        /// <param name="tol">Precision of the solution.</param>
        /// <param name="classWeight">Weights associated with classes in the form
        /// {class_label : weight}. If not given, all classes are
        /// supposed to have weight one.</param>
        /// <param name="solver">Solver to use in the computational.</param>
        public RidgeClassifier(
            double alpha = 1.0,
            bool fitIntercept = true,
            bool normalize = false,
            int? maxIter = null,
            double tol = 1e-3,
            ClassWeight<TLabel> classWeight = null,
            RidgeSolver solver = RidgeSolver.Auto) : base(fitIntercept, classWeight)
        {
            if (classWeight == ClassWeight<TLabel>.Auto)
            {
                throw new ArgumentException("ClassWeight.Auto is not supported.");
            }

            this.ridgeBase = new RidgeBase(this, alpha, normalize, maxIter, tol, solver);
        }

        /// <summary>
        /// Fit Ridge regression model.
        /// </summary>
        /// <param name="x">[n_samples,n_features]. Training data</param>
        /// <param name="y">Target values.</param>
        /// <returns>Instance of self.</returns>
        public LinearModel Fit(double[,] x, TLabel[] y)
        {
            return this.Fit(x.ToDenseMatrix(), y);
        }

        /// <summary>
        /// Fit Ridge regression model.
        /// </summary>
        /// <param name="x">[n_samples,n_features]. Training data</param>
        /// <param name="y">Target values.</param>
        /// <returns>Instance of self.</returns>
        public LinearModel Fit(Matrix<double> x, TLabel[] y)
        {
            this.labelBinarizer = new LabelBinarizer<TLabel>(posLabel : 1, negLabel : -1);
            Matrix Y = this.labelBinarizer.Fit(y).Transform(y);
            // second parameter is used only for ClassWeight.Auto, which we don't support here.
            // So fake it.
            Vector cw = this.ClassWeight.ComputeWeights(this.Classes, new int[0]);
            //# get the class weight corresponding to each sample
            Vector sampleWeightClasses = y.Select(v => cw[Array.BinarySearch(this.Classes, v)]).ToArray().ToDenseVector();
            this.ridgeBase.Fit(x, Y, sampleWeight : sampleWeightClasses);
            return this;
        }

        public override TLabel[] Classes
        {
            get { return this.labelBinarizer.Classes; }
        }
    }
}