// -----------------------------------------------------------------------
// <copyright file="RidgeClassifier.cs" company="Sharpkit.Learn">
// Author: Mathieu Blondel <mathieu@mblondel.org>
//         Reuben Fletcher-Costin <reuben.fletchercostin@gmail.com>
//         Fabian Pedregosa <fabian@fseoane.net>
//  License: BSD 3 clause
// </copyright>
// -----------------------------------------------------------------------

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
    public class RidgeClassifier<TLabel> : RidgeBase, IClassifier<TLabel> where TLabel:IEquatable<TLabel>
    {
        private LabelBinarizer<TLabel> labelBinarizer;

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
        /// <param name="classWeightEstimator">Weights associated with classes in the form
        /// {class_label : weight}. If not given, all classes are
        /// supposed to have weight one.</param>
        /// <param name="solver">Solver to use in the computational.</param>
        public RidgeClassifier(
            double alpha = 1.0,
            bool fitIntercept = true,
            bool normalize = false,
            int? maxIter = null,
            double tol = 1e-3,
            ClassWeightEstimator<TLabel> classWeightEstimator = null,
            RidgeSolver solver = RidgeSolver.Auto) : base(fitIntercept, alpha, normalize, maxIter, tol, solver)
        {
            if (classWeightEstimator == ClassWeightEstimator<TLabel>.Auto)
            {
                throw new ArgumentException("ClassWeight.Auto is not supported.");
            }

            this.ClassWeightEstimator = classWeightEstimator ?? ClassWeightEstimator<TLabel>.Uniform;
        }

        /// <summary>
        /// Fit Ridge regression model.
        /// </summary>
        /// <param name="x">[n_samples,n_features]. Training data</param>
        /// <param name="y">Target values.</param>
        /// <param name="sampleWeight">Sample weights.</param>
        /// <returns>Instance of self.</returns>
        public void Fit(Matrix<double> x, TLabel[] y, Vector<double> sampleWeight = null)
        {
            if (sampleWeight != null)
            {
                throw new ArgumentException("Sample weights are not supported by the classifier");
            }

            this.labelBinarizer = new LabelBinarizer<TLabel>(posLabel : 1, negLabel : -1);
            Matrix<double> Y = this.labelBinarizer.Fit(y).Transform(y);
            // second parameter is used only for ClassWeight.Auto, which we don't support here.
            // So fake it.
            Vector cw = this.ClassWeightEstimator.ComputeWeights(this.Classes, new int[0]);
            //# get the class weight corresponding to each sample
            Vector sampleWeightClasses = y.Select(v => cw[Array.BinarySearch(this.Classes, v)]).ToArray().ToDenseVector();
            base.Fit(x, Y, sampleWeight: sampleWeightClasses);
        }

        /// <summary>
        /// Perform classification on samples in X.
        /// For an one-class model, +1 or -1 is returned.
        /// </summary>
        /// <param name="x">[nSamples, nFeatures]. Samples.</param>
        /// <returns>[nSamples] Class labels for samples in <paramref name="x"/>.</returns>
        public TLabel[] Predict(Matrix<double> x)
        {
            var scores = this.DecisionFunction(x);
            if (scores.ColumnCount == 1)
            {
                return scores.Column(0).Select(v => v > 0 ? Classes[1] : Classes[0]).ToArray();
            }
            else
            {
                return scores.RowEnumerator().Select(r => Classes[r.Item2.MaximumIndex()]).ToArray();
            }
        }

        /// <summary>
        /// Calculates probability estimates.
        /// The returned estimates for all classes are ordered by the
        /// label of classes.
        /// </summary>
        /// <param name="x">[nSamples, nFeatures]. Samples.</param>
        /// <returns>
        /// [nSamples, nClasses]. The probability of the sample for each class in the model,
        /// where classes are ordered as they are in <see cref="IClassifier{TLabel}.Classes"/>.
        /// </returns>
        public Matrix<double> PredictProba(Matrix<double> x)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Gets ordered list of class labeled discovered int <see cref="IClassifier{TLabel}.Fit"/>.
        /// </summary>
        public TLabel[] Classes
        {
            get { return this.labelBinarizer.Classes; }
        }

        /// <summary>
        /// Gets or sets class weight estimator.
        /// </summary>
        public ClassWeightEstimator<TLabel> ClassWeightEstimator { get; private set; }
    }
}