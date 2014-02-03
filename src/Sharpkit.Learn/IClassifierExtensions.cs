// -----------------------------------------------------------------------
// <copyright file="IClassifierExtensions.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.LinearAlgebra.Generic;

namespace Sharpkit.Learn
{
    using System;

    /// <summary>
    /// TODO: Update summary.
    /// </summary>
    public static class IClassifierExtensions
    {
        /// <summary>
        /// Predict class labels for samples in X.
        /// </summary>
        /// <param name="x">[n_samples, n_features] Samples.</param>
        /// <returns>[n_samples] Predicted class label per sample.</returns>
        public static TLabel Predict<TLabel>(this IClassifier<TLabel> classifier, double[] x)
        {
            return classifier.Predict(x.ToDenseVector().ToRowMatrix())[0];
        }

        public static TLabel[] Predict<TLabel>(this IClassifier<TLabel> classifier, double[,] x)
        {
            return classifier.Predict(x.ToDenseMatrix());
        }

        public static Matrix<double> PredictProba<TLabel>(this IClassifier<TLabel> classifier, double[,] x)
        {
            return classifier.PredictProba(x.ToDenseMatrix());
        }

        public static void Fit<TLabel>(this IClassifier<TLabel> classifier, double[,] x, TLabel[] y, double[] sampleWeight = null)
        {
            classifier.Fit(x.ToDenseMatrix(), y, sampleWeight == null ? null : DenseVector.OfEnumerable(sampleWeight));
        }

        public static void Fit<TLabel>(this IClassifier<TLabel> classifier, Matrix<double> x, TLabel[] y, double[] sampleWeight = null)
        {
            classifier.Fit(x, y, sampleWeight == null ? null : DenseVector.OfEnumerable(sampleWeight));
        }

        /// <summary>
        /// Returns the mean accuracy on the given test data and labels.
        /// </summary>
        /// <typeparam name="TLabel"></typeparam>
        /// <param name="classifier"></param>
        /// <param name="X">(n_samples, n_features) Test samples.</param>
        /// <param name="y">(n_samples,) True labels for X.</param>
        /// <returns>Mean accuracy of self.predict(X) wrt. y.</returns>
        public static double Score<TLabel>(this IClassifier<TLabel> classifier, Matrix<double> X, TLabel[] y)
            where TLabel : IEquatable<TLabel>
        {
            return Metrics.Metrics.AccuracyScore(y, classifier.Predict(X));
        }

        /// <summary>
        /// Calculates log of probability estimates.
        /// The returned estimates for all classes are ordered by the
        /// label of classes.
        /// </summary>
        /// <param name="x">[nSamples, nFeatures]. Samples.</param>
        /// <returns>[nSamples, nClasses] Log-probability of the sample for each class in the
        /// model, where classes are ordered as they are in <see cref="Classes"/>.
        /// </returns>
        public static Matrix<double> PredictLogProba<TLabel>(this IClassifier<TLabel> classifier, Matrix<double> x)
        {
            return classifier.PredictProba(x).Log();
        }
    }
}
